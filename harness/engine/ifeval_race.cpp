// RecycleLM IFEval continuous-batching harness.
//
// One shared prefill per prompt. Modes:
//   baseline : one branch, seed S + idx
//   serialA  : one branch, seed S + idx (alias used by the retry driver)
//   serialB  : one branch, seed S + idx (driver passes S = base + 10000)
//   race     : two branches from one shared prefill, seeds S + idx and
//              S + idx + 10000, agree_k = 0
//
// Sampling matches harness/engine/ifeval.cpp: temp 0.6, top_k 20, top_p 0.95,
// dist seed. Chat format: system "You are a helpful assistant. Follow all
// instructions exactly."; user = prompt. The runtime appends the assistant
// header to the shared prefix.
//
// Input TSV: key <TAB> prompt [<TAB> idx]
//   Escapes: \t \n \r \\ (same convention as the other harness data files).
//   The optional third column preserves the official prompt index, which sets
//   the sampling seed (S + idx).
//
// Output CSV:
//   key,idx,mode,seed_a,seed_b,response_a,response_b,finish_a,finish_b,
//   n_tokens_a,n_tokens_b,all_done_ms,prefill_ms,joint_steps,steps,
//   active_hist,wall_ms,shared_prefix_tokens
//
// active_hist is "n:ticks" pairs joined by "|" (zero bins omitted).
//
// Usage:
//   ifeval_race.exe --model models\Llama-3.2-3B-Instruct-Q4_K_M.gguf \
//                   --data data\ifeval_541.tsv --mode race --seed 1234 \
//                   --max-tokens 1024 --out results\ifeval_race_smoke_raw.csv \
//                   [--keys 1000,1001] [--limit 50] [--verbose]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "parallel/runtime.hpp"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(const Clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::string csv_field(const std::string & s) {
    bool q = false;
    for (char c : s) if (c == ',' || c == '"' || c == '\n' || c == '\r') { q = true; break; }
    if (!q) return s;
    std::string o = "\"";
    for (char c : s) {
        if (c == '"') o += "\"\"";
        else o += c;
    }
    o += '"';
    return o;
}

static std::string sanitize_utf8(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        int len = 0;
        if (c < 0x80) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (len == 0) { o += "\xEF\xBF\xBD"; ++i; continue; }
        if (i + (size_t)len > s.size()) break;
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if (((unsigned char)s[i + (size_t)k] & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { o += "\xEF\xBF\xBD"; ++i; continue; }
        o.append(s, i, (size_t)len);
        i += (size_t)len;
    }
    return o;
}

static std::vector<std::string> split_tab(const std::string & s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find('\t', p);
        if (q == std::string::npos) q = s.size();
        out.push_back(s.substr(p, q - p));
        if (q == s.size()) break;
        p = q + 1;
    }
    return out;
}

static std::string unescape_field(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if      (c == 'n')  { o += '\n'; ++i; }
            else if (c == 't')  { o += '\t'; ++i; }
            else if (c == 'r')  { o += '\r'; ++i; }
            else if (c == '\\') { o += '\\'; ++i; }
            else                { o += s[i]; }
        } else {
            o += s[i];
        }
    }
    return o;
}

struct Item {
    std::string key, prompt;
    int idx = 0;
};

static const char * kSys =
    "You are a helpful assistant. Follow all instructions exactly.";

static std::string finish_name(parallel::FinishReason r) {
    return parallel::to_string(r);
}

static std::string hist_string(const std::vector<long long> & h) {
    std::string o;
    for (size_t i = 0; i < h.size(); ++i) {
        if (h[i] == 0) continue;
        if (!o.empty()) o += '|';
        o += std::to_string(i) + ":" + std::to_string(h[i]);
    }
    return o;
}

int main(int argc, char ** argv) {
    std::string model, data = "data\\ifeval_541.tsv", out, mode = "baseline", keys_arg;
    int max_new = 1024;
    int limit = 0;
    uint32_t seed = 1234;
    bool verbose = false;

    auto need = [&](int i, const char * flag) -> bool {
        if (i + 1 < argc) return true;
        std::fprintf(stderr, "ifeval_race: %s needs a value\n", flag);
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") { if (!need(i, "--model")) return 2; model = argv[++i]; }
        else if (k == "--data") { if (!need(i, "--data")) return 2; data = argv[++i]; }
        else if (k == "--out") { if (!need(i, "--out")) return 2; out = argv[++i]; }
        else if (k == "--mode") { if (!need(i, "--mode")) return 2; mode = argv[++i]; }
        else if (k == "--seed") { if (!need(i, "--seed")) return 2; seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10); }
        else if (k == "--max-tokens") { if (!need(i, "--max-tokens")) return 2; max_new = std::atoi(argv[++i]); }
        else if (k == "--limit") { if (!need(i, "--limit")) return 2; limit = std::atoi(argv[++i]); }
        else if (k == "--keys") { if (!need(i, "--keys")) return 2; keys_arg = argv[++i]; }
        else if (k == "--verbose") verbose = true;
        else { std::fprintf(stderr, "ifeval_race: unknown argument: %s\n", k.c_str()); return 2; }
    }

    const bool race = (mode == "race");
    if (mode != "baseline" && mode != "race" && mode != "serialA" && mode != "serialB") {
        std::fprintf(stderr, "ifeval_race: --mode must be baseline|race|serialA|serialB\n");
        return 2;
    }
    if (model.empty() || out.empty()) {
        std::fprintf(stderr, "ifeval_race: --model and --out are required\n");
        return 2;
    }
    if (max_new < 1) {
        std::fprintf(stderr, "ifeval_race: --max-tokens must be >= 1\n");
        return 2;
    }

    std::vector<Item> all;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "ifeval_race: cannot open --data: %s\n", data.c_str()); return 1; }
        std::string line;
        int lineno = 0, row = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (lineno == 1 && f[0] == "key") continue;
            if (f.size() < 2) {
                std::fprintf(stderr, "ifeval_race: data line %d has %zu columns (need >= 2)\n",
                             lineno, f.size());
                return 1;
            }
            Item it;
            it.key = unescape_field(f[0]);
            it.prompt = unescape_field(f[1]);
            it.idx = (f.size() >= 3 && !f[2].empty()) ? std::atoi(f[2].c_str()) : row;
            all.push_back(std::move(it));
            ++row;
        }
    }
    if (all.empty()) { std::fprintf(stderr, "ifeval_race: no items read from %s\n", data.c_str()); return 1; }

    if (!keys_arg.empty()) {
        std::vector<std::string> keep;
        size_t p = 0;
        while (p <= keys_arg.size()) {
            size_t q = keys_arg.find(',', p);
            if (q == std::string::npos) q = keys_arg.size();
            std::string k = keys_arg.substr(p, q - p);
            if (!k.empty()) keep.push_back(k);
            if (q == keys_arg.size()) break;
            p = q + 1;
        }
        std::vector<Item> filtered;
        for (const auto & it : all)
            if (std::find(keep.begin(), keep.end(), it.key) != keep.end()) filtered.push_back(it);
        if (filtered.empty()) { std::fprintf(stderr, "ifeval_race: --keys matched no rows\n"); return 1; }
        if (filtered.size() != keep.size()) {
            std::fprintf(stderr, "ifeval_race: --keys matched %zu/%zu requested keys\n",
                         filtered.size(), keep.size());
        }
        all.swap(filtered);
    }
    if (limit > 0 && limit < (int)all.size()) all.resize((size_t)limit);

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    parallel::RuntimeConfig rcfg;
    rcfg.model_path = model;
    rcfg.max_context_tokens = 8192;
    rcfg.max_branches = race ? 2 : 1;
    rcfg.n_gpu_layers = 99;
    rcfg.flash_attn = true;

    parallel::Runtime rt(rcfg);
    if (!rt.ok()) {
        std::fprintf(stderr, "ifeval_race: runtime init failed: %s\n", rt.error().c_str());
        return 1;
    }

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "ifeval_race: cannot open --out: %s\n", out.c_str());
        return 1;
    }
    std::fprintf(f,
                 "key,idx,mode,seed_a,seed_b,response_a,response_b,"
                 "finish_a,finish_b,n_tokens_a,n_tokens_b,all_done_ms,"
                 "prefill_ms,joint_steps,steps,active_hist,wall_ms,"
                 "shared_prefix_tokens\n");

    std::printf("model=%s\n", model.c_str());
    std::printf("input=%s rows=%d mode=%s max_tokens=%d seed=%u\n",
                data.c_str(), (int)all.size(), mode.c_str(), max_new, seed);
    std::fflush(stdout);

    const auto t_run0 = Clock::now();
    int n_eos = 0, n_max = 0;
    long long sum_steps = 0, sum_joint = 0;

    for (int ci = 0; ci < (int)all.size(); ++ci) {
        const Item & it = all[(size_t)ci];
        const uint32_t seed_a = seed + (uint32_t)it.idx;
        const uint32_t seed_b = seed_a + 10000u;

        const std::string prefix = rt.format_chat({ { "system", kSys }, { "user", it.prompt } });
        if (prefix.empty()) {
            std::fprintf(stderr, "ifeval_race: chat template failed (%s)\n", it.key.c_str());
            std::fclose(f);
            return 1;
        }

        parallel::RunRequest req;
        req.shared_prefix = prefix;
        req.agree_k = 0;

        auto make_branch = [&](const char * id, uint32_t s) {
            parallel::Branch b;
            b.id = id;
            b.suffix = "";
            b.max_new_tokens = max_new;
            b.role = parallel::Role::Auxiliary;
            b.sampling.temperature = 0.6f;
            b.sampling.top_k = 20;
            b.sampling.top_p = 0.95f;
            b.sampling.seed = s;
            return b;
        };
        req.branches.push_back(make_branch("A", seed_a));
        if (race) req.branches.push_back(make_branch("B", seed_b));

        std::string err;
        const auto t0 = Clock::now();
        parallel::Run run = rt.submit(req);
        while (auto e = run.next()) {
            if (e->kind == parallel::Event::Error) err = e->message;
        }
        parallel::RunResult res = run.result();
        const double wall_ms = ms_since(t0);
        if (!err.empty()) {
            std::fprintf(stderr, "ifeval_race: run failed (%s): %s\n", it.key.c_str(), err.c_str());
            std::fclose(f);
            return 1;
        }
        if (res.branches.empty()) {
            std::fprintf(stderr, "ifeval_race: no branch result (%s)\n", it.key.c_str());
            std::fclose(f);
            return 1;
        }

        const parallel::BranchResult * a = nullptr;
        const parallel::BranchResult * b = nullptr;
        for (const auto & br : res.branches) {
            if (br.id == "A") a = &br;
            else if (br.id == "B") b = &br;
        }
        if (!a) {
            std::fprintf(stderr, "ifeval_race: branch A missing (%s)\n", it.key.c_str());
            std::fclose(f);
            return 1;
        }

        const std::string ra = sanitize_utf8(a->text);
        const std::string rb = b ? sanitize_utf8(b->text) : std::string();
        const std::string fa = finish_name(a->finish);
        const std::string fb = b ? finish_name(b->finish) : std::string();
        const std::string sa = std::to_string(seed_a);
        const std::string sb = race ? std::to_string(seed_b) : std::string();
        const std::string na = std::to_string(a->generated_tokens);
        const std::string nb = b ? std::to_string(b->generated_tokens) : std::string();
        const std::string hist = hist_string(res.active_hist);

        std::fprintf(f,
                     "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f,%.3f,%lld,%lld,%s,%.3f,%d\n",
                     csv_field(it.key).c_str(), it.idx, mode.c_str(),
                     sa.c_str(), sb.c_str(),
                     csv_field(ra).c_str(), csv_field(rb).c_str(),
                     fa.c_str(), fb.c_str(),
                     na.c_str(), nb.c_str(),
                     res.all_done_ms, res.prefill_ms,
                     res.joint_steps, res.steps, hist.c_str(),
                     wall_ms, res.shared_prefix_tokens);
        std::fflush(f);

        if (fa == "eos") ++n_eos;
        if (fa == "token_limit") ++n_max;
        sum_steps += res.steps;
        sum_joint += res.joint_steps;

        std::printf("[%3d/%3d] key=%s wall_ms=%.0f prefill_ms=%.0f steps=%lld joint=%lld\n",
                    ci + 1, (int)all.size(), it.key.c_str(), wall_ms,
                    res.prefill_ms, res.steps, res.joint_steps);
        if (verbose) {
            std::printf("  A: n=%d finish=%s\n", a->generated_tokens, fa.c_str());
            std::printf("  A text: \"%s\"\n", ra.c_str());
            if (b) {
                std::printf("  B: n=%d finish=%s\n", b->generated_tokens, fb.c_str());
                std::printf("  B text: \"%s\"\n", rb.c_str());
            }
        }
        std::fflush(stdout);
    }

    const double total_s = ms_since(t_run0) / 1000.0;
    std::printf("wrote %s\n", out.c_str());
    std::printf("rows=%d eos=%d max=%d steps=%lld joint_steps=%lld wall_s=%.1f\n",
                (int)all.size(), n_eos, n_max, sum_steps, sum_joint, total_s);
    std::fclose(f);
    return 0;
}
