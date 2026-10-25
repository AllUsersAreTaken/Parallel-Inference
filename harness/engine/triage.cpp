// RecycleLM dual-stream + referee triage harness (parallel::Runtime).
//
// Per prompt, three runtime runs on one model:
//   baseline : branch A alone
//   pair     : branches A and B from one shared prefix (joint decode)
//   referee  : branch C on [P + A + B], grammar-locked to AGREE or CONFLICT
// The referee is a separate run because it depends on both A and B.
//
// Input TSV: id, category, prompt, answer, keywords, expected_rel
// Output CSV: baseline A, pair A/B, C verdict, stage timings.
//
// Usage:
//   triage.exe --model models\Llama-3.2-3B-Instruct-Q4_K_M.gguf \
//              --data data\nli_150.tsv --out results\triage_bench.csv \
//              --limit 150 --max-tokens 256 --seed 1234

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
    for (char c : s) { if (c == '"') o += "\"\""; else o += c; }
    o += '"';
    return o;
}

static std::string sanitize_utf8(const std::string & s) {
    std::string o; o.reserve(s.size()); size_t i = 0;
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
        for (int k = 1; k < len; ++k)
            if (((unsigned char)s[i + (size_t)k] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { o += "\xEF\xBF\xBD"; ++i; continue; }
        o.append(s, i, (size_t)len); i += (size_t)len;
    }
    return o;
}

static std::vector<std::string> split_tab(const std::string & s) {
    std::vector<std::string> out; size_t p = 0;
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
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if      (c == 'n')  { o += '\n'; ++i; }
            else if (c == 't')  { o += '\t'; ++i; }
            else if (c == 'r')  { o += '\r'; ++i; }
            else if (c == '\\') { o += '\\'; ++i; }
            else                { o += s[i]; }
        } else o += s[i];
    }
    return o;
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

struct Case {
    std::string id, category, prompt, answer, keywords, expected;
    int idx = 0;
};

struct RunOut {
    parallel::RunResult res;
    double wall_ms = 0.0;
    std::string err;
    bool ok = false;
};

static RunOut run_req(parallel::Runtime & rt, const parallel::RunRequest & req) {
    RunOut out;
    const auto t0 = Clock::now();
    parallel::Run run = rt.submit(req);
    while (auto e = run.next()) {
        if (e->kind == parallel::Event::Error) out.err = e->message;
    }
    out.res = run.result();
    out.wall_ms = ms_since(t0);
    out.ok = out.err.empty();
    return out;
}

static const parallel::BranchResult * find_branch(const parallel::RunResult & r,
                                                  const char * id) {
    for (const auto & b : r.branches) if (b.id == id) return &b;
    return nullptr;
}

static const char * kSys = "You are a helpful assistant.";
static const char * kRefSys =
    "Compare Response A and Response B. Do they state the same core facts, or "
    "do they contradict each other on dates, names, locations, or numbers? "
    "Output exactly one word: AGREE or CONFLICT.";

int main(int argc, char ** argv) {
    std::string model, data = "data\\nli_150.tsv", out;
    int max_new = 256, limit = 0;
    uint32_t seed = 1234;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--out") out = nx();
        else if (k == "--max-tokens") max_new = std::atoi(nx().c_str());
        else if (k == "--limit") limit = std::atoi(nx().c_str());
        else if (k == "--seed") seed = (uint32_t)std::strtoul(nx().c_str(), nullptr, 10);
        else if (k == "--verbose") verbose = true;
        else { std::fprintf(stderr, "triage: unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || out.empty()) {
        std::fprintf(stderr, "triage: --model and --out are required\n");
        return 2;
    }
    if (max_new < 1 || limit < 0) {
        std::fprintf(stderr, "triage: bad --max-tokens/--limit\n");
        return 2;
    }

    std::vector<Case> cases;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "triage: cannot open --data: %s\n", data.c_str()); return 1; }
        std::string line;
        int lineno = 0, row = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (lineno == 1 && f[0] == "id") continue;
            if (f.size() < 6) {
                std::fprintf(stderr, "triage: line %d has %zu columns (need 6)\n", lineno, f.size());
                return 1;
            }
            Case c;
            c.id = unescape_field(f[0]);
            c.category = unescape_field(f[1]);
            c.prompt = unescape_field(f[2]);
            c.answer = unescape_field(f[3]);
            c.keywords = unescape_field(f[4]);
            c.expected = unescape_field(f[5]);
            c.idx = row++;
            cases.push_back(std::move(c));
        }
    }
    if (cases.empty()) { std::fprintf(stderr, "triage: no rows in %s\n", data.c_str()); return 1; }
    if (limit > 0 && limit < (int)cases.size()) cases.resize((size_t)limit);

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    parallel::RuntimeConfig rcfg;
    rcfg.model_path = model;
    rcfg.max_context_tokens = 8192;
    rcfg.max_branches = 3;
    rcfg.n_gpu_layers = 99;
    rcfg.flash_attn = true;
    parallel::Runtime rt(rcfg);
    if (!rt.ok()) {
        std::fprintf(stderr, "triage: runtime init failed: %s\n", rt.error().c_str());
        return 1;
    }

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "triage: cannot open --out: %s\n", out.c_str()); return 1; }
    std::fprintf(f,
        "id,category,prompt,answer,keywords,expected,"
        "resp_a_base,resp_a_pair,resp_b_pair,"
        "ttft_a_base_ms,ttft_a_pair_ms,ttft_b_pair_ms,"
        "prefill_base_ms,prefill_pair_ms,wall_base_ms,wall_pair_ms,"
        "steps_base,steps_pair,joint_steps,active_hist,shared_prefix_tokens,"
        "n_a_base,n_a_pair,n_b_pair,finish_a,finish_b,"
        "c_verdict,c_text,c_tokens,c_finish,c_prefill_ms,c_ms,triage_total_ms\n");

    std::printf("model=%s\ninput=%s rows=%d max_tokens=%d seed=%u\n",
                model.c_str(), data.c_str(), (int)cases.size(), max_new, seed);
    std::fflush(stdout);

    const auto t_all0 = Clock::now();
    for (int ci = 0; ci < (int)cases.size(); ++ci) {
        const Case & c = cases[(size_t)ci];
        const uint32_t base_seed = seed + (uint32_t)c.idx;

        auto make_pair = [&](const char * id, uint32_t s) {
            parallel::Branch b;
            b.id = id;
            b.suffix = "";
            b.max_new_tokens = max_new;
            b.role = parallel::Role::Auxiliary;
            b.sampling.temperature = 0.7f;
            b.sampling.top_k = 20;
            b.sampling.top_p = 0.9f;
            b.sampling.seed = s;
            return b;
        };

        const std::string prefix = rt.format_chat({ { "system", kSys }, { "user", c.prompt } });
        if (prefix.empty()) { std::fprintf(stderr, "triage: chat template failed (%s)\n", c.id.c_str()); return 1; }

        // --- Arm 1: A alone
        parallel::RunRequest rb;
        rb.shared_prefix = prefix;
        rb.agree_k = 0;
        rb.branches.push_back(make_pair("A", base_seed));
        RunOut rbase = run_req(rt, rb);
        if (!rbase.ok) { std::fprintf(stderr, "triage: baseline run failed (%s): %s\n", c.id.c_str(), rbase.err.c_str()); return 1; }
        const parallel::BranchResult * ab = find_branch(rbase.res, "A");
        if (!ab) { std::fprintf(stderr, "triage: baseline A missing\n"); return 1; }

        // --- Arm 2: A + B joint
        parallel::RunRequest rp;
        rp.shared_prefix = prefix;
        rp.agree_k = 0;
        rp.branches.push_back(make_pair("A", base_seed));
        rp.branches.push_back(make_pair("B", base_seed + 10000u));
        RunOut rpair = run_req(rt, rp);
        if (!rpair.ok) { std::fprintf(stderr, "triage: pair run failed (%s): %s\n", c.id.c_str(), rpair.err.c_str()); return 1; }
        const parallel::BranchResult * ap = find_branch(rpair.res, "A");
        const parallel::BranchResult * bp = find_branch(rpair.res, "B");
        if (!ap || !bp) { std::fprintf(stderr, "triage: pair branches missing\n"); return 1; }

        const std::string a_pair = sanitize_utf8(ap->text);
        const std::string b_pair = sanitize_utf8(bp->text);

        // --- Arm 3: C on [P + A + B], grammar-locked verdict
        const std::string cuser =
            "Prompt: " + c.prompt + "\nResponse A: " + a_pair +
            "\nResponse B: " + b_pair + "\nVerdict:";
        const std::string cprefix = rt.format_chat({ { "system", kRefSys }, { "user", cuser } });
        if (cprefix.empty()) { std::fprintf(stderr, "triage: C chat template failed\n"); return 1; }

        parallel::Branch cb;
        cb.id = "C";
        cb.suffix = "";
        cb.max_new_tokens = 4;
        cb.role = parallel::Role::Auxiliary;
        cb.sampling.temperature = 0.01f;
        cb.sampling.top_k = 1;
        cb.sampling.top_p = 1.0f;
        cb.sampling.seed = base_seed;
        cb.grammar = "root ::= \"AGREE\" | \"CONFLICT\"\n";

        parallel::RunRequest rc;
        rc.shared_prefix = cprefix;
        rc.agree_k = 0;
        rc.branches.push_back(cb);
        RunOut rcrun = run_req(rt, rc);
        if (!rcrun.ok) { std::fprintf(stderr, "triage: C run failed (%s): %s\n", c.id.c_str(), rcrun.err.c_str()); return 1; }
        const parallel::BranchResult * cc = find_branch(rcrun.res, "C");
        if (!cc) { std::fprintf(stderr, "triage: C branch missing\n"); return 1; }

        const std::string c_text = sanitize_utf8(cc->text);
        const std::string verdict =
            c_text.rfind("AGREE", 0) == 0 ? "AGREE" :
            (c_text.rfind("CONFLICT", 0) == 0 ? "CONFLICT" : c_text);
        const double triage_total = rpair.wall_ms + rcrun.wall_ms;

        std::fprintf(f,
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%lld,%lld,%lld,%s,%d,%d,%d,%d,%s,%s,"
            "%s,%s,%d,%s,%.3f,%.3f,%.3f\n",
            csv_field(c.id).c_str(), csv_field(c.category).c_str(),
            csv_field(c.prompt).c_str(), csv_field(c.answer).c_str(),
            csv_field(c.keywords).c_str(), csv_field(c.expected).c_str(),
            csv_field(sanitize_utf8(ab->text)).c_str(),
            csv_field(a_pair).c_str(), csv_field(b_pair).c_str(),
            ab->ttft_ms, ap->ttft_ms, bp->ttft_ms,
            rbase.res.prefill_ms, rpair.res.prefill_ms,
            rbase.wall_ms, rpair.wall_ms,
            rbase.res.steps, rpair.res.steps, rpair.res.joint_steps,
            hist_string(rpair.res.active_hist).c_str(),
            rpair.res.shared_prefix_tokens,
            ab->generated_tokens, ap->generated_tokens, bp->generated_tokens,
            parallel::to_string(ab->finish), parallel::to_string(bp->finish),
            csv_field(verdict).c_str(), csv_field(c_text).c_str(),
            cc->generated_tokens, parallel::to_string(cc->finish),
            rcrun.res.prefill_ms, rcrun.wall_ms, triage_total);
        std::fflush(f);

        std::printf("[%3d/%3d] %s %-9s base=%.0f pair=%.0f joint=%lld C=%s (%.0f ms)\n",
                    ci + 1, (int)cases.size(), c.id.c_str(), c.category.c_str(),
                    rbase.wall_ms, rpair.wall_ms, rpair.res.joint_steps,
                    verdict.c_str(), rcrun.wall_ms);
        if (verbose) {
            std::printf("  A_base: %s\n", sanitize_utf8(ab->text).c_str());
            std::printf("  A_pair: %s\n", a_pair.c_str());
            std::printf("  B_pair: %s\n", b_pair.c_str());
            std::printf("  C: %s\n", c_text.c_str());
        }
        std::fflush(stdout);
    }

    const double total_s = ms_since(t_all0) / 1000.0;
    std::printf("wrote %s rows=%d wall_s=%.1f\n", out.c_str(), (int)cases.size(), total_s);
    std::fclose(f);
    return 0;
}
