// RecycleLM Playground backend.
//
// Modes:
//   batch  : N branches, one per prompt (default)
//   serial : each prompt alone (true N=1 baseline)
//   both   : serial baseline, then the batch
//   sweep  : first prompt alone, then replicated at N=1..sweep-max
//
// Writes one JSON object to --out (and stdout).

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

static std::string esc(const std::string & s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:   o += c;
        }
    }
    return o;
}

static std::string unescape(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if (c == 'n') { o += '\n'; ++i; }
            else if (c == 't') { o += '\t'; ++i; }
            else if (c == 'r') { o += '\r'; ++i; }
            else if (c == '\\') { o += '\\'; ++i; }
            else o += s[i];
        } else o += s[i];
    }
    return o;
}

struct Item { std::string id, prompt; };

static bool         g_record_probs = false;
static std::string  g_probs_path;

struct RunOut {
    parallel::RunResult res;
    double wall_ms = 0.0;
    bool ok = false;
    std::string err;
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

static parallel::Branch make_branch(parallel::Runtime & rt, int id, const Item & it,
                                    const std::string & sys, int max_new,
                                    float temp, float top_p, uint32_t seed) {
    parallel::Branch b;
    b.id = std::to_string(id);
    b.suffix = rt.format_chat({ { "system", sys }, { "user", it.prompt } });
    b.max_new_tokens = max_new;
    b.role = parallel::Role::Auxiliary;
    b.sampling.temperature = temp;
    b.sampling.top_k = 20;
    b.sampling.top_p = top_p;
    b.sampling.seed = seed;
    b.record_probs = g_record_probs;
    return b;
}

static RunOut run_batch(parallel::Runtime & rt, const std::vector<Item> & items,
                        const std::string & sys, int max_new, float temp, float top_p,
                        uint32_t seed) {
    parallel::RunRequest req;
    req.shared_prefix = "";
    req.agree_k = 0;
    for (size_t i = 0; i < items.size(); ++i)
        req.branches.push_back(make_branch(rt, (int)i + 1, items[i], sys, max_new,
                                           temp, top_p, seed + (uint32_t)i));
    return run_req(rt, req);
}

static RunOut run_replicated(parallel::Runtime & rt, const Item & it, int n,
                             const std::string & sys, int max_new, float temp,
                             float top_p, uint32_t seed) {
    parallel::RunRequest req;
    req.shared_prefix = "";
    req.agree_k = 0;
    for (int i = 0; i < n; ++i)
        req.branches.push_back(make_branch(rt, i + 1, it, sys, max_new,
                                           temp, top_p, seed + (uint32_t)i));
    return run_req(rt, req);
}

static std::string stream_json(const parallel::BranchResult & b, const Item & it,
                               size_t idx) {
    const double gen_s = b.completion_ms > 0 ? (b.completion_ms / 1000.0) : 0.0;
    const double tok_s = gen_s > 0 ? (b.generated_tokens / gen_s) : 0.0;
    std::string js = "{\"id\":\"" + esc(it.id) + "\"";
    js += ",\"text\":\"" + esc(b.text) + "\"";
    js += ",\"generated\":" + std::to_string(b.generated_tokens);
    js += ",\"ttft_ms\":" + std::to_string(b.ttft_ms);
    js += ",\"completion_ms\":" + std::to_string(b.completion_ms);
    js += ",\"tok_s\":" + std::to_string(tok_s);
    js += ",\"finish\":\"" + std::string(parallel::to_string(b.finish)) + "\"";
    js += ",\"prompt\":\"" + esc(idx < 1 ? it.prompt : it.prompt) + "\"}";
    return js;
}

static std::string sweep_json(int n, const RunOut & out) {
    double sum_comp = 0, max_comp = 0, sum_ttft = 0, sum_tok = 0;
    long long gen = 0;
    for (const auto & b : out.res.branches) {
        sum_comp += b.completion_ms;
        max_comp = std::max(max_comp, b.completion_ms);
        sum_ttft += b.ttft_ms;
        const double gs = b.completion_ms > 0 ? b.completion_ms / 1000.0 : 0.0;
        sum_tok += gs > 0 ? (b.generated_tokens / gs) : 0.0;
        gen += b.generated_tokens;
    }
    const double m = out.res.branches.empty() ? 1.0 : (double)out.res.branches.size();
    const double wall_s = out.wall_ms / 1000.0;
    const double aggregate = wall_s > 0 ? (gen / wall_s) : 0.0;
    std::string js = "{\"n\":" + std::to_string(n);
    js += ",\"wall_ms\":" + std::to_string(out.wall_ms);
    js += ",\"prefill_ms\":" + std::to_string(out.res.prefill_ms);
    js += ",\"avg_completion_ms\":" + std::to_string(sum_comp / m);
    js += ",\"max_completion_ms\":" + std::to_string(max_comp);
    js += ",\"avg_ttft_ms\":" + std::to_string(sum_ttft / m);
    js += ",\"avg_tok_s\":" + std::to_string(sum_tok / m);
    js += ",\"aggregate_tok_s\":" + std::to_string(aggregate);
    js += ",\"tokens\":" + std::to_string(gen);
    js += ",\"joint_steps\":" + std::to_string(out.res.joint_steps);
    js += ",\"steps\":" + std::to_string(out.res.steps);
    js += "}";
    return js;
}

int main(int argc, char ** argv) {
    std::string model, data, out, sys = "You are a helpful assistant.";
    std::string mode = "batch";
    int max_new = 256, limit = 0, sweep_max = 10;
    float temp = 0.7f, top_p = 0.9f;
    uint32_t seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--out") out = nx();
        else if (k == "--system") sys = nx();
        else if (k == "--mode") mode = nx();
        else if (k == "--max-tokens") max_new = std::atoi(nx().c_str());
        else if (k == "--limit") limit = std::atoi(nx().c_str());
        else if (k == "--sweep-max") sweep_max = std::atoi(nx().c_str());
        else if (k == "--temperature") temp = std::strtof(nx().c_str(), nullptr);
        else if (k == "--top-p") top_p = std::strtof(nx().c_str(), nullptr);
        else if (k == "--seed") seed = (uint32_t)std::strtoul(nx().c_str(), nullptr, 10);
        else if (k == "--probs") { g_probs_path = nx(); g_record_probs = true; }
        else { std::fprintf(stderr, "parallel_cli: unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || data.empty() || out.empty()) {
        std::fprintf(stderr, "parallel_cli: --model --data --out are required\n");
        return 2;
    }
    if (mode != "batch" && mode != "serial" && mode != "both" && mode != "sweep") {
        std::fprintf(stderr, "parallel_cli: --mode batch|serial|both|sweep\n");
        return 2;
    }
    sweep_max = std::max(1, std::min(16, sweep_max));

    std::vector<Item> items;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "parallel_cli: cannot open %s\n", data.c_str()); return 1; }
        std::string line;
        int lineno = 0, row = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const size_t t = line.find('\t');
            if (t == std::string::npos) continue;
            Item it;
            it.id = line.substr(0, t);
            it.prompt = unescape(line.substr(t + 1));
            if (lineno == 1 && it.id == "id") continue;
            it.id = std::to_string(++row);
            items.push_back(std::move(it));
        }
    }
    if (items.empty()) { std::fprintf(stderr, "parallel_cli: no prompts\n"); return 1; }
    if (limit > 0 && limit < (int)items.size()) items.resize((size_t)limit);
    if (items.size() > 16) items.resize(10);

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    parallel::RuntimeConfig rcfg;
    rcfg.model_path = model;
    rcfg.max_context_tokens = 8192;
    rcfg.max_branches = 16;
    rcfg.n_gpu_layers = 99;
    rcfg.flash_attn = true;
    parallel::Runtime rt(rcfg);
    if (!rt.ok()) {
        std::fprintf(stderr, "parallel_cli: runtime init failed: %s\n", rt.error().c_str());
        return 1;
    }

    std::string streams = "[]", serial = "[]", sweep = "[]";
    RunOut batch;
    std::vector<parallel::BranchResult> serial_brs;
    double batch_wall = 0.0;
    if (mode == "batch" || mode == "both") {
        batch = run_batch(rt, items, sys, max_new, temp, top_p, seed);
        if (!batch.ok) { std::fprintf(stderr, "parallel_cli: %s\n", batch.err.c_str()); return 1; }
        batch_wall = batch.wall_ms;
        streams = "[";
        for (size_t i = 0; i < batch.res.branches.size(); ++i) {
            if (i) streams += ",";
            streams += stream_json(batch.res.branches[i], items[i], i);
        }
        streams += "]";
    }
    if (mode == "serial" || mode == "both") {
        serial = "[";
        for (size_t i = 0; i < items.size(); ++i) {
            RunOut one = run_batch(rt, { items[i] }, sys, max_new, temp, top_p, seed + (uint32_t)i);
            if (!one.ok) { std::fprintf(stderr, "parallel_cli: serial %zu: %s\n", i, one.err.c_str()); return 1; }
            if (i) serial += ",";
            serial += stream_json(one.res.branches[0], items[i], i);
            serial_brs.push_back(one.res.branches[0]);
        }
        serial += "]";
    }
    if (mode == "sweep") {
        sweep = "[";
        for (int n = 1; n <= sweep_max; ++n) {
            RunOut o = run_replicated(rt, items[0], n, sys, max_new, temp, top_p, seed);
            if (!o.ok) { std::fprintf(stderr, "parallel_cli: sweep n=%d: %s\n", n, o.err.c_str()); return 1; }
            if (n > 1) sweep += ",";
            sweep += sweep_json(n, o);
        }
        sweep += "]";
    }

    // Optional raw-logit sidecar: one JSON object per emitted token.
    // m = mode, s = stream id, H = full-vocab entropy (nats, raw logits),
    // p1 = top-1 prob, mg = p1-p2, lp = emitted-token log-prob, off/len =
    // byte span of the emitted piece in that stream's text.
    if (g_record_probs && !g_probs_path.empty()) {
        FILE * pf = std::fopen(g_probs_path.c_str(), "wb");
        if (!pf) { std::fprintf(stderr, "parallel_cli: cannot write %s\n", g_probs_path.c_str()); return 1; }
        auto dump = [&](const char * mode_name, const std::vector<parallel::BranchResult> & brs) {
            for (const auto & b : brs)
                for (const auto & r : b.probs)
                    std::fprintf(pf,
                        "{\"m\":\"%s\",\"s\":\"%s\",\"step\":%d,\"pos\":%d,"
                        "\"tok\":%d,\"H\":%.5f,\"p1\":%.6f,\"mg\":%.6f,"
                        "\"lp\":%.5f,\"off\":%d,\"len\":%d}\n",
                        mode_name, b.id.c_str(), r.step, r.pos, r.token,
                        r.entropy, r.p1, r.margin, r.logp, r.text_off, r.text_len);
        };
        dump("batch", batch.res.branches);
        dump("serial", serial_brs);
        std::fclose(pf);
    }

    std::string js = "{\"ok\":true,\"mode\":\"" + mode + "\"";
    js += ",\"n\":" + std::to_string(items.size());
    js += ",\"wall_ms\":" + std::to_string(batch_wall);
    js += ",\"streams\":" + streams;
    js += ",\"serial\":" + serial;
    js += ",\"sweep\":" + sweep;
    js += "}";

    FILE * f = std::fopen(out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "parallel_cli: cannot write %s\n", out.c_str()); return 1; }
    std::fwrite(js.data(), 1, js.size(), f);
    std::fclose(f);
    std::printf("%s\n", js.c_str());
    return 0;
}
