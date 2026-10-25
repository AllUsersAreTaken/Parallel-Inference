// RecycleLM shared-prefix playground backend.
//
// Same modes as parallel_cli, plus a shared prefix:
//   batch  : N branches, one per prompt (default)
//   serial : each prompt alone (true N=1 baseline)
//   both   : serial baseline, then the batch
//   sweep  : first prompt alone, then replicated at N=1..sweep-max
//
// --prefix <path> is read as raw UTF-8 text and sent as RunRequest::shared_prefix.
// Without it the behavior is identical to parallel_cli. The TSV holds per-branch
// suffixes, which are appended after the shared prefix.
//
// --views <path> replaces --data with a four-column TSV:
//   name <tab> instruction <tab> grammar_path <tab> max_tokens
// The instruction is used as the raw branch suffix (no chat formatting), the
// grammar file (if any) is loaded verbatim, and max_tokens overrides
// --max-tokens for that branch. The shared prefix is unchanged.
//
// Writes one JSON object to --out (and stdout).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "parallel/runtime.hpp"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;
    out.assign(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    return true;
}

static std::string dir_of(const std::string & path) {
    const size_t s = path.find_last_of("\\/");
    return s == std::string::npos ? std::string() : path.substr(0, s + 1);
}

struct Item {
    std::string id, prompt, grammar;
    int max_new = 0;
    bool raw_suffix = false;
};

// Four-column views TSV: name, instruction, grammar_path (may be empty),
// max_tokens. Grammar paths are resolved against cwd first, then the views
// file's own directory.
static bool load_views(const std::string & path, std::vector<Item> & items,
                       std::string & err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { err = "cannot open " + path; return false; }
    const std::string dir = dir_of(path);
    std::string line;
    int lineno = 0;
    while (std::getline(fin, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const size_t first_tab = line.find('\t');
        if (lineno == 1 && first_tab != std::string::npos && line.substr(0, first_tab) == "name")
            continue;
        std::vector<std::string> cols;
        size_t start = 0;
        for (int c = 0; c < 3; ++c) {
            const size_t t = line.find('\t', start);
            if (t == std::string::npos) break;
            cols.push_back(line.substr(start, t - start));
            start = t + 1;
        }
        cols.push_back(line.substr(start));
        if (cols.size() != 4) {
            err = "views line " + std::to_string(lineno) + ": expected 4 tab-separated columns";
            return false;
        }
        Item it;
        it.id = cols[0];
        it.prompt = unescape(cols[1]);
        it.raw_suffix = true;
        it.max_new = cols[3].empty() ? 0 : std::atoi(cols[3].c_str());
        if (!cols[2].empty()) {
            if (!read_file(cols[2], it.grammar) && !read_file(dir + cols[2], it.grammar)) {
                err = "cannot open grammar " + cols[2];
                return false;
            }
        }
        if (it.id.empty() || it.prompt.empty()) {
            err = "views line " + std::to_string(lineno) + ": empty name or instruction";
            return false;
        }
        items.push_back(std::move(it));
    }
    return true;
}

static bool         g_record_probs = false;
static std::string  g_probs_path;
static std::string  g_hiddens_path;

// Optional per-row adapter ids and sampling seeds, in row order. Empty means
// the global path (adapter_id = -1, seed = seed + row).
static std::vector<int>      g_adapter_ids;
static std::vector<uint32_t> g_row_seeds;

// Strict comma-separated unsigned integers: no empty entries, no trailing
// garbage, no negative values. Errors are reported instead of clamped.
static bool parse_uint_list(const std::string & s, unsigned long long vmax,
                            std::vector<unsigned long long> & out, std::string & err) {
    out.clear();
    if (s.empty()) { err = "empty list"; return false; }
    size_t start = 0;
    while (true) {
        const size_t c = s.find(',', start);
        const std::string tok = s.substr(start, c == std::string::npos ? std::string::npos : c - start);
        if (tok.empty()) { err = "empty entry"; return false; }
        errno = 0;
        char * end = nullptr;
        const unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
        if (errno != 0 || end == tok.c_str() || *end != '\0' ||
            tok[0] == '-' || tok[0] == '+' || v > vmax) {
            err = "not a valid integer: '" + tok + "'";
            return false;
        }
        out.push_back(v);
        if (c == std::string::npos) break;
        start = c + 1;
    }
    return true;
}

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
                                    float temp, float top_p, uint32_t seed,
                                    int adapter_id = -1) {
    parallel::Branch b;
    b.adapter_id = adapter_id;
    b.id = std::to_string(id);
    b.suffix = it.raw_suffix
        ? it.prompt
        : rt.format_chat({ { "system", sys }, { "user", it.prompt } });
    b.grammar = it.grammar;
    b.max_new_tokens = it.max_new > 0 ? it.max_new : max_new;
    b.role = parallel::Role::Auxiliary;
    b.sampling.temperature = temp;
    b.sampling.top_k = 20;
    b.sampling.top_p = top_p;
    b.sampling.seed = seed;
    b.record_probs = g_record_probs;
    b.record_hidden = !g_hiddens_path.empty();
    return b;
}

static RunOut run_batch(parallel::Runtime & rt, const std::vector<Item> & items,
                        const std::string & shared_prefix, const std::string & sys,
                        int max_new, float temp, float top_p, uint32_t seed) {
    parallel::RunRequest req;
    req.shared_prefix = shared_prefix;
    req.agree_k = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const uint32_t row_seed = i < g_row_seeds.size() ? g_row_seeds[i] : seed + (uint32_t)i;
        const int      adapter  = i < g_adapter_ids.size() ? g_adapter_ids[i] : -1;
        req.branches.push_back(make_branch(rt, (int)i + 1, items[i], sys, max_new,
                                           temp, top_p, row_seed, adapter));
    }
    return run_req(rt, req);
}

static RunOut run_replicated(parallel::Runtime & rt, const Item & it, int n,
                             const std::string & shared_prefix, const std::string & sys,
                             int max_new, float temp, float top_p, uint32_t seed) {
    parallel::RunRequest req;
    req.shared_prefix = shared_prefix;
    req.agree_k = 0;
    for (int i = 0; i < n; ++i)
        req.branches.push_back(make_branch(rt, i + 1, it, sys, max_new,
                                           temp, top_p, seed + (uint32_t)i, -1));
    return run_req(rt, req);
}

static std::string stream_json(const parallel::BranchResult & b, const Item & it) {
    const double gen_s = b.completion_ms > 0 ? (b.completion_ms / 1000.0) : 0.0;
    const double tok_s = gen_s > 0 ? (b.generated_tokens / gen_s) : 0.0;
    std::string js = "{\"id\":\"" + esc(it.id) + "\"";
    js += ",\"text\":\"" + esc(b.text) + "\"";
    js += ",\"generated\":" + std::to_string(b.generated_tokens);
    js += ",\"suffix_tokens\":" + std::to_string(b.suffix_tokens);
    js += ",\"ttft_ms\":" + std::to_string(b.ttft_ms);
    js += ",\"completion_ms\":" + std::to_string(b.completion_ms);
    js += ",\"tok_s\":" + std::to_string(tok_s);
    js += ",\"finish\":\"" + std::string(parallel::to_string(b.finish)) + "\"";
    js += ",\"prompt\":\"" + esc(it.prompt) + "\"}";
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
    std::string model, data, views_path, out, prefix_path;
    std::string sys = "You are a helpful assistant.";
    std::string mode = "batch";
    int max_new = 256, limit = 0, sweep_max = 10;
    float temp = 0.7f, top_p = 0.9f;
    uint32_t seed = 1234;
    std::vector<std::string> lora_paths;
    bool lora_after_prefill = false;
    bool prefix_no_bos = false;
    int samples = 0;
    std::string prompt_path;
    bool allow_approx_prefix = false;
    std::string lora_per_row_arg;
    std::string seed_per_row_arg;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--views") views_path = nx();
        else if (k == "--out") out = nx();
        else if (k == "--system") sys = nx();
        else if (k == "--mode") mode = nx();
        else if (k == "--prefix") prefix_path = nx();
        else if (k == "--max-tokens") max_new = std::atoi(nx().c_str());
        else if (k == "--limit") limit = std::atoi(nx().c_str());
        else if (k == "--sweep-max") sweep_max = std::atoi(nx().c_str());
        else if (k == "--temperature") temp = std::strtof(nx().c_str(), nullptr);
        else if (k == "--top-p") top_p = std::strtof(nx().c_str(), nullptr);
        else if (k == "--seed") seed = (uint32_t)std::strtoul(nx().c_str(), nullptr, 10);
        else if (k == "--probs") { g_probs_path = nx(); g_record_probs = true; }
        else if (k == "--hiddens") { g_hiddens_path = nx(); }
        else if (k == "--lora") lora_paths.push_back(nx());
        else if (k == "--lora-per-row") lora_per_row_arg = nx();
        else if (k == "--seed-per-row") seed_per_row_arg = nx();
        else if (k == "--lora-after-prefill") lora_after_prefill = true;
        else if (k == "--prefix-no-bos") prefix_no_bos = true;
        else if (k == "--samples") samples = std::atoi(nx().c_str());
        else if (k == "--prompt") prompt_path = nx();
        else if (k == "--allow-approx-prefix") allow_approx_prefix = true;
        else { std::fprintf(stderr, "prefix_cli: unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || out.empty() || (data.empty() && views_path.empty() && prompt_path.empty())) {
        std::fprintf(stderr, "prefix_cli: --model --out and either --data, --views or --prompt are required\n");
        return 2;
    }
    if (mode != "batch" && mode != "serial" && mode != "both" && mode != "sweep" && mode != "samples") {
        std::fprintf(stderr, "prefix_cli: --mode batch|serial|both|sweep|samples\n");
        return 2;
    }
    sweep_max = std::max(1, std::min(16, sweep_max));

    std::string shared_prefix;
    if (!prefix_path.empty() && !read_file(prefix_path, shared_prefix)) {
        std::fprintf(stderr, "prefix_cli: cannot open %s\n", prefix_path.c_str());
        return 1;
    }

    std::vector<Item> items;
    if (!views_path.empty()) {
        std::string err;
        if (!load_views(views_path, items, err)) {
            std::fprintf(stderr, "prefix_cli: %s\n", err.c_str());
            return 1;
        }
    } else if (!data.empty()) {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "prefix_cli: cannot open %s\n", data.c_str()); return 1; }
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
    if (items.empty() && samples == 0) { std::fprintf(stderr, "prefix_cli: no prompts\n"); return 1; }
    if (limit > 0 && limit < (int)items.size()) items.resize((size_t)limit);
    if (items.size() > 16) items.resize(10);

    std::vector<int> row_order;
    bool reordered = false;

    if (!lora_per_row_arg.empty()) {
        if (mode != "batch") {
            std::fprintf(stderr, "prefix_cli: --lora-per-row requires --mode batch\n");
            return 2;
        }
        std::vector<unsigned long long> want_raw;
        std::string perr;
        if (!parse_uint_list(lora_per_row_arg, (unsigned long long)INT_MAX, want_raw, perr)) {
            std::fprintf(stderr, "prefix_cli: --lora-per-row: %s\n", perr.c_str());
            return 2;
        }
        std::vector<int> want(want_raw.begin(), want_raw.end());
        if (want.size() != items.size()) {
            std::fprintf(stderr, "prefix_cli: --lora-per-row has %zu entries for %zu rows\n",
                         want.size(), items.size());
            return 2;
        }
        for (int a : want) {
            if (a >= (int)lora_paths.size()) {
                std::fprintf(stderr, "prefix_cli: --lora-per-row adapter %d has no --lora\n", a);
                return 2;
            }
        }

        // decode requires contiguous adapter runs: stable-sort rows by adapter
        // and report the original row index of each output position
        for (size_t i = 0; i < items.size(); ++i) row_order.push_back((int)i);
        std::stable_sort(row_order.begin(), row_order.end(),
                         [&](int x, int y) { return want[(size_t)x] < want[(size_t)y]; });

        std::vector<uint32_t> in_seeds;
        if (!seed_per_row_arg.empty()) {
            std::vector<unsigned long long> seeds_raw;
            if (!parse_uint_list(seed_per_row_arg, 0xffffffffULL, seeds_raw, perr)) {
                std::fprintf(stderr, "prefix_cli: --seed-per-row: %s\n", perr.c_str());
                return 2;
            }
            for (unsigned long long v : seeds_raw) in_seeds.push_back((uint32_t)v);
            if (in_seeds.size() != items.size()) {
                std::fprintf(stderr, "prefix_cli: --seed-per-row has %zu entries for %zu rows\n",
                             in_seeds.size(), items.size());
                return 2;
            }
        }

        std::vector<Item>     sorted_items;
        std::vector<int>      sorted_adapters;

        for (int idx : row_order) {
            sorted_items.push_back(items[(size_t)idx]);
            sorted_adapters.push_back(want[(size_t)idx]);
            g_row_seeds.push_back(in_seeds.empty() ? seed + (uint32_t)idx : in_seeds[(size_t)idx]);
        }

        for (size_t i = 0; i < row_order.size(); ++i)
            if (row_order[i] != (int)i) reordered = true;

        items = std::move(sorted_items);
        g_adapter_ids = std::move(sorted_adapters);
    } else if (!seed_per_row_arg.empty()) {
        std::fprintf(stderr, "prefix_cli: --seed-per-row requires --lora-per-row\n");
        return 2;
    }

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    parallel::RuntimeConfig rcfg;
    rcfg.model_path = model;
    rcfg.max_context_tokens = 8192;
    rcfg.max_branches = 16;
    rcfg.n_gpu_layers = 99;
    rcfg.flash_attn = true;
    rcfg.lora_paths = lora_paths;
    rcfg.lora_after_prefill = lora_after_prefill;
    rcfg.prefix_add_special = !prefix_no_bos;
    rcfg.allow_approx_prefix = allow_approx_prefix;
    parallel::Runtime rt(rcfg);
    if (!rt.ok()) {
        std::fprintf(stderr, "prefix_cli: runtime init failed: %s\n", rt.error().c_str());
        return 1;
    }

    std::string streams = "[]", serial = "[]", sweep = "[]";
    RunOut batch;
    int shared_prefix_tokens = 0;
    double prefill_ms = 0.0;
    std::vector<parallel::BranchResult> serial_brs;
    double batch_wall = 0.0;
    if (mode == "batch" || mode == "both") {
        batch = run_batch(rt, items, shared_prefix, sys, max_new, temp, top_p, seed);
        if (!batch.ok) { std::fprintf(stderr, "prefix_cli: %s\n", batch.err.c_str()); return 1; }
        batch_wall = batch.wall_ms;
        shared_prefix_tokens = batch.res.shared_prefix_tokens;
        prefill_ms = batch.res.prefill_ms;
        streams = "[";
        for (size_t i = 0; i < batch.res.branches.size(); ++i) {
            if (i) streams += ",";
            streams += stream_json(batch.res.branches[i], items[i]);
        }
        streams += "]";
    }
    if (mode == "serial" || mode == "both") {
        serial = "[";
        for (size_t i = 0; i < items.size(); ++i) {
            RunOut one = run_batch(rt, { items[i] }, shared_prefix, sys, max_new, temp, top_p, seed + (uint32_t)i);
            if (!one.ok) { std::fprintf(stderr, "prefix_cli: serial %zu: %s\n", i, one.err.c_str()); return 1; }
            if (i) serial += ",";
            serial += stream_json(one.res.branches[0], items[i]);
            serial_brs.push_back(one.res.branches[0]);
        }
        serial += "]";
    }
    if (mode == "sweep") {
        sweep = "[";
        for (int n = 1; n <= sweep_max; ++n) {
            RunOut o = run_replicated(rt, items[0], n, shared_prefix, sys, max_new, temp, top_p, seed);
            if (!o.ok) { std::fprintf(stderr, "prefix_cli: sweep n=%d: %s\n", n, o.err.c_str()); return 1; }
            if (n > 1) sweep += ",";
            sweep += sweep_json(n, o);
        }
        sweep += "]";
    }
    if (mode == "samples" || samples > 0) {
        std::string ptext;
        if (!read_file(prompt_path, ptext)) { std::fprintf(stderr, "prefix_cli: cannot read %s\n", prompt_path.c_str()); return 1; }
        // chat prefix once (no assistant header); the runtime appends the marker per branch
        std::string sp = rt.format_chat({ { "system", sys }, { "user", ptext } });
        Item it; it.prompt = ""; it.raw_suffix = true; it.max_new = max_new;
        const int n = samples > 0 ? samples : 4;
        RunOut o = run_replicated(rt, it, n, sp, sys, max_new, temp, top_p, seed);
        if (!o.ok) { std::fprintf(stderr, "prefix_cli: samples: %s\n", o.err.c_str()); return 1; }
        batch = o;
        batch_wall = o.wall_ms;
        shared_prefix_tokens = o.res.shared_prefix_tokens;
        prefill_ms = o.res.prefill_ms;
        streams = "[";
        for (size_t i = 0; i < o.res.branches.size(); ++i) {
            if (i) streams += ",";
            streams += stream_json(o.res.branches[i], it);
        }
        streams += "]";
    }

    // Optional raw-logit sidecar: one JSON object per emitted token.
    // m = mode, s = stream id, H = full-vocab entropy (nats, raw logits),
    // p1 = top-1 prob, mg = p1-p2, lp = emitted-token log-prob, off/len =
    // byte span of the emitted piece in that stream's text.
    if (g_record_probs && !g_probs_path.empty()) {
        FILE * pf = std::fopen(g_probs_path.c_str(), "wb");
        if (!pf) { std::fprintf(stderr, "prefix_cli: cannot write %s\n", g_probs_path.c_str()); return 1; }
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

    // Optional last-layer embedding sidecar: one row per emitted step and one
    // row per cross-stream consensus step (batch mode; consensus is a property
    // of the run, so it is stored once in RunResult::hcons).
    if (!g_hiddens_path.empty()) {
        FILE * hf = std::fopen(g_hiddens_path.c_str(), "wb");
        if (!hf) { std::fprintf(stderr, "prefix_cli: cannot write %s\n", g_hiddens_path.c_str()); return 1; }
        for (const auto & b : batch.res.branches)
            for (const auto & r : b.hidden)
                std::fprintf(hf, "{\"k\":\"h\",\"m\":\"batch\",\"s\":\"%s\",\"step\":%d,\"norm\":%.6f,\"cmean\":%.6f}\n",
                             b.id.c_str(), r.step, r.norm, r.cos_mean);
        for (const auto & c : batch.res.hcons)
            std::fprintf(hf,
                "{\"k\":\"c\",\"m\":\"batch\",\"step\":%d,\"cos_mean\":%.6f,"
                "\"cos_min\":%.6f,\"cos_max\":%.6f,\"n\":%d}\n",
                c.step, c.cos_mean, c.cos_min, c.cos_max, c.n);
        std::fclose(hf);
    }

    std::string js = "{\"ok\":true,\"mode\":\"" + mode + "\"";
    js += ",\"n\":" + std::to_string(items.size());
    js += ",\"wall_ms\":" + std::to_string(batch_wall);
    js += ",\"steps\":" + std::to_string(batch.res.steps);
    js += ",\"hidden_consensus\":" + std::to_string(batch.res.hcons.size());
    js += ",\"active_hist\":[";
    for (size_t i = 0; i < batch.res.active_hist.size(); ++i) {
        if (i) js += ",";
        js += std::to_string(batch.res.active_hist[i]);
    }
    js += "]";
    js += ",\"lora_per_row\":[";
    for (size_t i = 0; i < g_adapter_ids.size(); ++i) {
        if (i) js += ",";
        js += std::to_string(g_adapter_ids[i]);
    }
    js += "]";
    js += ",\"seed_per_row\":[";
    for (size_t i = 0; i < g_row_seeds.size(); ++i) {
        if (i) js += ",";
        js += std::to_string(g_row_seeds[i]);
    }
    js += "]";
    js += ",\"row_order\":[";
    for (size_t i = 0; i < row_order.size(); ++i) {
        if (i) js += ",";
        js += std::to_string(row_order[i]);
    }
    js += "]";
    js += ",\"reordered\":";
    js += reordered ? "true" : "false";
    js += ",\"shared_prefix_tokens\":" + std::to_string(shared_prefix_tokens);
    js += ",\"prefill_ms\":" + std::to_string(prefill_ms);
    js += ",\"streams\":" + streams;
    js += ",\"serial\":" + serial;
    js += ",\"sweep\":" + sweep;
    js += "}";

    FILE * f = std::fopen(out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "prefix_cli: cannot write %s\n", out.c_str()); return 1; }
    std::fwrite(js.data(), 1, js.size(), f);
    std::fclose(f);
    std::printf("%s\n", js.c_str());
    return 0;
}
