// RecycleLM IFEval response generator.
//
// One model, one prompt per context, greedy-free sampling with the bench
// protocol: temp 0.6, top_k 20, top_p 0.95, seed = 1234 + prompt index
// (+ --seed-add for the retry pass). Max 256 new tokens.
//
// Modes:
//   baseline : one branch on seq 0 (default).
//   race     : one shared prefill on seq 0, then seq_cp(0,1); two branches
//              decode together in one Engine::tick loop. Branch A uses
//              seed + idx, branch B uses seed + idx + 10000.
//
// Input TSV: key <TAB> prompt
//   Escapes: \t \n \r \\ (same convention as the other harness data files).
//   Retry TSV (--retry-data): key <TAB> retry_prompt [<TAB> index]
//   The optional third column preserves the original prompt index so the retry
//   seed is 1234 + index + 10000 rather than 1234 + retry-file row.
//
// Output CSV (baseline):
//   key,response,prefill_ms,gen_ms,total_ms,n_tokens,idx,seed,finish
// Output CSV (race):
//   key,idx,mode,seed_a,seed_b,response_a,response_b,finish_a,finish_b,
//   n_tokens_a,n_tokens_b,prefill_ms,gen_ms,wall_ms,steps,joint_steps,
//   active_hist
//   wall_ms is the external per-prompt timer (chat template + prefill + both
//   branches); steps counts decode ticks; joint_steps counts ticks where both
//   branches were active; active_hist is "n:ticks" joined by "|".
//
// Usage:
//   ifeval.exe --model models\Llama-3.2-3B-Instruct-Q4_K_M.gguf \
//              --data data\ifeval_541.tsv --out results\ifeval_baseline_raw.csv \
//              [--mode baseline|race] [--limit 541] [--max-tokens 256]
//              [--seed 1234] [--seed-add 0] [--smoke] [--verbose]
//
//   ifeval.exe ... --retry-data results\ifeval_retry_input.tsv \
//                  --out results\ifeval_retry_raw.csv --seed-add 10000

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <cctype>
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
    o.reserve(s.size() + 16);
    for (char c : s) {
        if      (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else                o += c;
    }
    return o;
}

// Token pieces are raw bytes. A response cut at max_tokens can end in the
// middle of a multi-byte UTF-8 character; drop an incomplete trailing sequence
// and replace stray invalid bytes so the CSV is valid UTF-8.
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

static std::string hist_string(const std::vector<long long> & h) {
    std::string o;
    for (size_t i = 0; i < h.size(); ++i) {
        if (h[i] == 0) continue;
        if (!o.empty()) o += '|';
        o += std::to_string(i) + ":" + std::to_string(h[i]);
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

static std::vector<llama_token> tokenize(Engine & e, const std::string & s) {
    std::vector<llama_token> t(s.size() + 32);
    int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(),
                           false, true);
    if (n < 0) {
        t.resize((size_t)(-n));
        n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(),
                           false, true);
    }
    if (n < 0) { t.clear(); return t; }
    t.resize((size_t)n);
    return t;
}

static std::string token_piece(const llama_vocab * v, llama_token tok) {
    char buf[512];
    int n = llama_token_to_piece(v, tok, buf, (int)sizeof(buf), 0, false);
    if (n >= 0) return std::string(buf, (size_t)n);
    std::vector<char> big((size_t)(-n) + 1);
    n = llama_token_to_piece(v, tok, big.data(), (int)big.size(), 0, false);
    if (n <= 0) return std::string();
    return std::string(big.data(), (size_t)n);
}

static std::string apply_chat(Engine & e,
                              const std::vector<std::pair<std::string, std::string>> & msgs,
                              bool add_ass) {
    const char * tmpl = e.model_chat_template();
    if (!tmpl) return std::string();
    std::vector<llama_chat_message> m;
    size_t cap = 1024;
    for (const auto & kv : msgs) {
        m.push_back({ kv.first.c_str(), kv.second.c_str() });
        cap += kv.second.size() * 2 + 64;
    }
    std::vector<char> buf(cap);
    int n = llama_chat_apply_template(tmpl, m.data(), m.size(), add_ass,
                                      buf.data(), (int)buf.size());
    if (n < 0) return std::string();
    if (n > (int)buf.size()) {
        buf.resize((size_t)n);
        n = llama_chat_apply_template(tmpl, m.data(), m.size(), add_ass,
                                      buf.data(), (int)buf.size());
        if (n < 0) return std::string();
    }
    return std::string(buf.data(), (size_t)n);
}

static std::string chat_prompt(Engine & e, const std::string & sys,
                               const std::string & user) {
    return apply_chat(e, { { "system", sys }, { "user", user } }, true);
}

static llama_sampler * make_sampler(uint32_t seed) {
    auto sp = llama_sampler_chain_default_params();
    auto * c = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(c, llama_sampler_init_dist(seed));
    return c;
}

struct StreamProc : Process {
    Engine & e;
    llama_seq_id seq;
    llama_pos pos;
    int max_new;
    int n = 0;
    std::vector<llama_token> toks;
    std::string text;
    llama_sampler * smpl = nullptr;
    llama_token next = 0;
    bool fin = false, eos = false, failed = false;
    int base_ = 0;

    StreamProc(Engine & en, llama_seq_id s, llama_pos p0, int mn, uint32_t seed)
        : e(en), seq(s), pos(p0), max_new(mn) {
        smpl = make_sampler(seed);
    }
    ~StreamProc() override { if (smpl) llama_sampler_free(smpl); }

    void seq_ids(std::vector<llama_seq_id> & o) const override { o.push_back(seq); }
    bool done() const override { return fin; }

    void mark_fin(bool end_of_seq) {
        if (fin) return;
        fin = true;
        eos = eos || end_of_seq;
    }

    void start() {
        if (!smpl) { failed = true; mark_fin(true); return; }
        next = llama_sampler_sample(smpl, e.ctx(), -1);
        if (llama_vocab_is_eog(e.vocab(), next) || max_new <= 0) mark_fin(true);
    }

    void want(int budget, int base, std::vector<Request> & out) override {
        (void)budget;
        if (fin) return;
        base_ = base;
        out.push_back({ seq, pos, next, true });
    }

    void on_logits(int idx, const float * logits) override {
        (void)logits;
        if (idx != base_ || fin) return;
        if (llama_vocab_is_eog(e.vocab(), next)) { mark_fin(true); return; }
        toks.push_back(next);
        text += token_piece(e.vocab(), next);
        ++n;
        ++pos;
        if (n >= max_new) { mark_fin(false); return; }
        next = llama_sampler_sample(smpl, e.ctx(), idx);
        if (llama_vocab_is_eog(e.vocab(), next)) mark_fin(true);
    }
};

static const char * kSys =
    "You are a helpful assistant. Follow all instructions exactly.";

int main(int argc, char ** argv) {
    std::string model, data, retry_data, out, mode = "baseline";
    int limit = 541;
    int max_new = 256;
    uint32_t seed = 1234;
    uint32_t seed_add = 0;
    bool smoke = false;
    bool verbose = false;

    auto need = [&](int i, const char * flag) -> bool {
        if (i + 1 < argc) return true;
        std::fprintf(stderr, "ifeval: %s needs a value\n", flag);
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") { if (!need(i, "--model")) return 2; model = argv[++i]; }
        else if (k == "--data") { if (!need(i, "--data")) return 2; data = argv[++i]; }
        else if (k == "--retry-data") { if (!need(i, "--retry-data")) return 2; retry_data = argv[++i]; }
        else if (k == "--out") { if (!need(i, "--out")) return 2; out = argv[++i]; }
        else if (k == "--mode") { if (!need(i, "--mode")) return 2; mode = argv[++i]; }
        else if (k == "--limit") { if (!need(i, "--limit")) return 2; limit = std::atoi(argv[++i]); }
        else if (k == "--max-tokens") { if (!need(i, "--max-tokens")) return 2; max_new = std::atoi(argv[++i]); }
        else if (k == "--seed") { if (!need(i, "--seed")) return 2; seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10); }
        else if (k == "--seed-add") { if (!need(i, "--seed-add")) return 2; seed_add = (uint32_t)std::strtoul(argv[++i], nullptr, 10); }
        else if (k == "--smoke") smoke = true;
        else if (k == "--verbose") verbose = true;
        else { std::fprintf(stderr, "ifeval: unknown argument: %s\n", k.c_str()); return 2; }
    }

    const bool race = (mode == "race");
    if (mode != "baseline" && mode != "race") {
        std::fprintf(stderr, "ifeval: --mode must be baseline or race\n");
        return 2;
    }
    if (race && !retry_data.empty()) {
        std::fprintf(stderr, "ifeval: --mode race does not take --retry-data\n");
        return 2;
    }

    const bool retry_mode = !retry_data.empty();
    if (model.empty() || out.empty() || (data.empty() && !retry_mode)) {
        std::fprintf(stderr, "ifeval: --model --out and --data (or --retry-data) are required\n");
        return 2;
    }
    if (limit < 1 || max_new < 1) {
        std::fprintf(stderr, "ifeval: --limit and --max-tokens must be >= 1\n");
        return 2;
    }
    const std::string src = retry_mode ? retry_data : data;

    std::vector<Item> all;
    {
        std::ifstream fin(src, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "ifeval: cannot open --data: %s\n", src.c_str()); return 1; }
        std::string line;
        int lineno = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (lineno == 1 && f[0] == "key") continue;
            if (f.size() < 2) {
                std::fprintf(stderr, "ifeval: data line %d has %zu columns (need >= 2)\n",
                             lineno, f.size());
                return 1;
            }
            Item it;
            it.key = unescape_field(f[0]);
            it.prompt = unescape_field(f[1]);
            it.idx = (f.size() >= 3 && !f[2].empty()) ? std::atoi(f[2].c_str()) : (int)all.size();
            all.push_back(std::move(it));
        }
    }
    if (all.empty()) { std::fprintf(stderr, "ifeval: no items read from %s\n", src.c_str()); return 1; }

    if (limit > (int)all.size()) limit = (int)all.size();
    if (smoke && limit > 10) limit = 10;
    all.resize((size_t)limit);

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = 8192;
    cfg.n_seq_max = race ? 2 : 1;
    cfg.max_batch = 2048;
    cfg.fa = true;
    cfg.kv_unified = true;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "ifeval: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "ifeval: cannot open --out: %s\n", out.c_str());
        e.shutdown();
        return 1;
    }
    if (race) {
        std::fprintf(f,
                     "key,idx,mode,seed_a,seed_b,response_a,response_b,"
                     "finish_a,finish_b,n_tokens_a,n_tokens_b,prefill_ms,"
                     "gen_ms,wall_ms,steps,joint_steps,active_hist\n");
    } else {
        std::fprintf(f,
                     "key,response,prefill_ms,gen_ms,total_ms,n_tokens,idx,seed,finish\n");
    }

    std::printf("model=%s\n", model.c_str());
    std::printf("input=%s rows=%d mode=%s max_tokens=%d seed=%u seed_add=%u\n",
                src.c_str(), (int)all.size(),
                race ? "race" : (retry_mode ? "retry" : "baseline"),
                max_new, seed, seed_add);
    std::fflush(stdout);

    const auto t_run0 = Clock::now();
    int n_ok = 0, n_max = 0, n_eos = 0;

    for (int ci = 0; ci < (int)all.size(); ++ci) {
        const Item & it = all[(size_t)ci];
        const uint32_t cseed = seed + (uint32_t)it.idx + seed_add;
        const auto t_case = Clock::now();

        const std::string prompt = chat_prompt(e, kSys, it.prompt);
        if (prompt.empty()) {
            std::fprintf(stderr, "ifeval: chat template failed (%s)\n", it.key.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        const std::vector<llama_token> ptoks = tokenize(e, prompt);
        if (ptoks.empty()) {
            std::fprintf(stderr, "ifeval: tokenization failed (%s)\n", it.key.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        if ((int)ptoks.size() + max_new + 8 > cfg.n_ctx) {
            std::fprintf(stderr, "ifeval: prompt too long for n_ctx (%s: %zu tokens)\n",
                         it.key.c_str(), ptoks.size());
            std::fclose(f); e.shutdown(); return 1;
        }

        e.clear();
        e.reset_processes();
        const auto t0 = Clock::now();
        if (!e.prefill(0, ptoks, true)) {
            std::fprintf(stderr, "ifeval: prefill failed (%s)\n", it.key.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        if (race) e.seq_cp(0, 1);
        e.sync();
        const double prefill_ms = ms_since(t0);

        if (race) {
            const uint32_t cseed_b = cseed + 10000u;
            StreamProc a(e, 0, (llama_pos)ptoks.size(), max_new, cseed);
            StreamProc b(e, 1, (llama_pos)ptoks.size(), max_new, cseed_b);
            if (!a.smpl || !b.smpl) {
                std::fprintf(stderr, "ifeval: sampler init failed (%s)\n", it.key.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }
            const auto tg0 = Clock::now();
            a.start();
            b.start();
            e.add(&a);
            e.add(&b);
            long long steps = 0, joint_steps = 0;
            std::vector<long long> hist(3, 0);
            while (!a.done() || !b.done()) {
                int active = 0;
                if (e.tick(2, active) < 0) {
                    std::fprintf(stderr, "ifeval: decode failed (%s)\n", it.key.c_str());
                    std::fclose(f); e.shutdown(); return 1;
                }
                if (active <= 0) break;
                if (active >= (int)hist.size()) hist.resize((size_t)active + 1, 0);
                ++hist[(size_t)active];
                ++steps;
                if (active >= 2) ++joint_steps;
            }
            e.sync();
            const double gen_ms = ms_since(tg0);
            if (a.failed || b.failed) {
                std::fprintf(stderr, "ifeval: generation failed (%s)\n", it.key.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }

            const char * fa = a.eos ? "eos" : (a.n >= max_new ? "max" : "other");
            const char * fb = b.eos ? "eos" : (b.n >= max_new ? "max" : "other");
            if (a.eos) ++n_eos;
            if (a.n >= max_new) ++n_max;
            ++n_ok;

            const std::string ra = sanitize_utf8(a.text);
            const std::string rb = sanitize_utf8(b.text);
            std::fprintf(f,
                         "%s,%d,race,%u,%u,%s,%s,%s,%s,%d,%d,%.3f,%.3f,%.3f,"
                         "%lld,%lld,%s\n",
                         csv_field(it.key).c_str(), it.idx, cseed, cseed_b,
                         csv_field(ra).c_str(), csv_field(rb).c_str(),
                         fa, fb, a.n, b.n, prefill_ms, gen_ms,
                         ms_since(t_case), steps, joint_steps,
                         hist_string(hist).c_str());
            std::fflush(f);

            std::printf("[%3d/%3d] key=%s A n=%d finish=%s | B n=%d finish=%s | "
                        "joint=%lld/%lld wall_ms=%.0f\n",
                        ci + 1, (int)all.size(), it.key.c_str(),
                        a.n, fa, b.n, fb, joint_steps, steps, ms_since(t_case));
            if (verbose) {
                std::printf("  prompt: \"%s\"\n", esc(it.prompt).c_str());
                std::printf("  A: \"%s\"\n", esc(ra).c_str());
                std::printf("  B: \"%s\"\n", esc(rb).c_str());
            }
        } else {
            StreamProc gen(e, 0, (llama_pos)ptoks.size(), max_new, cseed);
            if (!gen.smpl) {
                std::fprintf(stderr, "ifeval: sampler init failed (%s)\n", it.key.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }
            const auto tg0 = Clock::now();
            gen.start();
            e.add(&gen);
            while (!gen.done()) {
                int active = 0;
                if (e.tick(1, active) < 0) {
                    std::fprintf(stderr, "ifeval: decode failed (%s)\n", it.key.c_str());
                    std::fclose(f); e.shutdown(); return 1;
                }
            }
            e.sync();
            const double gen_ms = ms_since(tg0);
            if (gen.failed) {
                std::fprintf(stderr, "ifeval: generation failed (%s)\n", it.key.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }

            const char * finish = gen.eos ? "eos" : (gen.n >= max_new ? "max" : "other");
            if (gen.eos) ++n_eos;
            if (gen.n >= max_new) ++n_max;
            ++n_ok;

            const std::string resp = sanitize_utf8(gen.text);
            std::fprintf(f, "%s,%s,%.3f,%.3f,%.3f,%d,%d,%u,%s\n",
                         csv_field(it.key).c_str(),
                         csv_field(resp).c_str(),
                         prefill_ms, gen_ms, ms_since(t_case),
                         gen.n, it.idx, cseed, finish);
            std::fflush(f);

            std::printf("[%3d/%3d] key=%s n=%d finish=%s gen_ms=%.0f\n",
                        ci + 1, (int)all.size(), it.key.c_str(), gen.n, finish,
                        gen_ms);
            if (verbose) {
                std::printf("  prompt: \"%s\"\n", esc(it.prompt).c_str());
                std::printf("  resp:   \"%s\"\n", esc(resp).c_str());
            }
        }
        std::fflush(stdout);
        e.reset_processes();
    }

    const double total_s = ms_since(t_run0) / 1000.0;
    std::printf("wrote %s\n", out.c_str());
    std::printf("rows=%d eos=%d max=%d wall_s=%.1f\n", n_ok, n_eos, n_max, total_s);
    std::fclose(f);
    e.shutdown();
    return 0;
}
