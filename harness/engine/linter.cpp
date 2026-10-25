// RecycleLM linter-triggered KV rollback bench (2x2 correction experiment).
//
// The actor writes one Python function. The linter is deterministic: any
// `import` / `from` line is forbidden. For triggered cases the orchestrator
// erases part of the actor span from the KV cache and asks for a rewrite.
//
// Actor prompt: system "Output only one ```python code block and nothing
// else." + the case prompt, with add_ass=true so the assistant turn is open.
// The actor's EOG is sampled but never decoded, so follow-up prompts start
// with an explicit <|im_end|> that closes the actor turn in the KV.
//
// Arms (triggered cases only; clean cases return the actor text for every arm):
//   control          - final = actor_text; no linter, no correction.
//   partial_comment  - erase from the import-line start, inject an in-turn
//                      LINT comment with the import line's indentation, and
//                      continue the same assistant turn.
//   partial_userturn - same truncation, but close the turn and inject a user
//                      turn before continuing.
//   full_comment     - erase the whole code block (fence, else char 0), close
//                      the actor turn, inject a module-level LINT comment.
//   full_userturn    - erase the whole block and inject a rewrite user turn.
//   flagonly         - no rewrite; final = actor_text + linter error line.
//   plain_retry      - discard the actor KV (clear + re-prefill the original
//                      prompt from scratch) then inject the same user-turn
//                      correction content as full_userturn. No seq_rm. This is
//                      a standard API reprompt and exists to test whether the
//                      partial-KV rollback has any advantage over a plain
//                      prompt re-send. Runs last because e.clear() drops the
//                      copy used by the other correction arms.
//
// Mapping: char_start snaps to the last newline at or before it. The token
// prefix is the largest prefix whose decoded length is <= that snap. No
// one-token step-back: the old `best - 1` rule dropped the token that ends
// exactly at the import line (e.g. `):\n`) and produced `def gcd(a, b`.
//
// Usage:
//   linter.exe --model M --data data\linter_cases_strong_100.tsv \
//              --out results\linter_2x2_1234.csv --limit 100 \
//              --actor 192 --cont 192 --mode all --seed 1234 [--verbose]
//
//   linter.exe ... --cases t_isqrt_1,k_isqrt_1
//   linter.exe --model M --mapping-selftest
//   linter.exe --model M --mapping-selftest \
//              --selftest-csv results\linter_trigger_strong_1234.csv

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
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(const Clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double us_between(const Clock::time_point & a, const Clock::time_point & b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
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

struct Case {
    std::string id, type, tmpl, prompt, func, hint;
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

static std::string text_of(Engine & e, const std::vector<llama_token> & toks,
                           int begin, int end) {
    std::string s;
    for (int i = begin; i < end && i < (int)toks.size(); ++i) {
        s += token_piece(e.vocab(), toks[(size_t)i]);
    }
    return s;
}

static bool is_ascii(const std::string & s) {
    for (unsigned char c : s) if (c >= 0x80) return false;
    return true;
}

// Start of the line containing char_start (the position after the last
// newline at or before char_start).
static size_t snap_line_start(const std::string & text, size_t char_start) {
    if (char_start == 0 || text.empty()) return 0;
    if (char_start > text.size()) char_start = text.size();
    const size_t nl = text.rfind('\n', char_start - 1);
    return (nl == std::string::npos) ? 0 : nl + 1;
}

// Largest token prefix whose decoded length is <= the snapped char position.
// No step-back: a token that ends exactly at the snap is retained. For ASCII
// text, map_exact records whether the retained prefix is byte-identical to
// text.substr(0, snap); a strict prefix means the next token crosses the snap.
static int keep_token_for_char(Engine & e, const std::string & text,
                               const std::vector<llama_token> & toks,
                               size_t char_start, size_t * snap_out,
                               int * map_exact) {
    const size_t snap = snap_line_start(text, char_start);
    if (snap_out) *snap_out = snap;

    int best = 0;
    std::string pref;
    for (int i = 1; i <= (int)toks.size(); ++i) {
        pref += token_piece(e.vocab(), toks[(size_t)i - 1]);
        if (pref.size() <= snap) best = i;
        else break;
    }

    std::string kept = text_of(e, toks, 0, best);
    while (best > 0 && kept.size() > snap) {
        --best;
        kept = text_of(e, toks, 0, best);
    }

    int exact = 0;
    if (is_ascii(text)) {
        const std::string target = text.substr(0, snap);
        if (kept == target) {
            exact = 1;
        } else if (kept.size() > target.size()) {
            while (best > 0 && kept.size() > target.size()) {
                --best;
                kept = text_of(e, toks, 0, best);
            }
            exact = (kept == target) ? 1 : 0;
        }
    }
    if (map_exact) *map_exact = exact;
    return best;
}

// First line matching ^\s*(import|from)\s+ (multiline, no regex dependency).
// pos        = first non-whitespace character of the import keyword.
// line_start = the character just after the previous newline (indentation
//              included), i.e. the partial-rollback point.
// indent     = the line's leading whitespace.
struct ImportHit {
    bool found = false;
    size_t pos = 0;
    size_t line_start = 0;
    std::string indent;
};

static ImportHit find_first_import(const std::string & s) {
    size_t ls = 0;
    while (ls <= s.size()) {
        const size_t nl = s.find('\n', ls);
        const size_t le = (nl == std::string::npos) ? s.size() : nl;
        size_t p = ls;
        while (p < le && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r')) ++p;
        auto kw = [&](const char * w) {
            const size_t n = std::strlen(w);
            if (p + n >= le) return false;
            if (s.compare(p, n, w) != 0) return false;
            const char c = s[p + n];
            return c == ' ' || c == '\t' || c == '\r';
        };
        if (kw("import") || kw("from")) {
            return { true, p, ls, s.substr(ls, p - ls) };
        }
        if (nl == std::string::npos) break;
        ls = nl + 1;
    }
    return {};
}

// top       = no `def ` before the import
// inside    = a `def ` before the import and the import line is indented
// after_def = a `def ` before the import and the import is at column 0
static int classify_import_pos(const std::string & text, const ImportHit & hit) {
    if (!hit.found) return -1;
    const size_t defp = text.rfind("def ", hit.pos);
    if (defp == std::string::npos) return 0;
    if (!hit.indent.empty()) return 1;
    return 2;
}

static const char * import_pos_name(int p) {
    if (p == 0) return "top";
    if (p == 1) return "inside";
    if (p == 2) return "after_def";
    return "na";
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

struct GenOut {
    std::vector<llama_token> toks;
    std::string text;
    int n = 0;
    bool eos = false;
    bool failed = false;
    double ms = 0.0;
};

static GenOut generate(Engine & e, llama_seq_id seq, llama_pos pos0, int max_tokens,
                       uint32_t seed) {
    GenOut g;
    llama_sampler * smpl = make_sampler(seed);
    std::vector<llama_token> one(1);
    const auto t0 = Clock::now();
    llama_token tok = llama_sampler_sample(smpl, e.ctx(), -1);
    while (g.n < max_tokens) {
        if (llama_vocab_is_eog(e.vocab(), tok)) { g.eos = true; break; }
        const std::string piece = token_piece(e.vocab(), tok);
        one[0] = tok;
        if (!e.decode_from(seq, one, pos0 + (llama_pos)g.n, true)) {
            g.failed = true;
            break;
        }
        e.sync();
        g.toks.push_back(tok);
        g.text += piece;
        ++g.n;
        if (g.n >= max_tokens) break;
        tok = llama_sampler_sample(smpl, e.ctx(), -1);
    }
    e.sync();
    g.ms = ms_since(t0);
    llama_sampler_free(smpl);
    return g;
}

static std::vector<llama_token> chat_prompt(Engine & e, const std::string & sys,
                                            const std::string & user) {
    const char * tmpl = e.model_chat_template();
    if (!tmpl) {
        return tokenize(e, "<|im_start|>system\n" + sys + "<|im_end|>\n"
                           "<|im_start|>user\n" + user + "<|im_end|>\n"
                           "<|im_start|>assistant\n");
    }
    llama_chat_message msgs[2] = {
        { "system", sys.c_str() },
        { "user",   user.c_str() },
    };
    size_t cap = std::max<size_t>(4096, user.size() * 2 + sys.size() * 2 + 1024);
    std::vector<char> buf(cap);
    int plen = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
    if (plen < 0) return std::vector<llama_token>();
    if (plen > (int)buf.size()) {
        buf.resize((size_t)plen);
        plen = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
        if (plen < 0) return std::vector<llama_token>();
    }
    return tokenize(e, std::string(buf.data(), (size_t)plen));
}

enum ArmId {
    ARM_CONTROL = 0,
    ARM_PARTIAL_COMMENT,
    ARM_PARTIAL_USERTURN,
    ARM_FULL_COMMENT,
    ARM_FULL_USERTURN,
    ARM_FLAGONLY,
    ARM_PLAIN_RETRY,
    ARM_COUNT
};

static const char * ARM_NAMES[ARM_COUNT] = {
    "control", "partial_comment", "partial_userturn",
    "full_comment", "full_userturn", "flagonly", "plain_retry"
};

struct ArmRow {
    std::string arm;
    std::string cont;
    std::string final_text;
    std::string inject;
    int first_import = 0;
    int partial = 0;
    int map_exact = -1;
    int rm_ok = -1;
    int flagged = 0;
    long long char_start = -1;
    long long keep_token = -1;
    long long keep_pos = -1;
    long long pos_max_before = -1;
    long long pos_max_after = -1;
    double rm_us = 0.0;
    double prefill_ms = 0.0;
    double actor_prefill_ms = 0.0;
    double cont_ms = 0.0;
    double total_ms = 0.0;
};

static void write_row(FILE * f, const Case & c, const ArmRow & r, const std::string & actor,
                      const char * import_pos, double actor_ms, uint32_t seed) {
    std::fprintf(f,
                 "%s,%s,%u,%s,%s,%s,%s,%d,-1,-1,-1,-1,%s,%lld,%lld,%lld,%d,%d,%d,"
                 "%.3f,%lld,%lld,%d,%.3f,%.3f,%.3f,%s,%s,%.3f,%.3f\n",
                 csv_field(c.id).c_str(), csv_field(c.type).c_str(), seed,
                 csv_field(r.arm).c_str(), csv_field(actor).c_str(),
                 csv_field(r.cont).c_str(), csv_field(r.final_text).c_str(),
                 r.first_import, import_pos, r.char_start, r.keep_token, r.keep_pos,
                 r.partial, r.map_exact, r.rm_ok, r.rm_us, r.pos_max_before,
                 r.pos_max_after, r.flagged, actor_ms, r.cont_ms, r.total_ms,
                 csv_field(c.tmpl).c_str(), csv_field(r.inject).c_str(),
                 r.prefill_ms, r.actor_prefill_ms);
}

static std::string arm_inject(bool full, bool userturn, const ImportHit & hit,
                              const Case & c, const std::string & kept) {
    const std::string known = " Known edge case: " + c.hint;
    if (!full) {
        if (!userturn) {
            std::string inj = hit.indent +
                "# LINT: Forbidden import removed. Continue this function using "
                "only builtins." + known + "\n" + hit.indent;
            if (!kept.empty() && kept.back() != '\n') inj = "\n" + inj;
            return inj;
        }
        return "<|im_end|>\n<|im_start|>user\nForbidden import removed. Continue "
               "this function using only builtins." + known +
               "<|im_end|>\n<|im_start|>assistant\n";
    }
    if (!userturn) {
        return "<|im_end|>\n# LINT: The previous attempt used a forbidden import. "
               "Rewrite `" + c.func + "` from scratch using only builtins." +
               known + "\n";
    }
    return "<|im_end|>\n<|im_start|>user\nThe previous attempt used a forbidden "
           "import. Rewrite `" + c.func + "` from scratch using only builtins." +
           known + "<|im_end|>\n<|im_start|>assistant\n";
}

// ---------------------------------------------------------------- selftest

static std::string strip_cr(std::string s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) if (c != '\r') o += c;
    return o;
}

static std::vector<std::vector<std::string>> read_csv(const std::string & path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return rows;
    const std::string data((std::istreambuf_iterator<char>(fin)),
                           std::istreambuf_iterator<char>());
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < data.size(); ++i) {
        const char c = data[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < data.size() && data[i + 1] == '"') { field += '"'; ++i; }
                else quoted = false;
            } else {
                field += c;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            row.push_back(field);
            field.clear();
        } else if (c == '\n') {
            row.push_back(field);
            field.clear();
            rows.push_back(row);
            row.clear();
        } else if (c != '\r') {
            field += c;
        }
    }
    if (!field.empty() || !row.empty()) {
        row.push_back(field);
        rows.push_back(row);
    }
    return rows;
}

static int mapping_selftest(Engine & e, const std::string & csv_path) {
    const auto rows = read_csv(csv_path);
    if (rows.empty()) {
        std::fprintf(stderr, "linter: cannot read selftest csv: %s\n", csv_path.c_str());
        return 1;
    }
    size_t ai_col = std::string::npos, id_col = std::string::npos;
    for (size_t i = 0; i < rows[0].size(); ++i) {
        if (rows[0][i] == "actor_text") ai_col = i;
        if (rows[0][i] == "id") id_col = i;
    }
    if (ai_col == std::string::npos) {
        std::fprintf(stderr, "linter: selftest csv has no actor_text column\n");
        return 1;
    }

    std::vector<std::pair<std::string, std::string>> cases;
    for (size_t r = 1; r < rows.size() && cases.size() < 10; ++r) {
        if (rows[r].size() <= ai_col) continue;
        std::string actor = strip_cr(rows[r][ai_col]);
        if (!find_first_import(actor).found) continue;
        const std::string id = (id_col != std::string::npos &&
                                rows[r].size() > id_col) ? rows[r][id_col] : "";
        cases.emplace_back(id, actor);
    }

    std::printf("linter mapping selftest\n");
    std::printf("csv=%s\n", csv_path.c_str());
    std::printf("cases=%zu (ASCII prefix must equal actor_text[:snap])\n\n",
                cases.size());

    int pass = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const std::string & id = cases[i].first;
        const std::string & actor = cases[i].second;
        const ImportHit hit = find_first_import(actor);
        const std::vector<llama_token> toks = tokenize(e, actor);
        size_t snap = 0;
        int map_exact = 0;
        const int keep = keep_token_for_char(e, actor, toks, hit.line_start, &snap,
                                             &map_exact);
        const std::string kept = text_of(e, toks, 0, keep);
        const std::string target = actor.substr(0, snap);
        const bool ok = (kept == target);
        if (ok) ++pass;
        std::printf("[%2zu] %-14s char_start=%zu snap=%zu keep=%d kept_len=%zu "
                    "target_len=%zu exact=%d %s\n",
                    i + 1, id.c_str(), hit.line_start, snap, keep, kept.size(),
                    target.size(), map_exact, ok ? "PASS" : "FAIL");
        if (!ok) {
            std::printf("     kept  =\"%s\"\n", esc(kept).c_str());
            std::printf("     target=\"%s\"\n", esc(target).c_str());
        }
    }
    std::printf("\nsummary: %d/%zu PASS\n", pass, cases.size());
    return pass == (int)cases.size() ? 0 : 1;
}

// -------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::string model, data, out, mode_s, cases_s, selftest_csv;
    int limit = 100;
    int actor_max = 192;
    int cont_max = 192;
    uint32_t seed = 1234;
    bool verbose = false;
    bool mapping_selftest_mode = false;

    auto need = [&](int i, const char * flag) -> bool {
        if (i + 1 < argc) return true;
        std::fprintf(stderr, "linter: %s needs a value\n", flag);
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") {
            if (!need(i, "--model")) return 2;
            model = argv[++i];
        } else if (k == "--data") {
            if (!need(i, "--data")) return 2;
            data = argv[++i];
        } else if (k == "--out") {
            if (!need(i, "--out")) return 2;
            out = argv[++i];
        } else if (k == "--limit") {
            if (!need(i, "--limit")) return 2;
            limit = std::atoi(argv[++i]);
        } else if (k == "--actor") {
            if (!need(i, "--actor")) return 2;
            actor_max = std::atoi(argv[++i]);
        } else if (k == "--cont") {
            if (!need(i, "--cont")) return 2;
            cont_max = std::atoi(argv[++i]);
        } else if (k == "--mode") {
            if (!need(i, "--mode")) return 2;
            mode_s = argv[++i];
        } else if (k == "--seed") {
            if (!need(i, "--seed")) return 2;
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (k == "--cases") {
            if (!need(i, "--cases")) return 2;
            cases_s = argv[++i];
        } else if (k == "--mapping-selftest") {
            mapping_selftest_mode = true;
        } else if (k == "--selftest-csv") {
            if (!need(i, "--selftest-csv")) return 2;
            selftest_csv = argv[++i];
        } else if (k == "--verbose") {
            verbose = true;
        } else {
            std::fprintf(stderr, "linter: unknown argument: %s\n", k.c_str());
            return 2;
        }
    }

    if (model.empty()) {
        std::fprintf(stderr, "linter: --model is required\n");
        return 2;
    }
    if (mapping_selftest_mode) {
        if (selftest_csv.empty())
            selftest_csv = "results\\linter_trigger_strong_1234.csv";
        Engine::Config cfg;
        cfg.model = model;
        cfg.n_ctx = 2048;
        cfg.n_seq_max = 2;
        cfg.max_batch = 1024;
        cfg.fa = true;
        cfg.kv_unified = true;
        llama_log_set([](ggml_log_level l, const char * t, void *) {
            if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
        }, nullptr);
        Engine e;
        if (!e.init(cfg)) {
            std::fprintf(stderr, "linter: engine init failed (model=%s)\n", model.c_str());
            return 1;
        }
        const int rc = mapping_selftest(e, selftest_csv);
        e.shutdown();
        return rc;
    }

    if (data.empty() || out.empty()) {
        std::fprintf(stderr, "linter: --data --out are required\n");
        return 2;
    }
    if (limit < 1 || actor_max < 1 || cont_max < 1) {
        std::fprintf(stderr, "linter: --limit/--actor/--cont must be >= 1\n");
        return 2;
    }
    if (mode_s.empty()) mode_s = "all";

    bool do_arm[ARM_COUNT] = {};
    if (mode_s == "all") {
        for (int a = 0; a < ARM_COUNT; ++a) do_arm[a] = true;
    } else {
        size_t p = 0;
        while (p <= mode_s.size()) {
            size_t q = mode_s.find(',', p);
            if (q == std::string::npos) q = mode_s.size();
            const std::string m = mode_s.substr(p, q - p);
            bool found = false;
            for (int a = 0; a < ARM_COUNT; ++a) {
                if (m == ARM_NAMES[a]) { do_arm[a] = true; found = true; break; }
            }
            if (!found) {
                std::fprintf(stderr, "linter: bad --mode entry: %s\n", m.c_str());
                return 2;
            }
            if (q == mode_s.size()) break;
            p = q + 1;
        }
        bool any = false;
        for (int a = 0; a < ARM_COUNT; ++a) any = any || do_arm[a];
        if (!any) {
            std::fprintf(stderr, "linter: --mode has no arms\n");
            return 2;
        }
    }

    std::vector<std::string> case_filter;
    if (!cases_s.empty()) {
        size_t p = 0;
        while (p <= cases_s.size()) {
            size_t q = cases_s.find(',', p);
            if (q == std::string::npos) q = cases_s.size();
            const std::string id = cases_s.substr(p, q - p);
            if (id.empty()) {
                std::fprintf(stderr, "linter: empty entry in --cases\n");
                return 2;
            }
            case_filter.push_back(id);
            if (q == cases_s.size()) break;
            p = q + 1;
        }
    }

    std::vector<Case> all_cases;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) {
            std::fprintf(stderr, "linter: cannot open --data: %s\n", data.c_str());
            return 1;
        }
        std::string line;
        int lineno = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (f.size() < 5) {
                std::fprintf(stderr, "linter: data line %d has %zu columns (need 5)\n",
                             lineno, f.size());
                return 1;
            }
            if (lineno == 1 && f[0] == "id") continue;
            all_cases.push_back({ f[0], f[1], f[2], f[3], f[4],
                                  f.size() > 5 ? f[5] : std::string(),
                                  (int)all_cases.size() });
        }
    }
    if (all_cases.empty()) {
        std::fprintf(stderr, "linter: no cases read from %s\n", data.c_str());
        return 1;
    }

    std::vector<Case> selected;
    if (!case_filter.empty()) {
        for (const auto & id : case_filter) {
            bool found = false;
            for (const auto & c : all_cases) {
                if (c.id == id) { selected.push_back(c); found = true; break; }
            }
            if (!found) std::fprintf(stderr, "linter: --cases id not found: %s\n", id.c_str());
        }
    } else {
        selected = all_cases;
    }
    if (limit > 0 && (int)selected.size() > limit) selected.resize((size_t)limit);
    if (selected.empty()) {
        std::fprintf(stderr, "linter: no cases selected\n");
        return 1;
    }

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = 2048;
    cfg.n_seq_max = 2;
    cfg.max_batch = 1024;
    cfg.fa = true;
    cfg.kv_unified = true;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "linter: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "linter: cannot open --out: %s\n", out.c_str());
        e.shutdown();
        return 1;
    }
    std::fprintf(f, "id,type,seed,arm,actor_text,cont_text,final_text,first_import,"
                    "first_tests,artifact_import,last_tests,artifact_correct,"
                    "import_pos,char_start,keep_token,keep_pos,partial,map_exact,"
                    "rm_ok,rm_us,pos_max_before,pos_max_after,flagged,actor_ms,"
                    "cont_ms,total_ms,template,inject_text,prefill_ms,"
                    "actor_prefill_ms\n");

    const std::string sys =
        "You are a Python programmer. Output only one ```python code block and nothing else.";

    int n_trigger = 0, n_rolled = 0, n_rm_fail = 0, n_map_inexact = 0;
    const auto t_run0 = Clock::now();

    for (int ci = 0; ci < (int)selected.size(); ++ci) {
        const Case & c = selected[(size_t)ci];
        const uint32_t cseed = seed + (uint32_t)c.idx;
        const auto t_case = Clock::now();

        e.clear();
        e.reset_processes();

        std::vector<llama_token> aptoks = chat_prompt(e, sys, c.prompt);
        if (aptoks.empty()) {
            std::fprintf(stderr, "linter: actor tokenization failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }

        const auto ta0 = Clock::now();
        if (!e.prefill(0, aptoks, true)) {
            std::fprintf(stderr, "linter: actor prefill failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        e.sync();
        const double actor_prefill_ms = ms_since(ta0);
        const llama_pos pos_start = (llama_pos)aptoks.size();
        GenOut actor = generate(e, 0, pos_start, actor_max, cseed);
        e.sync();
        const double actor_ms = ms_since(ta0);
        if (actor.failed) {
            std::fprintf(stderr, "linter: actor decode failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        const llama_pos pos_current = pos_start + (llama_pos)actor.n;
        const ImportHit hit = find_first_import(actor.text);
        const int first_import = hit.found ? 1 : 0;
        const int import_pos = classify_import_pos(actor.text, hit);
        const char * import_pos_s = import_pos_name(import_pos);
        if (first_import) ++n_trigger;

        auto restore_actor = [&]() {
            llama_memory_seq_rm(e.mem(), 0, -1, -1);
            llama_memory_seq_cp(e.mem(), 1, 0, -1, -1);
        };
        if (first_import) e.seq_cp(0, 1);

        for (int ai = 0; ai < ARM_COUNT; ++ai) {
            if (!do_arm[ai]) continue;
            ArmRow r;
            r.arm = ARM_NAMES[ai];
            r.first_import = first_import;
            r.flagged = first_import;
            r.actor_prefill_ms = actor_prefill_ms;

            if (!first_import) {
                r.final_text = actor.text;
                r.total_ms = actor_ms;
                write_row(f, c, r, actor.text, import_pos_s, actor_ms, seed);
                continue;
            }

            if (ai == ARM_CONTROL) {
                r.final_text = actor.text;
                r.total_ms = actor_ms;
                write_row(f, c, r, actor.text, import_pos_s, actor_ms, seed);
                continue;
            }

            if (ai == ARM_FLAGONLY) {
                r.final_text = actor.text + "\n# LINT ERROR: forbidden import\n";
                r.total_ms = actor_ms;
                write_row(f, c, r, actor.text, import_pos_s, actor_ms, seed);
                continue;
            }

            // Plain retry: throw away the actor KV and re-send the original
            // prompt from scratch, then inject the same user correction as
            // full_userturn. This is the standard API reprompt baseline.
            if (ai == ARM_PLAIN_RETRY) {
                r.inject = arm_inject(true, true, hit, c, "");
                r.char_start = 0;
                r.keep_token = 0;
                r.keep_pos = 0;
                r.partial = 0;
                r.map_exact = 1;
                r.rm_ok = -1;

                e.clear();
                e.reset_processes();

                const auto tp0 = Clock::now();
                const bool pref_ok = !aptoks.empty() && e.prefill(0, aptoks, true);
                e.sync();
                r.prefill_ms = ms_since(tp0);

                const auto tc0 = Clock::now();
                std::vector<llama_token> inj_toks = tokenize(e, r.inject);
                GenOut g;
                if (!pref_ok || inj_toks.empty() ||
                    !e.decode_from(0, inj_toks, pos_start, true)) {
                    g.failed = true;
                } else {
                    e.sync();
                    g = generate(e, 0, pos_start + (llama_pos)inj_toks.size(),
                                 cont_max, cseed);
                    e.sync();
                }
                if (g.failed) {
                    r.cont_ms = -1.0;
                    r.final_text = r.inject;
                    r.total_ms = actor_ms + r.prefill_ms;
                    std::fprintf(stderr, "linter: %s decode failed (id=%s)\n",
                                 r.arm.c_str(), c.id.c_str());
                } else {
                    r.cont = g.text;
                    r.final_text = r.inject + g.text;
                    r.cont_ms = ms_since(tc0);
                    r.total_ms = actor_ms + r.prefill_ms + r.cont_ms;
                }
                write_row(f, c, r, actor.text, import_pos_s, actor_ms, seed);

                if (verbose) {
                    std::printf("  %-17s prefill_ms=%.1f cont_ms=%.1f total_ms=%.1f\n",
                                r.arm.c_str(), r.prefill_ms, r.cont_ms, r.total_ms);
                    std::printf("    inject=\"%s\"\n", esc(r.inject).c_str());
                    std::printf("    final=\"%s\"\n", esc(r.final_text).c_str());
                    std::fflush(stdout);
                }
                continue;
            }

            const bool full = (ai == ARM_FULL_COMMENT || ai == ARM_FULL_USERTURN);
            const bool userturn = (ai == ARM_PARTIAL_USERTURN ||
                                   ai == ARM_FULL_USERTURN);

            size_t char_start = 0;
            if (full) {
                const size_t fence = actor.text.rfind("```python", hit.pos);
                char_start = (fence != std::string::npos) ? fence : 0;
            } else {
                char_start = hit.line_start;
            }

            size_t snap = 0;
            int map_exact = 0;
            const int keep_token = keep_token_for_char(e, actor.text, actor.toks,
                                                       char_start, &snap, &map_exact);
            if (!map_exact) ++n_map_inexact;
            const llama_pos keep_pos = pos_start + (llama_pos)keep_token;
            const std::string kept = text_of(e, actor.toks, 0, keep_token);

            r.char_start = (long long)char_start;
            r.keep_token = keep_token;
            r.keep_pos = (long long)keep_pos;
            r.partial = keep_token > 0 ? 1 : 0;
            r.map_exact = map_exact;
            r.inject = arm_inject(full, userturn, hit, c, kept);

            restore_actor();
            r.pos_max_before = (long long)llama_memory_seq_pos_max(e.mem(), 0);
            const auto r0 = Clock::now();
            r.rm_ok = llama_memory_seq_rm(e.mem(), 0, keep_pos, -1) ? 1 : 0;
            r.rm_us = us_between(r0, Clock::now());
            r.pos_max_after = (long long)llama_memory_seq_pos_max(e.mem(), 0);
            if (r.rm_ok != 1) ++n_rm_fail;
            ++n_rolled;

            const auto tc0 = Clock::now();
            std::vector<llama_token> inj_toks = tokenize(e, r.inject);
            GenOut g;
            if (inj_toks.empty() || !e.decode_from(0, inj_toks, keep_pos, true)) {
                g.failed = true;
            } else {
                e.sync();
                g = generate(e, 0, keep_pos + (llama_pos)inj_toks.size(),
                             cont_max, cseed);
                e.sync();
            }
            if (g.failed) {
                r.cont_ms = -1.0;
                r.final_text = kept;
                r.total_ms = actor_ms + r.rm_us / 1000.0;
                std::fprintf(stderr, "linter: %s decode failed (id=%s)\n",
                             r.arm.c_str(), c.id.c_str());
            } else {
                r.cont = g.text;
                r.final_text = kept + r.inject + g.text;
                r.cont_ms = ms_since(tc0);
                r.total_ms = actor_ms + r.rm_us / 1000.0 + r.cont_ms;
            }
            write_row(f, c, r, actor.text, import_pos_s, actor_ms, seed);

            if (verbose) {
                std::printf("  %-17s char_start=%zu snap=%zu keep=%lld keep_pos=%lld "
                            "partial=%d map_exact=%d rm_ok=%d pos_max=%lld->%lld "
                            "rm_us=%.1f cont_ms=%.1f\n",
                            r.arm.c_str(), char_start, snap, r.keep_token,
                            r.keep_pos, r.partial, r.map_exact, r.rm_ok,
                            r.pos_max_before, r.pos_max_after, r.rm_us, r.cont_ms);
                std::printf("    inject=\"%s\"\n", esc(r.inject).c_str());
                std::printf("    final=\"%s\"\n", esc(r.final_text).c_str());
                std::fflush(stdout);
            }
        }

        std::fflush(f);
        std::printf("[%3d/%3d] %s %-10s actor=%3d import=%d pos=%s case_ms=%.0f\n",
                    ci + 1, (int)selected.size(), c.id.c_str(), c.type.c_str(),
                    actor.n, first_import, import_pos_s, ms_since(t_case));
        std::fflush(stdout);
    }

    std::fclose(f);
    const double run_ms = ms_since(t_run0);
    e.shutdown();

    const int n = (int)selected.size();
    std::printf("\nlinter: cases=%d trigger=%d (%.1f%%) rolled=%d rm_fail=%d "
                "map_inexact=%d wall=%.1fs\n",
                n, n_trigger, n ? 100.0 * n_trigger / n : 0.0, n_rolled,
                n_rm_fail, n_map_inexact, run_ms / 1000.0);

    {
        std::ofstream mf(out + ".meta.txt", std::ios::binary);
        if (mf) {
            mf << "model=" << model << "\n";
            mf << "data=" << data << "\n";
            mf << "out=" << out << "\n";
            mf << "cases=" << n << " limit=" << limit << " actor=" << actor_max
               << " cont=" << cont_max << " mode=" << mode_s << " seed=" << seed << "\n";
            mf << "actor_prompt=chat_template_add_ass_system_output_one_python_block\n";
            mf << "correction=raw_chatml_explicit_im_end_boundary_actor_eog_not_decoded\n";
            mf << "plain_retry=clear_reprefill_original_prompt_same_userturn\n";
            mf << "mapping=snap_line_start_prefix_walk_no_step_back\n";
            mf << "sampling=temp0.6_topk20_topp0.95\n";
            mf << "n_ctx=2048 n_seq_max=2 kv_unified=1 fa=1\n";
            mf << "trigger=" << n_trigger << " rolled=" << n_rolled
               << " rm_fail=" << n_rm_fail << " map_inexact=" << n_map_inexact
               << " wall_s=" << (run_ms / 1000.0) << "\n";
        }
    }
    return 0;
}
