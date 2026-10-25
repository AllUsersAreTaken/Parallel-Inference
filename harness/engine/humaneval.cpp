// RecycleLM HumanEval bench: Full Retry + User Turn on a standard benchmark.
//
// One model (default Llama-3.2-3B-Instruct Q4_K_M), one seed per problem, three
// arms. The arm is what differs; the actor prompt is identical in every arm.
//
//   baseline   - generate <=256 tokens from the chat prompt, extract the code.
//   swarm      - actor as above; a thinker (same model) audits the actor output
//                with the audit user turn appended to the actor turn. If the
//                thinker replies [FATAL: reason], the actor KV is discarded
//                (clear) and the original prompt + one user turn containing the
//                reason is re-prefilled from scratch (full retry, <=256 tokens).
//                Otherwise the actor artifact is kept.
//   concurrent - the thinker starts after 64 actor tokens: seq_cp(0,1), append
//                the audit turn on seq 1, then both sequences decode in the same
//                engine batch (one tick, two tokens). If the actor ends before
//                the thinker, the thinker is tail-waited. Fatal -> full retry.
//
//   oracle     - regenerate the baseline code with the same seed (no retry), put
//                the code plus its public docstring examples in a test-oracle
//                prompt, and read logit(FATAL first piece) - logit(OK first
//                piece) from the first-token logits. Writes its own CSV (see
//                --oracle-out); --actor/--seed must match the baseline run.
//   oracle_forced - same, but the prompt ends "Answer exactly FATAL or OK. Do
//                not explain. Verdict:" and a GBNF grammar
//                root ::= "FATAL" | "OK" constrains the sampler. The F-O score
//                uses token ids for the bare "F" and "O" pieces.
//
// pass1/syntax_ok are placeholders here (the harness cannot execute Python);
// scripts/eval_humaneval.py fills them and writes results/humaneval_bench.csv.
//
// Usage:
//   humaneval.exe --model M --data data\humaneval_100.tsv \
//                 --out results\humaneval_bench_raw.csv --limit 100 \
//                 --mode all --seed 1234 [--smoke] [--verbose]
//   humaneval.exe ... --mode baseline --actor 512 --seed 1234
//   humaneval.exe ... --mode oracle --actor 512 --seed 1234 \
//                 --out results\humaneval_oracle_raw.csv --oracle-tokens 16

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
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static double ms_since(const Clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double ms_between(const Clock::time_point & a, const Clock::time_point & b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string lower_copy(const std::string & s) {
    std::string o = s;
    for (char & c : o) c = (char)std::tolower((unsigned char)c);
    return o;
}

static std::string trim_copy(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
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
    std::string id, entry, prompt, test;
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

// Continuation that closes the open assistant turn and opens a user turn, then
// the next assistant turn. Derived from the model's chat template (the Llama 3
// template trims message content, so the offset is computed against the trimmed
// actor text). The suffix starts with the turn-close special token, so
// tokenizing it alone cannot merge with the actor's last token.
static std::string turn_suffix(Engine & e, const std::string & sys,
                               const std::string & user,
                               const std::string & assistant_text,
                               const std::string & user2) {
    const std::string base = apply_chat(e, { { "system", sys }, { "user", user } }, false);
    const std::string full = apply_chat(e, {
        { "system", sys }, { "user", user },
        { "assistant", assistant_text }, { "user", user2 },
    }, true);
    if (base.empty() || full.empty() || full.size() <= base.size() ||
        full.compare(0, base.size(), base) != 0) {
        return std::string();
    }
    size_t start = base.size();
    const std::string a = trim_copy(assistant_text);
    if (!a.empty()) {
        const size_t q = full.find(a, base.size());
        if (q != std::string::npos) start = q + a.size();
    }
    return full.substr(start);
}

static std::string retry_prompt(Engine & e, const std::string & sys,
                                const std::string & user, const std::string & reason) {
    return apply_chat(e, {
        { "system", sys },
        { "user", user },
        { "user", reason },
    }, true);
}

static bool has_def(const std::string & code, const std::string & entry) {
    const std::string key = "def " + entry;
    size_t p = 0;
    while ((p = code.find(key, p)) != std::string::npos) {
        const size_t q = p + key.size();
        if (q >= code.size()) return true;
        const unsigned char c = (unsigned char)code[q];
        if (!(std::isalnum(c) || c == '_')) return true;
        p = q;
    }
    return false;
}

struct Extract {
    std::string code;
    int stitch = 0;   // 0 as-is, 1 prompt+block, 2 preamble+block
    int fenced = 0;
};

static Extract extract_code(const std::string & text, const std::string & prompt,
                            const std::string & entry) {
    Extract r;
    std::vector<std::string> py, any;
    size_t i = 0;
    while (i < text.size()) {
        const size_t open = text.find("```", i);
        if (open == std::string::npos) break;
        const size_t nl = text.find('\n', open + 3);
        if (nl == std::string::npos) break;
        const std::string lang = lower_copy(trim_copy(text.substr(open + 3, nl - open - 3)));
        const bool is_py = (lang == "py") || (lang.rfind("python", 0) == 0);
        size_t close = text.find("```", nl + 1);
        std::string body;
        if (close == std::string::npos) {
            body = text.substr(nl + 1);
            i = text.size();
        } else {
            body = text.substr(nl + 1, close - nl - 1);
            i = close + 3;
        }
        any.push_back(body);
        if (is_py) py.push_back(body);
        if (i >= text.size()) break;
    }

    if (!py.empty()) { r.code = py.back(); r.fenced = 1; }
    else if (!any.empty()) { r.code = any.back(); r.fenced = 1; }
    else {
        size_t p = 0;
        while (p < text.size()) {
            const size_t e = text.find('\n', p);
            const std::string line = trim_copy(
                text.substr(p, (e == std::string::npos ? text.size() : e) - p));
            if (line.rfind("def ", 0) == 0 || line.rfind("from ", 0) == 0 ||
                line.rfind("import ", 0) == 0 || line.rfind("class ", 0) == 0) {
                r.code = text.substr(p);
                break;
            }
            if (e == std::string::npos) break;
            p = e + 1;
        }
        if (r.code.empty()) r.code = text;
    }

    r.code = trim_copy(r.code);

    if (!has_def(r.code, entry)) {
        if (prompt.find("def " + entry) != std::string::npos) {
            std::string body = r.code;
            if (!body.empty() && body[0] != '\n') body = "\n" + body;
            r.code = prompt + body;
            r.stitch = 1;
        }
    } else {
        const size_t d = prompt.find("def ");
        const std::string pre = (d == std::string::npos) ? std::string() : prompt.substr(0, d);
        std::vector<std::string> imports;
        size_t p = 0;
        while (p < pre.size()) {
            const size_t e = pre.find('\n', p);
            const std::string line = trim_copy(
                pre.substr(p, (e == std::string::npos ? pre.size() : e) - p));
            if (line.rfind("import ", 0) == 0 || line.rfind("from ", 0) == 0)
                imports.push_back(line);
            if (e == std::string::npos) break;
            p = e + 1;
        }
        bool missing = false;
        for (const auto & im : imports) {
            if (r.code.find(im) == std::string::npos) { missing = true; break; }
        }
        if (missing && !pre.empty()) {
            std::string body = r.code;
            if (!body.empty() && body[0] != '\n') body = "\n" + body;
            r.code = pre + body;
            r.stitch = 2;
        }
    }
    return r;
}

struct Verdict {
    bool fatal = false;
    bool parsed = false;
    std::string reason;
    std::string form;
};

// The audit prompt asks for the bracketed form "[FATAL: ...]" / "OK". The model
// in this build emits a bare leading "FATAL: ..." instead, so both are accepted
// and the observed form is recorded (think_form) rather than silently coerced.
static Verdict parse_verdict(const std::string & text) {
    Verdict v;
    const std::string low = lower_copy(text);

    const size_t f = low.find("[fatal");
    if (f != std::string::npos) {
        v.fatal = true;
        v.parsed = true;
        v.form = "bracket_fatal";
        const size_t c = text.find(':', f);
        if (c != std::string::npos) {
            size_t e = text.find(']', c + 1);
            if (e == std::string::npos) e = text.find_first_of("\n\r", c + 1);
            if (e == std::string::npos) e = text.size();
            v.reason = trim_copy(text.substr(c + 1, e - c - 1));
        }
        if (v.reason.empty()) v.reason = trim_copy(text);
        return v;
    }

    // First alphabetic character, skipping whitespace and light markdown.
    size_t s = 0;
    while (s < text.size() && !std::isalpha((unsigned char)text[s])) ++s;
    if (s < text.size() && low.compare(s, 5, "fatal") == 0) {
        v.fatal = true;
        v.parsed = true;
        v.form = "bare_fatal";
        const size_t c = text.find(':', s);
        if (c != std::string::npos) {
            size_t e = text.find_first_of("\n\r", c + 1);
            if (e == std::string::npos) e = text.size();
            v.reason = trim_copy(text.substr(c + 1, e - c - 1));
        }
        if (v.reason.empty()) v.reason = trim_copy(text);
        return v;
    }

    if (low.find("ok") != std::string::npos) {
        v.parsed = true;
        v.form = "ok";
    } else {
        v.form = "unparsed";
    }
    return v;
}

// ---------------- test-aware oracle (Phase 2) ----------------
//
// The baseline code is regenerated with the same seed (no retry). Public
// examples are the docstring lines starting with ">>>" plus their expected
// output lines (everything until a blank line or the next ">>>"). If there are
// none, the docstring text is used. The first-token logits for " FATAL" and
// " OK" are read from the prefill. Llama-3.2 tokenizes " FATAL" as " F" +
// "ATAL" (435, 31174) and " OK" as 10619, so the score uses the first piece of
// each and the exact ids/pieces are written to the CSV.

struct OracleExamples {
    std::string text;
    std::string source;  // docstring_examples | docstring_text
    int n = 0;
};

static std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        const size_t q = s.find('\n', p);
        if (q == std::string::npos) { out.push_back(s.substr(p)); break; }
        out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

static std::string docstring_text(const std::string & prompt) {
    const size_t a = prompt.find("\"\"\"");
    if (a == std::string::npos) return prompt;
    const size_t b = prompt.find("\"\"\"", a + 3);
    if (b == std::string::npos) return prompt;
    return prompt.substr(a + 3, b - a - 3);
}

static OracleExamples oracle_examples(const std::string & prompt) {
    OracleExamples r;
    const auto lines = split_lines(prompt);
    std::vector<std::string> ex;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string t = trim_copy(lines[i]);
        if (t.rfind(">>>", 0) != 0) continue;
        ex.push_back(t);
        ++r.n;
        for (size_t j = i + 1; j < lines.size(); ++j) {
            const std::string u = trim_copy(lines[j]);
            if (u.empty() || u.rfind(">>>", 0) == 0) break;
            ex.push_back(u);
        }
    }
    if (!ex.empty()) {
        std::string joined;
        for (const auto & e : ex) { joined += e; joined += "\n"; }
        r.text = trim_copy(joined);
        r.source = "docstring_examples";
    } else {
        r.text = trim_copy(docstring_text(prompt));
        r.source = "docstring_text";
    }
    return r;
}

static std::string oracle_user(const std::string & code, const OracleExamples & ex) {
    std::string s = "You are a Python test oracle. Here is a function and the public "
                    "examples from its docstring:\n\n";
    s += code;
    s += "\n\n";
    s += (ex.source == "docstring_examples") ? "Public docstring examples:\n"
                                             : "Docstring:\n";
    s += ex.text;
    s += "\n\nStep through each example. If the code would fail any example, answer "
         "FATAL; otherwise OK.\nVerdict:";
    return s;
}

static std::string oracle_user_forced(const std::string & code,
                                      const OracleExamples & ex) {
    std::string s = "You are a Python test oracle. Here is a function and the public "
                    "examples from its docstring:\n\n";
    s += code;
    s += "\n\n";
    s += (ex.source == "docstring_examples") ? "Public docstring examples:\n"
                                             : "Docstring:\n";
    s += ex.text;
    s += "\n\nAnswer exactly FATAL or OK. Do not explain. Verdict:";
    return s;
}

static int first_token_id(Engine & e, const std::string & s, std::string & piece) {
    const auto t = tokenize(e, s);
    if (t.empty()) { piece.clear(); return -1; }
    piece = token_piece(e.vocab(), t[0]);
    return (int)t[0];
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

// Grammar-constrained forced verdict: the model can only emit "FATAL" or "OK".
// The grammar is applied before the usual truncation/sampling chain, so the
// first token must be a prefix of one of those words (F, O, FATAL, OK, ...).
static llama_sampler * make_forced_sampler(Engine & e, uint32_t seed) {
    auto sp = llama_sampler_chain_default_params();
    auto * c = llama_sampler_chain_init(sp);
    if (!c) return nullptr;
    llama_sampler * g = llama_sampler_init_grammar(
        e.vocab(), "root ::= \"FATAL\" | \"OK\"\n", "root");
    if (!g) { llama_sampler_free(c); return nullptr; }
    llama_sampler_chain_add(c, g);
    llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
    llama_sampler_chain_add(c, llama_sampler_init_dist(seed));
    return c;
}

// One stream of the engine. The sampled EOG is never decoded: the turn stays
// open and the continuation supplies the close token.
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
    Clock::time_point t_fin{};

    StreamProc(Engine & en, llama_seq_id s, llama_pos p0, int mn, uint32_t seed,
               bool forced = false)
        : e(en), seq(s), pos(p0), max_new(mn) {
        smpl = forced ? make_forced_sampler(e, seed) : make_sampler(seed);
    }
    ~StreamProc() override { if (smpl) llama_sampler_free(smpl); }

    void seq_ids(std::vector<llama_seq_id> & o) const override { o.push_back(seq); }
    bool done() const override { return fin; }

    void mark_fin(bool end_of_seq) {
        if (fin) return;
        fin = true;
        eos = eos || end_of_seq;
        t_fin = Clock::now();
    }

    void start() {
        if (!smpl) { mark_fin(true); return; }
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

struct RunStats {
    int ticks = 0;
    int joint_ticks = 0;
    bool failed = false;
};

static RunStats run_procs(Engine & e, const std::vector<StreamProc *> & ps, int budget,
                          int stop_after_n = -1, StreamProc * watch = nullptr) {
    RunStats st;
    while (true) {
        bool any = false;
        for (StreamProc * p : ps) if (!p->done()) { any = true; break; }
        if (!any) break;
        if (stop_after_n >= 0 && watch && watch->n >= stop_after_n) break;
        int active = 0;
        const double dt = e.tick(budget, active);
        if (dt < 0) { st.failed = true; break; }
        if (active >= 2) ++st.joint_ticks;
        ++st.ticks;
    }
    return st;
}

struct ArmRow {
    std::string arm;
    std::string code;
    std::string actor_code;
    std::string thinker_text;
    std::string audit_reason;
    std::string note;
    int thinker_fatal = 0;
    int retried = 0;
    int thinker_parsed = 0;
    std::string think_form;
    int actor_n = 0;
    int final_n = 0;
    int thinker_n = 0;
    int stitch = 0;
    int actor_stitch = 0;
    int tail_wait = 0;
    double actor_ms = 0.0;
    double think_ms = 0.0;
    double retry_ms = 0.0;
    double total_ms = 0.0;
    double overlap_ms = 0.0;
    double prefill_ms = 0.0;
    int joint_ticks = 0;
};

int main(int argc, char ** argv) {
    std::string model, data, out, mode_s = "all", oracle_out;
    int limit = 100;
    int actor_max = 256;
    int think_max = 64;
    int fork_at = 64;
    int oracle_max = 16;
    uint32_t seed = 1234;
    bool smoke = false;
    bool verbose = false;

    auto need = [&](int i, const char * flag) -> bool {
        if (i + 1 < argc) return true;
        std::fprintf(stderr, "humaneval: %s needs a value\n", flag);
        return false;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") { if (!need(i, "--model")) return 2; model = argv[++i]; }
        else if (k == "--data") { if (!need(i, "--data")) return 2; data = argv[++i]; }
        else if (k == "--out") { if (!need(i, "--out")) return 2; out = argv[++i]; }
        else if (k == "--limit") { if (!need(i, "--limit")) return 2; limit = std::atoi(argv[++i]); }
        else if (k == "--actor") { if (!need(i, "--actor")) return 2; actor_max = std::atoi(argv[++i]); }
        else if (k == "--think") { if (!need(i, "--think")) return 2; think_max = std::atoi(argv[++i]); }
        else if (k == "--fork") { if (!need(i, "--fork")) return 2; fork_at = std::atoi(argv[++i]); }
        else if (k == "--oracle-tokens") { if (!need(i, "--oracle-tokens")) return 2; oracle_max = std::atoi(argv[++i]); }
        else if (k == "--oracle-out") { if (!need(i, "--oracle-out")) return 2; oracle_out = argv[++i]; }
        else if (k == "--mode" || k == "--arms") { if (!need(i, "--mode")) return 2; mode_s = argv[++i]; }
        else if (k == "--seed") { if (!need(i, "--seed")) return 2; seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10); }
        else if (k == "--smoke") smoke = true;
        else if (k == "--verbose") verbose = true;
        else { std::fprintf(stderr, "humaneval: unknown argument: %s\n", k.c_str()); return 2; }
    }

    if (model.empty() || data.empty() || out.empty()) {
        std::fprintf(stderr, "humaneval: --model --data --out are required\n");
        return 2;
    }
    if (limit < 1 || actor_max < 1 || think_max < 1 || fork_at < 1 || oracle_max < 1) {
        std::fprintf(stderr, "humaneval: --limit/--actor/--think/--fork/--oracle-tokens must be >= 1\n");
        return 2;
    }

    bool do_base = false, do_swarm = false, do_conc = false, do_oracle = false;
    bool do_oracle_forced = false;
    {
        size_t p = 0;
        while (p <= mode_s.size()) {
            size_t q = mode_s.find(',', p);
            if (q == std::string::npos) q = mode_s.size();
            const std::string m = mode_s.substr(p, q - p);
            if (m == "all") { do_base = do_swarm = do_conc = true; }
            else if (m == "baseline") do_base = true;
            else if (m == "swarm") do_swarm = true;
            else if (m == "concurrent") do_conc = true;
            else if (m == "oracle") do_oracle = true;
            else if (m == "oracle_forced") { do_oracle = true; do_oracle_forced = true; }
            else { std::fprintf(stderr, "humaneval: bad --mode entry: %s\n", m.c_str()); return 2; }
            if (q == mode_s.size()) break;
            p = q + 1;
        }
    }
    if (!do_base && !do_swarm && !do_conc && !do_oracle) {
        std::fprintf(stderr, "humaneval: --mode has no arms\n");
        return 2;
    }

    std::vector<Item> all;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) { std::fprintf(stderr, "humaneval: cannot open --data: %s\n", data.c_str()); return 1; }
        std::string line;
        int lineno = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (f.size() < 4) {
                std::fprintf(stderr, "humaneval: data line %d has %zu columns (need 4)\n",
                             lineno, f.size());
                return 1;
            }
            if (lineno == 1 && f[0] == "task_id") continue;
            all.push_back({ unescape_field(f[0]), unescape_field(f[1]),
                            unescape_field(f[2]), unescape_field(f[3]),
                            (int)all.size() });
        }
    }
    if (all.empty()) { std::fprintf(stderr, "humaneval: no tasks read from %s\n", data.c_str()); return 1; }

    if (limit > (int)all.size()) limit = (int)all.size();
    if (smoke && limit > 5) limit = 5;
    all.resize((size_t)limit);

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = 8192;
    cfg.n_seq_max = 4;
    cfg.max_batch = 1024;
    cfg.fa = true;
    cfg.kv_unified = true;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "humaneval: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    FILE * f = nullptr;
    if (do_base || do_swarm || do_conc) {
        f = std::fopen(out.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "humaneval: cannot open --out: %s\n", out.c_str());
            e.shutdown();
            return 1;
        }
        std::fprintf(f, "task_id,arm,idx,entry_point,pass1,syntax_ok,thinker_text,"
                        "thinker_fatal,thinker_parsed,retried,actor_ms,think_ms,retry_ms,"
                        "total_ms,overlap_ms,joint_ticks,prefill_ms,actor_n,final_n,"
                        "thinker_n,stitch,actor_stitch,audit_reason,think_form,tail_wait,note,"
                        "actor_text,actor_code,code\n");
    }

    FILE * fo = nullptr;
    if (do_oracle) {
        std::string opath = oracle_out;
        if (opath.empty())
            opath = (do_base || do_swarm || do_conc) ? out + ".oracle.csv" : out;
        oracle_out = opath;
        fo = std::fopen(opath.c_str(), "w");
        if (!fo) {
            std::fprintf(stderr, "humaneval: cannot open oracle out: %s\n", opath.c_str());
            if (f) std::fclose(f);
            e.shutdown();
            return 1;
        }
        std::fprintf(fo, "task_id,idx,entry_point,seed,actor_max,actor_n,truncated,"
                         "stitch,example_source,n_examples,examples,fatal_token_id,"
                         "fatal_token_piece,ok_token_id,ok_token_piece,logit_fatal,"
                         "logit_ok,score,verdict_text,oracle_fatal,oracle_parsed,"
                         "oracle_n,oracle_ms,total_ms,note,code\n");
    }

    auto close_out = [&]() {
        if (f) { std::fclose(f); f = nullptr; }
        if (fo) { std::fclose(fo); fo = nullptr; }
    };

    int fatal_id = -1, ok_id = -1;
    std::string fatal_piece, ok_piece;
    if (do_oracle) {
        if (do_oracle_forced) {
            fatal_id = first_token_id(e, "F", fatal_piece);
            ok_id = first_token_id(e, "O", ok_piece);
            std::printf("forced oracle grammar: root ::= \"FATAL\" | \"OK\"\n");
            for (const char * s : { "F", "O", "FATAL", "OK" }) {
                std::printf("  '%s' ->", s);
                for (llama_token t : tokenize(e, s))
                    std::printf(" %d:'%s'", (int)t, token_piece(e.vocab(), t).c_str());
                std::printf("\n");
            }
            std::printf("  F-O score uses %d:'%s' - %d:'%s'\n",
                        fatal_id, fatal_piece.c_str(), ok_id, ok_piece.c_str());
        } else {
            fatal_id = first_token_id(e, " FATAL", fatal_piece);
            ok_id = first_token_id(e, " OK", ok_piece);
            std::printf("oracle tokens ' FATAL':");
            for (llama_token t : tokenize(e, " FATAL"))
                std::printf(" %d:'%s'", (int)t, token_piece(e.vocab(), t).c_str());
            std::printf("; ' OK':");
            for (llama_token t : tokenize(e, " OK"))
                std::printf(" %d:'%s'", (int)t, token_piece(e.vocab(), t).c_str());
            std::printf("  first-piece score uses %d:'%s' - %d:'%s'\n",
                        fatal_id, fatal_piece.c_str(), ok_id, ok_piece.c_str());
        }
        std::fflush(stdout);
        if (fatal_id < 0 || ok_id < 0) {
            std::fprintf(stderr, "humaneval: oracle tokenization failed\n");
            close_out();
            e.shutdown();
            return 1;
        }
    }

    const std::string sys =
        "You are a Python programmer. Complete the function. "
        "Output only one ```python code block.";
    const std::string audit_user =
        "You are a Python test oracle. The code above is a solution. Reply exactly "
        "[FATAL: <one-sentence reason>] if it will fail the unit tests, else OK.";

    const auto t_run0 = Clock::now();
    int n_rows = 0, n_fatal = 0, n_retried = 0, n_unparsed = 0, n_short = 0;
    int n_oracle = 0;

    auto write_row = [&](const Item & it, const ArmRow & r, const std::string & actor_text) {
        std::fprintf(f,
                     "%s,%s,%d,%s,%d,%d,%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,"
                     "%d,%d,%d,%d,%d,%s,%s,%d,%s,%s,%s,%s\n",
                     csv_field(it.id).c_str(), csv_field(r.arm).c_str(), it.idx,
                     csv_field(it.entry).c_str(), -1, -1,
                     csv_field(r.thinker_text).c_str(),
                     r.thinker_fatal, r.thinker_parsed, r.retried,
                     r.actor_ms, r.think_ms, r.retry_ms, r.total_ms,
                     r.overlap_ms, r.joint_ticks, r.prefill_ms,
                     r.actor_n, r.final_n, r.thinker_n, r.stitch, r.actor_stitch,
                     csv_field(r.audit_reason).c_str(),
                     csv_field(r.think_form).c_str(), r.tail_wait,
                     csv_field(r.note).c_str(), csv_field(actor_text).c_str(),
                     csv_field(r.actor_code).c_str(),
                     csv_field(r.code).c_str());
        ++n_rows;
    };

    auto write_oracle_row = [&](const Item & it, const std::string & code, int actor_n,
                                int stitch, const OracleExamples & ex,
                                int fid, const std::string & fpiece,
                                int oid, const std::string & opiece,
                                double lf, double lo, double score,
                                const std::string & verdict_text, int ofatal,
                                int oparsed, int on, double oracle_ms, double total_ms,
                                const std::string & note) {
        std::fprintf(fo,
                     "%s,%d,%s,%u,%d,%d,%d,%d,%s,%d,%s,%d,%s,%d,%s,%.6f,%.6f,%.6f,"
                     "%s,%d,%d,%d,%.3f,%.3f,%s,%s\n",
                     csv_field(it.id).c_str(), it.idx, csv_field(it.entry).c_str(), seed,
                     actor_max, actor_n, actor_n >= actor_max ? 1 : 0, stitch,
                     csv_field(ex.source).c_str(), ex.n, csv_field(ex.text).c_str(),
                     fid, csv_field(fpiece).c_str(), oid, csv_field(opiece).c_str(),
                     lf, lo, score, csv_field(verdict_text).c_str(),
                     ofatal, oparsed, on, oracle_ms, total_ms,
                     csv_field(note).c_str(), csv_field(code).c_str());
    };

    for (int ci = 0; ci < (int)all.size(); ++ci) {
        const Item & it = all[(size_t)ci];
        const uint32_t cseed = seed + (uint32_t)it.idx;
        const std::string user = it.prompt + "\nComplete the function.";
        std::vector<llama_token> ptoks = tokenize(e, chat_prompt(e, sys, user));
        if (ptoks.empty()) {
            std::fprintf(stderr, "humaneval: prompt tokenization failed (%s)\n", it.id.c_str());
            close_out(); e.shutdown(); return 1;
        }

        const auto t_case = Clock::now();
        std::printf("[%3d/%3d] %s %s\n", ci + 1, (int)all.size(), it.id.c_str(),
                    it.entry.c_str());
        std::fflush(stdout);

        std::string conc_actor;
        double base_ms = 0.0, swarm_ms = 0.0, conc_ms = 0.0;
        int base_n = 0, swarm_actor_n = 0, conc_actor_n = 0;
        int swarm_fatal = 0, conc_fatal = 0;

        // ---------------- baseline ----------------
        if (do_base) {
            ArmRow r;
            r.arm = "baseline";
            const auto t0 = Clock::now();
            e.clear();
            e.reset_processes();
            const auto tp0 = Clock::now();
            if (!e.prefill(0, ptoks, true)) {
                std::fprintf(stderr, "humaneval: baseline prefill failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();
            r.prefill_ms = ms_since(tp0);

            StreamProc actor(e, 0, (llama_pos)ptoks.size(), actor_max, cseed);
            const auto ta0 = Clock::now();
            actor.start();
            e.add(&actor);
            const RunStats st = run_procs(e, { &actor }, 1);
            e.sync();
            if (st.failed || actor.failed) {
                std::fprintf(stderr, "humaneval: baseline actor failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            r.actor_ms = ms_between(ta0, actor.fin ? actor.t_fin : Clock::now());
            r.actor_n = actor.n;
            const Extract ex = extract_code(actor.text, it.prompt, it.entry);
            r.code = ex.code;
            r.actor_code = ex.code;
            r.stitch = ex.stitch;
            r.actor_stitch = ex.stitch;
            r.final_n = actor.n;
            r.total_ms = ms_since(t0);
            e.reset_processes();
            base_ms = r.total_ms;
            base_n = actor.n;
            write_row(it, r, actor.text);
            if (verbose || smoke) {
                std::printf("  baseline  n=%d stitch=%d total_ms=%.0f\n",
                            r.actor_n, r.stitch, r.total_ms);
                std::printf("%s\n", r.code.c_str());
            }
        }

        // ---------------- test-aware oracle (baseline code + docstring examples) ----
        if (do_oracle) {
            const auto t0 = Clock::now();
            e.clear();
            e.reset_processes();
            if (!e.prefill(0, ptoks, true)) {
                std::fprintf(stderr, "humaneval: oracle baseline prefill failed (%s)\n",
                             it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();

            StreamProc actor(e, 0, (llama_pos)ptoks.size(), actor_max, cseed);
            actor.start();
            e.add(&actor);
            const RunStats ast = run_procs(e, { &actor }, 1);
            e.sync();
            if (ast.failed || actor.failed) {
                std::fprintf(stderr, "humaneval: oracle baseline actor failed (%s)\n",
                             it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            const Extract bex = extract_code(actor.text, it.prompt, it.entry);
            const int base_n = actor.n;
            e.reset_processes();

            const OracleExamples ox = oracle_examples(it.prompt);
            const std::string ouser = do_oracle_forced
                ? oracle_user_forced(bex.code, ox)
                : oracle_user(bex.code, ox);
            const std::vector<llama_token> otoks =
                tokenize(e, apply_chat(e, { { "user", ouser } }, true));
            if (otoks.empty()) {
                std::fprintf(stderr, "humaneval: oracle prompt tokenization failed (%s)\n",
                             it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }

            e.clear();
            e.reset_processes();
            const auto tq0 = Clock::now();
            if (!e.prefill(0, otoks, true)) {
                std::fprintf(stderr, "humaneval: oracle prefill failed (%s)\n",
                             it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();
            const float * lg = llama_get_logits_ith(e.ctx(), -1);
            double lf = 0.0, lo = 0.0, score = 0.0;
            if (lg) {
                lf = (double)lg[(size_t)fatal_id];
                lo = (double)lg[(size_t)ok_id];
                score = lf - lo;
            }

            StreamProc oracle(e, 0, (llama_pos)otoks.size(), oracle_max, cseed,
                              do_oracle_forced);
            if (!oracle.smpl) {
                std::fprintf(stderr, "humaneval: oracle sampler init failed (%s)\n",
                             it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            oracle.start();
            e.add(&oracle);
            const RunStats ost = run_procs(e, { &oracle }, 1);
            e.sync();
            if (ost.failed || oracle.failed) {
                std::fprintf(stderr, "humaneval: oracle decode failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            const double oracle_ms = ms_since(tq0);
            const double total_ms = ms_since(t0);
            const Verdict vd = parse_verdict(oracle.text);
            std::string note;
            if (do_oracle_forced) {
                note = "forced_grammar";
                if (!oracle.toks.empty())
                    note += ";first_token=" + std::to_string((int)oracle.toks[0]) +
                            ":'" + token_piece(e.vocab(), oracle.toks[0]) + "'";
            }
            if (!lg) note += ";no_logits";
            else if (!vd.parsed) note += ";unparsed_verdict";
            write_oracle_row(it, bex.code, base_n, bex.stitch, ox,
                             fatal_id, fatal_piece, ok_id, ok_piece,
                             lf, lo, score, oracle.text,
                             vd.fatal ? 1 : 0, vd.parsed ? 1 : 0, oracle.n,
                             oracle_ms, total_ms, note);
            ++n_oracle;
            if (verbose || smoke) {
                std::printf("  oracle  base_n=%d examples=%s(%d) score=%.4f "
                            "logit_fatal=%.4f logit_ok=%.4f verdict=%s n=%d "
                            "first=%d:'%s' oracle_ms=%.0f total_ms=%.0f\n",
                            base_n, ox.source.c_str(), ox.n, score, lf, lo,
                            vd.fatal ? "FATAL" : (vd.parsed ? "OK" : "unparsed"),
                            oracle.n,
                            oracle.toks.empty() ? -1 : (int)oracle.toks[0],
                            oracle.toks.empty() ? "" :
                                token_piece(e.vocab(), oracle.toks[0]).c_str(),
                            oracle_ms, total_ms);
                std::printf("  oracle text: \"%s\"\n", esc(oracle.text).c_str());
            }
            e.reset_processes();
        }

        // ---------------- swarm (serial thinker) ----------------
        if (do_swarm) {
            ArmRow r;
            r.arm = "swarm";
            const auto t0 = Clock::now();
            e.clear();
            e.reset_processes();
            const auto tp0 = Clock::now();
            if (!e.prefill(0, ptoks, true)) {
                std::fprintf(stderr, "humaneval: swarm prefill failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();
            r.prefill_ms = ms_since(tp0);

            StreamProc actor(e, 0, (llama_pos)ptoks.size(), actor_max, cseed);
            const auto ta0 = Clock::now();
            actor.start();
            e.add(&actor);
            const RunStats st = run_procs(e, { &actor }, 1);
            e.sync();
            if (st.failed || actor.failed) {
                std::fprintf(stderr, "humaneval: swarm actor failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            r.actor_ms = ms_between(ta0, actor.fin ? actor.t_fin : Clock::now());
            r.actor_n = actor.n;
            swarm_actor_n = actor.n;
            e.reset_processes();

            const Extract ex0 = extract_code(actor.text, it.prompt, it.entry);
            const auto tt0 = Clock::now();
            e.seq_cp(0, 1);
            const std::string suffix = turn_suffix(e, sys, user, actor.text, audit_user);
            std::vector<llama_token> stoks = tokenize(e, suffix);
            const llama_pos pos0 = (llama_pos)ptoks.size() + (llama_pos)actor.n;
            if (suffix.empty() || stoks.empty() ||
                !e.decode_from(1, stoks, pos0, true)) {
                std::fprintf(stderr, "humaneval: swarm audit prefill failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();

            StreamProc thinker(e, 1, pos0 + (llama_pos)stoks.size(), think_max, cseed);
            thinker.start();
            e.add(&thinker);
            const RunStats tst = run_procs(e, { &thinker }, 1);
            e.sync();
            if (tst.failed || thinker.failed) {
                std::fprintf(stderr, "humaneval: thinker failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            r.think_ms = ms_between(tt0, thinker.fin ? thinker.t_fin : Clock::now());
            r.thinker_text = thinker.text;
            r.thinker_n = thinker.n;
            r.note = "serial_not_overlapped";
            e.reset_processes();

            const Verdict vd = parse_verdict(thinker.text);
            r.thinker_fatal = vd.fatal ? 1 : 0;
            r.thinker_parsed = vd.parsed ? 1 : 0;
            r.audit_reason = vd.reason;
            r.think_form = vd.form;
            if (vd.fatal) ++n_fatal;
            if (!vd.parsed) ++n_unparsed;

            r.code = ex0.code;
            r.actor_code = ex0.code;
            r.stitch = ex0.stitch;
            r.actor_stitch = ex0.stitch;
            r.final_n = actor.n;

            if (vd.fatal) {
                const auto tr0 = Clock::now();
                e.clear();
                std::string reason = vd.reason;
                if (reason.empty()) reason = trim_copy(thinker.text);
                if (reason.size() > 240) reason.resize(240);
                const std::string user2 =
                    "Your previous attempt failed the audit: " + reason +
                    ". Write a corrected, complete solution for the same problem. "
                    "Output only one ```python code block.";
                std::vector<llama_token> rtoks = tokenize(e, retry_prompt(e, sys, user, user2));
                if (rtoks.empty() || !e.prefill(0, rtoks, true)) {
                    std::fprintf(stderr, "humaneval: swarm retry prefill failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                e.sync();
                StreamProc ra(e, 0, (llama_pos)rtoks.size(), actor_max, cseed);
                ra.start();
                e.add(&ra);
                const RunStats rst = run_procs(e, { &ra }, 1);
                e.sync();
                if (rst.failed || ra.failed) {
                    std::fprintf(stderr, "humaneval: swarm retry failed (%s)\n", it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                const Extract rex = extract_code(ra.text, it.prompt, it.entry);
                r.code = rex.code;
                r.stitch = rex.stitch;
                r.final_n = ra.n;
                r.retried = 1;
                ++n_retried;
                e.reset_processes();
                r.retry_ms = ms_since(tr0);
            }

            r.total_ms = ms_since(t0);
            e.reset_processes();
            swarm_ms = r.total_ms;
            swarm_fatal = r.thinker_fatal;
            write_row(it, r, actor.text);

            if (verbose || smoke) {
                std::printf("  swarm     actor_n=%d fatal=%d parsed=%d retried=%d "
                            "think_n=%d total_ms=%.0f\n",
                            r.actor_n, r.thinker_fatal, r.thinker_parsed, r.retried,
                            r.thinker_n, r.total_ms);
                std::printf("  thinker: \"%s\"\n", esc(r.thinker_text).c_str());
                std::printf("%s\n", r.code.c_str());
            }
        }

        // ---------------- concurrent (two streams, one batch) ----------------
        if (do_conc) {
            ArmRow r;
            r.arm = "concurrent";
            const auto t0 = Clock::now();
            e.clear();
            e.reset_processes();
            const auto tp0 = Clock::now();
            if (!e.prefill(0, ptoks, true)) {
                std::fprintf(stderr, "humaneval: concurrent prefill failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }
            e.sync();
            r.prefill_ms = ms_since(tp0);

            StreamProc actor(e, 0, (llama_pos)ptoks.size(), actor_max, cseed);
            const auto ta0 = Clock::now();
            actor.start();
            e.add(&actor);

            const RunStats pre = run_procs(e, { &actor }, 1, fork_at, &actor);
            if (pre.failed) {
                std::fprintf(stderr, "humaneval: concurrent actor failed (%s)\n", it.id.c_str());
                close_out(); e.shutdown(); return 1;
            }

            bool forked = false;
            double overlap_ms = 0.0;
            int joint_ticks = 0;
            std::string suffix = turn_suffix(e, sys, user, actor.text, audit_user);
            std::vector<llama_token> stoks;
            llama_pos tpos = 0;
            Verdict vd;
            std::string thinker_text;
            int thinker_n = 0;

            if (actor.done()) {
                r.note = "fork_skipped_actor_finished";
                ++n_short;
                // Serial audit on the finished actor, still recorded as the
                // concurrent arm with zero overlap.
                e.seq_cp(0, 1);
                suffix = turn_suffix(e, sys, user, actor.text, audit_user);
                stoks = tokenize(e, suffix);
                tpos = (llama_pos)ptoks.size() + (llama_pos)actor.n;
                const auto tt0 = Clock::now();
                if (stoks.empty() || !e.decode_from(1, stoks, tpos, true)) {
                    std::fprintf(stderr, "humaneval: concurrent audit prefill failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                e.sync();
                StreamProc thinker(e, 1, tpos + (llama_pos)stoks.size(), think_max, cseed);
                thinker.start();
                e.add(&thinker);
                run_procs(e, { &thinker }, 1);
                e.sync();
                r.think_ms = ms_between(tt0, thinker.fin ? thinker.t_fin : Clock::now());
                thinker_text = thinker.text;
                thinker_n = thinker.n;
                vd = parse_verdict(thinker.text);
                e.reset_processes();
            } else {
                forked = true;
                e.seq_cp(0, 1);
                stoks = tokenize(e, suffix);
                tpos = (llama_pos)ptoks.size() + (llama_pos)actor.n;
                const auto tk0 = Clock::now();
                if (stoks.empty() || !e.decode_from(1, stoks, tpos, true)) {
                    std::fprintf(stderr, "humaneval: concurrent audit prefill failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                e.sync();
                const double tpre = ms_since(tk0);

                StreamProc thinker(e, 1, tpos + (llama_pos)stoks.size(), think_max, cseed);
                thinker.start();
                e.add(&thinker);

                const auto tf0 = Clock::now();
                const RunStats js = run_procs(e, { &actor, &thinker }, 2);
                e.sync();
                overlap_ms = ms_between(tf0, std::min(actor.t_fin, thinker.t_fin));
                joint_ticks = js.joint_ticks;
                if (js.failed) {
                    std::fprintf(stderr, "humaneval: concurrent joint decode failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                r.think_ms = tpre + ms_between(tk0, thinker.fin ? thinker.t_fin : Clock::now());
                thinker_text = thinker.text;
                thinker_n = thinker.n;
                vd = parse_verdict(thinker.text);
                r.tail_wait = (thinker.fin && actor.fin &&
                               thinker.t_fin > actor.t_fin) ? 1 : 0;
                e.reset_processes();
            }

            r.actor_ms = ms_between(ta0, actor.fin ? actor.t_fin : Clock::now());
            r.actor_n = actor.n;
            r.thinker_text = thinker_text;
            r.thinker_n = thinker_n;
            r.overlap_ms = overlap_ms;
            r.joint_ticks = joint_ticks;
            r.thinker_fatal = vd.fatal ? 1 : 0;
            r.thinker_parsed = vd.parsed ? 1 : 0;
            r.audit_reason = vd.reason;
            r.think_form = vd.form;
            if (vd.fatal) ++n_fatal;
            if (!vd.parsed) ++n_unparsed;
            if (!forked) r.note = "fork_skipped_actor_finished";

            const Extract ex0 = extract_code(actor.text, it.prompt, it.entry);
            r.code = ex0.code;
            r.actor_code = ex0.code;
            r.stitch = ex0.stitch;
            r.actor_stitch = ex0.stitch;
            r.final_n = actor.n;
            conc_actor = actor.text;
            conc_actor_n = actor.n;

            if (vd.fatal) {
                const auto tr0 = Clock::now();
                e.clear();
                std::string reason = vd.reason;
                if (reason.empty()) reason = trim_copy(thinker_text);
                if (reason.size() > 240) reason.resize(240);
                const std::string user2 =
                    "Your previous attempt failed the audit: " + reason +
                    ". Write a corrected, complete solution for the same problem. "
                    "Output only one ```python code block.";
                std::vector<llama_token> rtoks = tokenize(e, retry_prompt(e, sys, user, user2));
                if (rtoks.empty() || !e.prefill(0, rtoks, true)) {
                    std::fprintf(stderr, "humaneval: concurrent retry prefill failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                e.sync();
                StreamProc ra(e, 0, (llama_pos)rtoks.size(), actor_max, cseed);
                ra.start();
                e.add(&ra);
                const RunStats rst = run_procs(e, { &ra }, 1);
                e.sync();
                if (rst.failed || ra.failed) {
                    std::fprintf(stderr, "humaneval: concurrent retry failed (%s)\n",
                                 it.id.c_str());
                    close_out(); e.shutdown(); return 1;
                }
                const Extract rex = extract_code(ra.text, it.prompt, it.entry);
                r.code = rex.code;
                r.stitch = rex.stitch;
                r.final_n = ra.n;
                r.retried = 1;
                ++n_retried;
                e.reset_processes();
                r.retry_ms = ms_since(tr0);
            }

            r.total_ms = ms_since(t0);
            e.reset_processes();
            conc_ms = r.total_ms;
            conc_fatal = r.thinker_fatal;
            write_row(it, r, conc_actor);

            if (verbose || smoke) {
                std::printf("  concurrent actor_n=%d fatal=%d retried=%d joint=%d "
                            "overlap_ms=%.0f tail=%d total_ms=%.0f\n",
                            r.actor_n, r.thinker_fatal, r.retried, r.joint_ticks,
                            r.overlap_ms, r.tail_wait, r.total_ms);
                std::printf("  thinker: \"%s\"\n", esc(r.thinker_text).c_str());
                std::printf("%s\n", r.code.c_str());
            }
        }

        if (f) std::fflush(f);
        if (fo) std::fflush(fo);
        const double case_ms = ms_since(t_case);
        std::printf("  [case_ms=%.0f] base=%d/%0.0fms swarm=%d/%0.0fms conc=%d/%0.0fms "
                    "actor_n: base=%d swarm=%d conc=%d\n",
                    case_ms, do_base ? 1 : 0, base_ms, swarm_fatal, swarm_ms,
                    conc_fatal, conc_ms, base_n, swarm_actor_n, conc_actor_n);
        std::fflush(stdout);
    }

    close_out();
    const double run_ms = ms_since(t_run0);

    std::printf("done: tasks=%d rows=%d fatal=%d retried=%d unparsed=%d short_actor=%d "
                "wall=%.1fs\n",
                (int)all.size(), n_rows, n_fatal, n_retried, n_unparsed, n_short,
                run_ms / 1000.0);
    if (do_oracle) {
        std::printf("oracle rows=%d out=%s fatal_token=%d:'%s' ok_token=%d:'%s' "
                    "score=logit(first_fatal_piece)-logit(first_ok_piece)\n",
                    n_oracle, oracle_out.c_str(), fatal_id, fatal_piece.c_str(),
                    ok_id, ok_piece.c_str());
    }

    {
        std::ofstream mf(out + ".meta.txt", std::ios::binary);
        if (mf) {
            mf << "model=" << model << "\n";
            mf << "data=" << data << "\n";
            mf << "out=" << out << "\n";
            mf << "tasks=" << (int)all.size() << " arms=" << mode_s
               << " actor=" << actor_max << " think=" << think_max
               << " fork=" << fork_at << " seed=" << seed << "\n";
            mf << "sampling=temp0.6_topk20_topp0.95\n";
            mf << "n_ctx=8192 n_seq_max=4 kv_unified=1 fa=1\n";
            mf << "swarm=serial_not_overlapped concurrent=fork_at_64_two_stream_tick2\n";
            mf << "pass1_placeholder=-1 filled_by=scripts/eval_humaneval.py\n";
            mf << "fatal=" << n_fatal << " retried=" << n_retried
               << " unparsed=" << n_unparsed << " short_actor=" << n_short
               << " wall_s=" << (run_ms / 1000.0) << "\n";
            if (do_oracle) {
                mf << "oracle_out=" << oracle_out << " oracle_tokens=" << oracle_max
                   << " oracle_rows=" << n_oracle << "\n";
                mf << "oracle_fatal_token=" << fatal_id << ":'" << fatal_piece
                   << "' oracle_ok_token=" << ok_id << ":'" << ok_piece << "'\n";
                mf << "oracle_score=logit(first_fatal_piece)-logit(first_ok_piece)\n";
                mf << "oracle_examples=docstring_lines_starting_with_>>>_else_docstring\n";
                mf << "oracle_forced=" << (do_oracle_forced ? 1 : 0)
                   << " oracle_grammar="
                   << (do_oracle_forced ? "root ::= \"FATAL\" | \"OK\"" : "none")
                   << "\n";
            }
        }
    }

    e.shutdown();
    return 0;
}
