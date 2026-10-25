// RecycleLM KV-cache rollback bench.
//
// Primary question: after an actor writes a biography that contains a
// contradictory sentence, can erasing ONLY the offending sentence span from
// the KV cache let the actor regenerate a faithful continuation?
//
// The actor turn is left OPEN by add_ass=true. The actor's EOG is sampled but
// never decoded, so every follow-up prompt starts with an explicit <|im_end|>
// token that closes the actor turn in the KV:
//
//   <|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
//
// Arms (same actor KV, same thinker, same continuation budget):
//   control      - no audit; "Continue the biography."
//   norollback   - audit appended at pos_current, no truncation
//   spanrollback - audit quotes the offending sentence; seq_rm keep_pos..end,
//                  correction instruction decoded at keep_pos
//   tailrollback - legacy: drop the last --rollback (12) tokens
//
// Oracle mode (--mode oracle) removes the thinker from span placement:
//   control           - no audit; "Continue the biography."
//   oracle_norollback - contradiction: unquoted correction instruction appended
//                       at pos_current; clean: continue instruction
//   oracle_span       - contradiction: seq_rm keep_pos..end, where keep_pos is
//                       the deterministic event-2 / y2 sentence; correction
//                       instruction decoded at keep_pos; clean: continue
//
// Usage:
//   rollback.exe --model M --data D --out results\rollback_span_bench.csv \
//                --limit 100 --mode all --actor 96 --think 64 --cont 80 \
//                --seed 1234 --verbose
//
//   rollback.exe ... --cases c001,c002,k001,k002   # explicit case ids
//   rollback.exe ... --smoke                       # first 2 of each type
//   rollback.exe ... --mode spanrollback           # one arm
//
// tailrollback may not be combined with spanrollback in one run (both truncate
// sequence 0 and the actor KV cannot be restored without a re-prefill).

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

static double us_between(const Clock::time_point & a, const Clock::time_point & b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static std::string lower_copy(const std::string & s) {
    std::string o = s;
    for (char & c : o) c = (char)std::tolower((unsigned char)c);
    return o;
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

static std::string trim_copy(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
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
    std::string id, type, tmpl, source, y1, y2, e1, e2;
    int idx = 0;
};

static std::vector<llama_token> tokenize(Engine & e, const std::string & s) {
    std::vector<llama_token> t(s.size() + 32);
    int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
    if (n < 0) {
        t.resize((size_t)(-n));
        n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
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
// No step-back: a token that ends exactly at the snap is retained (the old
// `best - 1` rule dropped the token ending at the rollback point).
static int keep_token_for_char(Engine & e, const std::string & text,
                               const std::vector<llama_token> & toks,
                               size_t char_start, int * map_exact = nullptr) {
    const size_t snap = snap_line_start(text, char_start);

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

struct OracleSpan {
    bool found = false;
    size_t char_start = 0;
    std::string sentence;
};

static std::vector<std::string> oracle_keywords(const std::string & tmpl) {
    if (tmpl == "born_graduation")   return { "born", "graduat" };
    if (tmpl == "founded_retired")   return { "found", "retir" };
    if (tmpl == "prize_death")       return { "prize", "died" };
    if (tmpl == "book_death")        return { "publish", "died" };
    if (tmpl == "married_birth")     return { "marri", "born" };
    if (tmpl == "school_graduation") return { "school", "graduat" };
    if (tmpl == "factory_sold")      return { "built", "sold" };
    if (tmpl == "joined_left")       return { "join", "left" };
    if (tmpl == "wrote_death")       return { "wrote", "died" };
    if (tmpl == "award_birth")       return { "award", "born" };
    return {};
}

// Sentence spans [start, end). Split on ". ", "! ", "? " and "\n"; the
// punctuation stays with the sentence it terminates.
static std::vector<std::pair<size_t, size_t>> split_sentences(const std::string & text) {
    std::vector<std::pair<size_t, size_t>> spans;
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        const bool term = (text[i] == '.' || text[i] == '!' || text[i] == '?') &&
                          i + 1 < text.size() && text[i + 1] == ' ';
        if (term) {
            spans.emplace_back(start, i + 1);
            i += 2;
            start = i;
        } else if (text[i] == '\n') {
            spans.emplace_back(start, i);
            ++i;
            start = i;
        } else {
            ++i;
        }
    }
    if (start < text.size()) spans.emplace_back(start, text.size());
    return spans;
}

// Deterministic span oracle: contradiction cases only. Prefer the sentence
// containing y2; if y2 is absent, the first sentence containing any event-2
// keyword for the template. No match -> no rollback.
static OracleSpan find_oracle_span(const Case & c, const std::string & actor_text) {
    OracleSpan o;
    if (c.type != "contradiction" || actor_text.empty()) return o;
    const auto spans = split_sentences(actor_text);
    if (spans.empty()) return o;

    auto take = [&](size_t s, size_t e) {
        o.found = true;
        o.char_start = s;
        o.sentence = actor_text.substr(s, e - s);
    };

    if (!c.y2.empty()) {
        const size_t p = actor_text.find(c.y2);
        if (p != std::string::npos) {
            for (const auto & sp : spans) {
                if (p >= sp.first && p < sp.second) { take(sp.first, sp.second); return o; }
            }
        }
    }

    const auto kws = oracle_keywords(c.tmpl);
    if (!kws.empty()) {
        const std::string low = lower_copy(actor_text);
        for (const auto & sp : spans) {
            const std::string seg = low.substr(sp.first, sp.second - sp.first);
            for (const std::string & k : kws) {
                if (seg.find(k) != std::string::npos) { take(sp.first, sp.second); return o; }
            }
        }
    }
    return o;
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

// The sampled EOG is never decoded into the KV: the next prompt supplies an
// explicit <|im_end|> at this boundary.
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

static bool has_year(const std::string & text, const std::string & year) {
    if (year.empty()) return false;
    size_t p = 0;
    while ((p = text.find(year, p)) != std::string::npos) {
        const bool lb = (p == 0) || !std::isdigit((unsigned char)text[p - 1]);
        const bool rb = (p + year.size() >= text.size()) ||
                        !std::isdigit((unsigned char)text[p + year.size()]);
        if (lb && rb) return true;
        p += 1;
    }
    return false;
}

static int invented_year_count(const std::string & text, const Case & c) {
    std::vector<std::string> found;
    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
        if (!std::isdigit((unsigned char)text[i])) { ++i; continue; }
        size_t j = i;
        while (j < n && std::isdigit((unsigned char)text[j])) ++j;
        if (j - i == 4) {
            const std::string y = text.substr(i, 4);
            if (y != c.y1 && y != c.y2 &&
                std::find(found.begin(), found.end(), y) == found.end()) {
                found.push_back(y);
            }
        }
        i = j;
    }
    return (int)found.size();
}

static bool contains_explicit(const std::string & text) {
    static const char * keys[] = {
        "impossible", "cannot", "inconsisten", "contradict", "not possible",
        "not the", "rather than", "instead", "should be", "corrected", "error"
    };
    const std::string low = lower_copy(text);
    for (const char * k : keys) if (low.find(k) != std::string::npos) return true;
    return false;
}

static bool contains_needle(const std::string & hay, const std::string & needle) {
    if (needle.empty() || hay.empty()) return false;
    if (hay.find(needle) != std::string::npos) return true;
    if (needle.size() > 40) {
        const std::string pfx = needle.substr(0, 40);
        if (hay.find(pfx) != std::string::npos) return true;
        return lower_copy(hay).find(lower_copy(pfx)) != std::string::npos;
    }
    return lower_copy(hay).find(lower_copy(needle)) != std::string::npos;
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

static std::string turn_prompt(const std::string & user) {
    return "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

struct Audit {
    bool fatal = false;
    std::string quote;
    bool quote_found = false;
    bool used_prefix40 = false;
    size_t char_start = 0;
};

static Audit parse_audit(const std::string & text, const std::string & actor_text) {
    Audit a;
    const std::string low = lower_copy(text);
    const size_t f = low.find("[fatal");
    if (f == std::string::npos) return a;
    a.fatal = true;

    std::string q;
    const size_t q1 = text.find('"', f);
    if (q1 != std::string::npos) {
        const size_t q2 = text.find('"', q1 + 1);
        if (q2 != std::string::npos && q2 > q1) {
            q = text.substr(q1 + 1, q2 - q1 - 1);
        } else {
            size_t e = text.find_first_of("]\n\r", q1 + 1);
            if (e == std::string::npos) e = text.size();
            q = text.substr(q1 + 1, e - q1 - 1);
        }
    }
    if (trim_copy(q).empty()) {
        const size_t c = text.find(':', f);
        if (c != std::string::npos) {
            size_t e = text.find_first_of("]\n\r", c + 1);
            if (e == std::string::npos) e = text.size();
            q = text.substr(c + 1, e - c - 1);
        }
    }
    a.quote = trim_copy(q);
    if (a.quote.empty()) return a;

    size_t p = actor_text.find(a.quote);
    if (p != std::string::npos) {
        a.char_start = p;
        a.quote_found = true;
        return a;
    }
    const std::string alow = lower_copy(actor_text);
    const std::string qlow = lower_copy(a.quote);
    p = alow.find(qlow);
    if (p != std::string::npos) {
        a.char_start = p;
        a.quote_found = true;
        return a;
    }
    if (a.quote.size() > 40) {
        const std::string pfx = a.quote.substr(0, 40);
        p = actor_text.find(pfx);
        if (p == std::string::npos) p = alow.find(lower_copy(pfx));
        if (p != std::string::npos) {
            a.char_start = p;
            a.quote_found = true;
            a.used_prefix40 = true;
        }
    }
    return a;
}

struct ArmResult {
    std::string name;
    std::string cont_text;
    std::string final_text;
    std::string dropped_text;
    std::string note;
    int gen_cont = 0;
    int rolled = 0;
    int rm_ok = -1;
    bool both_dates = false;
    bool expl = false;
    bool bad = false;
    bool bad_cont = false;
    int invented = 0;
    int oracle_found = 0;
    long long char_start = -1;
    int keep_token = -1;
    int partial = 0;
    llama_pos keep_pos = -1;
    llama_pos pos_max_before = -1;
    llama_pos pos_max_after = -1;
    double rm_us = 0.0;
    double cont_ms = 0.0;
    double total_ms = 0.0;
};

static void write_row(FILE * f, const Case & c, const ArmResult & r,
                      const std::string & actor_text, const std::string & thinker_text,
                      const Audit & a, int keep_token, llama_pos keep_pos,
                      llama_pos pos_current, int gen_actor,
                      double actor_ms, double think_ms) {
    std::fprintf(f,
                 "%s,%s,%s,%s,%s,%s,%d,%s,%d,%lld,%d,%lld,%d,%s,%s,%s,%d,%d,%d,%d,"
                 "%.3f,%.3f,%.3f,%.3f,%.3f,%lld,%lld,%d,%s\n",
                 csv_field(c.id).c_str(), csv_field(c.type).c_str(),
                 csv_field(c.tmpl).c_str(), csv_field(r.name).c_str(),
                 csv_field(actor_text).c_str(), csv_field(thinker_text).c_str(),
                 a.fatal ? 1 : 0, csv_field(a.quote).c_str(), a.quote_found ? 1 : 0,
                 a.quote_found ? (long long)a.char_start : -1LL,
                 keep_token, (long long)keep_pos, r.rolled,
                 csv_field(r.dropped_text).c_str(), csv_field(r.final_text).c_str(),
                 csv_field(r.cont_text).c_str(),
                 r.bad ? 1 : 0, r.both_dates ? 1 : 0, r.expl ? 1 : 0, r.invented,
                 actor_ms, think_ms, r.rm_us, r.cont_ms, r.total_ms,
                 (long long)r.pos_max_before, (long long)r.pos_max_after,
                 r.rm_ok, csv_field(r.note).c_str());
}

static void write_oracle_row(FILE * f, const Case & c, const ArmResult & r,
                             const std::string & actor_text, double actor_ms) {
    std::fprintf(f,
                 "%s,%s,%s,%s,%s,%s,%s,%d,%lld,%d,%lld,%d,%d,%d,%.3f,%lld,%lld,%s,"
                 "%d,%d,%d,%d,%.3f,%.3f,%.3f\n",
                 csv_field(c.id).c_str(), csv_field(c.type).c_str(),
                 csv_field(c.tmpl).c_str(), csv_field(r.name).c_str(),
                 csv_field(actor_text).c_str(), csv_field(r.cont_text).c_str(),
                 csv_field(r.final_text).c_str(),
                 r.oracle_found, r.char_start, r.keep_token, (long long)r.keep_pos,
                 r.partial, r.rolled, r.rm_ok, r.rm_us,
                 (long long)r.pos_max_before, (long long)r.pos_max_after,
                 csv_field(r.dropped_text).c_str(), r.bad_cont ? 1 : 0,
                 r.expl ? 1 : 0, r.both_dates ? 1 : 0, r.invented,
                 actor_ms, r.cont_ms, r.total_ms);
}

int main(int argc, char ** argv) {
    std::string model, data, out, mode_s, cases_s;
    int limit = 100;
    int rollback_k = 12;
    int actor_max = 96;
    int think_max = 64;
    int cont_max = 80;
    uint32_t seed = 1234;
    bool smoke = false;
    bool verbose = false;

    auto need = [&](int i, const char * flag) -> bool {
        if (i + 1 < argc) return true;
        std::fprintf(stderr, "rollback: %s needs a value\n", flag);
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
        } else if (k == "--rollback") {
            if (!need(i, "--rollback")) return 2;
            rollback_k = std::atoi(argv[++i]);
        } else if (k == "--actor") {
            if (!need(i, "--actor")) return 2;
            actor_max = std::atoi(argv[++i]);
        } else if (k == "--think") {
            if (!need(i, "--think")) return 2;
            think_max = std::atoi(argv[++i]);
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
        } else if (k == "--smoke") {
            smoke = true;
        } else if (k == "--verbose") {
            verbose = true;
        } else {
            std::fprintf(stderr, "rollback: unknown argument: %s\n", k.c_str());
            return 2;
        }
    }

    if (model.empty() || data.empty() || out.empty()) {
        std::fprintf(stderr, "rollback: --model --data --out are required\n");
        return 2;
    }
    if (limit < 1 || rollback_k < 1 || actor_max < 1 || think_max < 1 || cont_max < 1) {
        std::fprintf(stderr, "rollback: --limit/--rollback/--actor/--think/--cont must be >= 1\n");
        return 2;
    }
    if (mode_s.empty()) mode_s = "all";

    bool do_control = false, do_norollback = false, do_span = false, do_tail = false;
    bool do_oracle = false;
    if (mode_s == "oracle") {
        do_oracle = true;
    } else if (mode_s == "all") {
        do_control = do_norollback = do_span = true;
    } else {
        size_t p = 0;
        while (p <= mode_s.size()) {
            size_t q = mode_s.find(',', p);
            if (q == std::string::npos) q = mode_s.size();
            const std::string m = mode_s.substr(p, q - p);
            if (m == "control") do_control = true;
            else if (m == "norollback") do_norollback = true;
            else if (m == "spanrollback" || m == "rollback") do_span = true;
            else if (m == "tailrollback") do_tail = true;
            else {
                std::fprintf(stderr, "rollback: bad --mode entry: %s\n", m.c_str());
                return 2;
            }
            if (q == mode_s.size()) break;
            p = q + 1;
        }
        if (!do_control && !do_norollback && !do_span && !do_tail) {
            std::fprintf(stderr, "rollback: --mode has no arms\n");
            return 2;
        }
    }
    if (!do_oracle && do_tail && do_span) {
        std::fprintf(stderr, "rollback: tailrollback and spanrollback cannot run in one pass "
                             "(actor KV cannot be restored)\n");
        return 2;
    }

    std::vector<std::string> case_filter;
    if (!cases_s.empty()) {
        size_t p = 0;
        while (p <= cases_s.size()) {
            size_t q = cases_s.find(',', p);
            if (q == std::string::npos) q = cases_s.size();
            const std::string id = cases_s.substr(p, q - p);
            if (id.empty()) {
                std::fprintf(stderr, "rollback: empty entry in --cases\n");
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
            std::fprintf(stderr, "rollback: cannot open --data: %s\n", data.c_str());
            return 1;
        }
        std::string line;
        int lineno = 0;
        while (std::getline(fin, line)) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto f = split_tab(line);
            if (f.size() < 8) {
                std::fprintf(stderr, "rollback: data line %d has %zu columns (need 8)\n",
                             lineno, f.size());
                return 1;
            }
            if (lineno == 1 && f[0] == "id") continue;
            all_cases.push_back({ f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                                  (int)all_cases.size() });
        }
    }
    if (all_cases.empty()) {
        std::fprintf(stderr, "rollback: no cases read from %s\n", data.c_str());
        return 1;
    }

    std::vector<Case> selected;
    if (!case_filter.empty()) {
        for (const auto & id : case_filter) {
            bool found = false;
            for (const auto & c : all_cases) {
                if (c.id == id) { selected.push_back(c); found = true; break; }
            }
            if (!found) std::fprintf(stderr, "rollback: --cases id not found: %s\n", id.c_str());
        }
    } else if (smoke) {
        int nc = 0, nk = 0;
        for (const auto & c : all_cases) {
            if (c.type == "contradiction" && nc < 2) { selected.push_back(c); ++nc; }
            else if (c.type == "clean" && nk < 2) { selected.push_back(c); ++nk; }
            if (nc >= 2 && nk >= 2) break;
        }
    } else {
        selected = all_cases;
    }
    if ((int)selected.size() > limit) selected.resize((size_t)limit);
    if (selected.empty()) {
        std::fprintf(stderr, "rollback: no cases selected\n");
        return 1;
    }

    const int n_ctx = 4096;
    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = n_ctx;
    cfg.n_seq_max = 4;
    cfg.max_batch = 1024;
    cfg.fa = true;
    cfg.kv_unified = true;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "rollback: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "rollback: cannot open --out: %s\n", out.c_str());
        e.shutdown();
        return 1;
    }
    if (do_oracle) {
        std::fprintf(f, "id,type,template,arm,actor_text,cont_text,final_text,"
                        "oracle_found,char_start,keep_token,keep_pos,partial,rolled,"
                        "rm_ok,rm_us,pos_max_before,pos_max_after,dropped_text,"
                        "bad_cont,explicit,both_dates,invented_years,actor_ms,"
                        "cont_ms,total_ms\n");
    } else {
        std::fprintf(f, "id,type,template,arm,actor_text,thinker_text,thinker_fatal,"
                        "quote,quote_found,char_start,keep_token,keep_pos,rolled,"
                        "dropped_text,final_text,cont_text,bad_sentence_present,both_dates,explicit,"
                        "invented_years,actor_ms,think_ms,rm_us,cont_ms,total_ms,"
                        "pos_max_before,pos_max_after,rm_ok,note\n");
    }

    const std::string sys =
        "You are a careful biographer. Use only the facts in the source.";
    const std::string audit_user =
        "You are a strict fact-checker. The biography above was written by an actor; "
        "the source facts are also above. Audit the biography text itself.\n"
        "A factual contradiction exists ONLY if:\n"
        "(1) an event is dated before the person's birth,\n"
        "(2) an event is dated after the person's death, or\n"
        "(3) two events must occur in a fixed order but are dated in reverse order "
        "(e.g., retired before founding, graduated before starting school).\n"
        "Long gaps (20-30 years), unusual ages (e.g., graduating at 21), and plausible "
        "ranges are NOT contradictions.\n"
        "If you find a contradiction, reply exactly:\n"
        "[FATAL: \"<the first offending sentence, copied verbatim from the biography above>\"]\n"
        "If there is no contradiction, reply exactly: OK";

    const std::string ctrl_instr = turn_prompt("Continue the biography.");
    const std::string ok_instr =
        turn_prompt("The fact-checker found no contradiction. Continue the biography.");
    const std::string oracle_continue_instr = turn_prompt("Continue the biography.");
    const std::string oracle_fix_instr =
        "<|im_end|>\n<|im_start|>user\nA factual contradiction was found in the "
        "biography above. Rewrite the biography from the point of the error. State "
        "the inconsistency explicitly and keep the original facts; do not invent or "
        "change any dates.<|im_end|>\n<|im_start|>assistant\n";

    const auto t_run0 = Clock::now();
    int n_rolled = 0, n_fatal = 0, n_quote_found = 0, n_skipped = 0, n_pfx40 = 0;
    int n_oracle_found = 0, n_partial = 0, n_oracle_rolled = 0, n_map_inexact = 0;

    for (int ci = 0; ci < (int)selected.size(); ++ci) {
        const Case & c = selected[(size_t)ci];
        const uint32_t cseed = seed + (uint32_t)c.idx;
        const auto t_case = Clock::now();

        e.clear();
        e.reset_processes();

        const std::string user = c.source +
            "\n\nWrite a short biography (four to six sentences). "
            "You must mention both dates in the source.";
        std::vector<llama_token> aptoks = chat_prompt(e, sys, user);
        if (aptoks.empty()) {
            std::fprintf(stderr, "rollback: actor tokenization failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }

        const auto ta0 = Clock::now();
        if (!e.prefill(0, aptoks, true)) {
            std::fprintf(stderr, "rollback: actor prefill failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        e.sync();
        const llama_pos pos_start = (llama_pos)aptoks.size();
        GenOut actor = generate(e, 0, pos_start, actor_max, cseed);
        e.sync();
        const double actor_ms = ms_since(ta0);
        if (actor.failed) {
            std::fprintf(stderr, "rollback: actor decode failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        const llama_pos pos_current = pos_start + (llama_pos)actor.n;

        if (actor.n < 20) {
            ++n_skipped;
            if (do_oracle) {
                const char * onames[3] = { "control", "oracle_norollback", "oracle_span" };
                for (int a = 0; a < 3; ++a) {
                    ArmResult r;
                    r.name = onames[a];
                    r.final_text = actor.text;
                    r.keep_pos = pos_current;
                    r.invented = invented_year_count(actor.text, c);
                    r.total_ms = actor_ms;
                    write_oracle_row(f, c, r, actor.text, actor_ms);
                }
                std::printf("[%3d/%3d] %s actor=%d SKIP (<20)\n",
                            ci + 1, (int)selected.size(), c.id.c_str(), actor.n);
                std::fflush(f);
                std::fflush(stdout);
                continue;
            }
            const char * names[4] = { "control", "norollback", "tailrollback", "spanrollback" };
            const bool flags[4] = { do_control, do_norollback, do_tail, do_span };
            for (int a = 0; a < 4; ++a) {
                if (!flags[a]) continue;
                ArmResult r;
                r.name = names[a];
                r.final_text = actor.text;
                r.note = "skip_short_actor";
                r.total_ms = actor_ms;
                write_row(f, c, r, actor.text, "", Audit(), actor.n, pos_current,
                          pos_current, actor.n, actor_ms, 0.0);
            }
            std::printf("[%3d/%3d] %s actor=%d SKIP (<20)\n",
                        ci + 1, (int)selected.size(), c.id.c_str(), actor.n);
            std::fflush(f);
            std::fflush(stdout);
            continue;
        }

        if (do_oracle) {
            const OracleSpan osp = find_oracle_span(c, actor.text);
            const int oracle_found = osp.found ? 1 : 0;
            int o_keep_token = -1;
            int o_map_exact = 0;
            llama_pos o_keep_pos = pos_current;
            if (oracle_found) {
                o_keep_token = keep_token_for_char(e, actor.text, actor.toks,
                                                   osp.char_start, &o_map_exact);
                o_keep_pos = pos_start + (llama_pos)o_keep_token;
                ++n_oracle_found;
                if (o_keep_token > 0) ++n_partial;
                if (!o_map_exact) ++n_map_inexact;
            }
            const std::string bad_needle =
                osp.found ? osp.sentence.substr(0, std::min<size_t>(40, osp.sentence.size()))
                          : std::string();

            const bool is_con = (c.type == "contradiction");
            const std::string & fix_instr = is_con ? oracle_fix_instr
                                                   : oracle_continue_instr;
            std::vector<llama_token> ctrl_toks = tokenize(e, oracle_continue_instr);
            std::vector<llama_token> fix_toks = tokenize(e, fix_instr);
            if (ctrl_toks.empty() || fix_toks.empty()) {
                std::fprintf(stderr, "rollback: oracle tokenization failed (id=%s)\n",
                             c.id.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }

            auto run_oracle = [&](const char * name,
                                  const std::vector<llama_token> & instr,
                                  llama_pos p0, const std::string & base, bool rolled,
                                  int rm_ok, double rm_us, llama_pos pmb,
                                  llama_pos pma,
                                  const std::string & dropped) -> ArmResult {
                ArmResult r;
                r.name = name;
                r.oracle_found = oracle_found;
                r.char_start = oracle_found ? (long long)osp.char_start : -1LL;
                r.keep_token = oracle_found ? o_keep_token : -1;
                r.keep_pos = p0;
                r.partial = (oracle_found && o_keep_token > 0) ? 1 : 0;
                r.rolled = rolled ? 1 : 0;
                r.rm_ok = rm_ok;
                r.rm_us = rm_us;
                r.pos_max_before = pmb;
                r.pos_max_after = pma;
                r.dropped_text = dropped;
                const auto tc0 = Clock::now();
                if (!e.decode_from(0, instr, p0, true)) {
                    r.cont_ms = -1.0;
                    r.note = "decode_failed";
                    r.total_ms = actor_ms + rm_us / 1000.0;
                    return r;
                }
                e.sync();
                GenOut g = generate(e, 0, p0 + (llama_pos)instr.size(), cont_max, cseed);
                e.sync();
                r.cont_ms = ms_since(tc0);
                if (g.failed) {
                    r.cont_ms = -1.0;
                    r.note = "decode_failed";
                    r.total_ms = actor_ms + rm_us / 1000.0;
                    return r;
                }
                r.cont_text = g.text;
                r.gen_cont = g.n;
                r.final_text = base + g.text;
                r.bad_cont = !bad_needle.empty() &&
                             r.cont_text.find(bad_needle) != std::string::npos;
                r.expl = contains_explicit(r.cont_text);
                r.both_dates = has_year(r.final_text, c.y1) &&
                               has_year(r.final_text, c.y2);
                r.invented = invented_year_count(r.final_text, c);
                r.total_ms = actor_ms + rm_us / 1000.0 + r.cont_ms;
                return r;
            };

            const ArmResult ctrl = run_oracle("control", ctrl_toks, pos_current,
                                              actor.text, false, -1, 0.0, -1, -1, "");
            llama_memory_seq_rm(e.mem(), 0, pos_current, -1);

            const ArmResult no = run_oracle("oracle_norollback", fix_toks, pos_current,
                                            actor.text, false, -1, 0.0, -1, -1, "");
            llama_memory_seq_rm(e.mem(), 0, pos_current, -1);

            int rm_ok = -1, rolled = 0;
            double rm_us = 0.0;
            llama_pos pmb = -1, pma = -1, sp0 = pos_current;
            std::string sbase = actor.text, sdrop;
            if (is_con && oracle_found) {
                rolled = 1;
                ++n_oracle_rolled;
                pmb = llama_memory_seq_pos_max(e.mem(), 0);
                const auto r0 = Clock::now();
                rm_ok = llama_memory_seq_rm(e.mem(), 0, o_keep_pos, -1) ? 1 : 0;
                rm_us = us_between(r0, Clock::now());
                pma = llama_memory_seq_pos_max(e.mem(), 0);
                sp0 = o_keep_pos;
                sbase = text_of(e, actor.toks, 0, o_keep_token);
                sdrop = text_of(e, actor.toks, o_keep_token, actor.n);
            }
            const ArmResult span = run_oracle("oracle_span", fix_toks, sp0, sbase,
                                              rolled != 0, rm_ok, rm_us, pmb, pma,
                                              sdrop);

            write_oracle_row(f, c, ctrl, actor.text, actor_ms);
            write_oracle_row(f, c, no, actor.text, actor_ms);
            write_oracle_row(f, c, span, actor.text, actor_ms);
            std::fflush(f);

            const double case_ms = ms_since(t_case);
            std::printf("[%3d/%3d] %s %-13s actor=%3d oracle_found=%d char_start=%lld "
                        "keep_token=%d keep_pos=%lld partial=%d rolled=%d rm_ok=%d "
                        "rm_us=%.1f pos_max=%lld->%lld case_ms=%.0f\n",
                        ci + 1, (int)selected.size(), c.id.c_str(), c.type.c_str(),
                        actor.n, oracle_found, (long long)(oracle_found ? osp.char_start : -1),
                        o_keep_token, (long long)(rolled ? o_keep_pos : pos_current),
                        span.partial, rolled, rm_ok, rm_us,
                        (long long)pmb, (long long)pma, case_ms);
            if (verbose || smoke) {
                std::printf("  actor[%d]=\"%s\"\n", actor.n, esc(actor.text).c_str());
                std::printf("  oracle_found=%d char_start=%lld keep_token=%d keep_pos=%lld "
                            "partial=%d\n", oracle_found,
                            (long long)(oracle_found ? osp.char_start : -1),
                            o_keep_token, (long long)(rolled ? o_keep_pos : pos_current),
                            span.partial);
                if (oracle_found) {
                    std::printf("  sentence=\"%s\"\n", esc(osp.sentence).c_str());
                    std::printf("  dropped=\"%s\"\n", esc(sdrop).c_str());
                }
                std::printf("  span: rm_ok=%d pos_max_before=%lld pos_max_after=%lld "
                            "(expect keep_pos-1=%lld)\n",
                            rm_ok, (long long)pmb, (long long)pma,
                            (long long)(rolled ? o_keep_pos - 1 : -1));
                const ArmResult * order[3] = { &ctrl, &no, &span };
                for (const ArmResult * r : order) {
                    std::printf("  %-17s gen=%d rolled=%d bad_cont=%d both=%d expl=%d "
                                "inv=%d ms=%.1f total_ms=%.1f\n",
                                r->name.c_str(), r->gen_cont, r->rolled,
                                r->bad_cont ? 1 : 0, r->both_dates ? 1 : 0,
                                r->expl ? 1 : 0, r->invented, r->cont_ms,
                                r->total_ms);
                    std::printf("  %-17s final=\"%s\"\n", r->name.c_str(),
                                esc(r->final_text).c_str());
                }
                std::fflush(stdout);
            }
            continue;
        }

        e.seq_cp(0, 1);

        std::string thinker_text;
        double think_ms = 0.0;
        {
            const std::string th_prompt = turn_prompt(audit_user);
            std::vector<llama_token> ttoks = tokenize(e, th_prompt);
            if (ttoks.empty()) {
                std::fprintf(stderr, "rollback: thinker tokenization failed (id=%s)\n", c.id.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }
            const auto tt0 = Clock::now();
            if (!e.decode_from(1, ttoks, pos_current, true)) {
                std::fprintf(stderr, "rollback: thinker prompt decode failed (id=%s)\n", c.id.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }
            e.sync();
            GenOut think = generate(e, 1, pos_current + (llama_pos)ttoks.size(), think_max, cseed);
            e.sync();
            think_ms = ms_since(tt0);
            if (think.failed) {
                std::fprintf(stderr, "rollback: thinker generation failed (id=%s)\n", c.id.c_str());
                std::fclose(f); e.shutdown(); return 1;
            }
            thinker_text = think.text;
        }

        const Audit audit = parse_audit(thinker_text, actor.text);
        if (audit.fatal) ++n_fatal;
        if (audit.quote_found) ++n_quote_found;
        if (audit.used_prefix40) ++n_pfx40;

        int span_keep_token = actor.n;
        int span_map_exact = 0;
        llama_pos span_keep_pos = pos_current;
        std::string span_kept_text = actor.text;
        std::string span_dropped_text;
        if (audit.fatal && audit.quote_found) {
            span_keep_token = keep_token_for_char(e, actor.text, actor.toks,
                                                  audit.char_start, &span_map_exact);
            span_keep_pos = pos_start + (llama_pos)span_keep_token;
            span_kept_text = text_of(e, actor.toks, 0, span_keep_token);
            span_dropped_text = text_of(e, actor.toks, span_keep_token, actor.n);
            if (!span_map_exact) ++n_map_inexact;
        }

        const std::string fix_instr = turn_prompt(
            "The fact-checker found this contradiction: " + audit.quote +
            ". Rewrite the biography from this point. State the inconsistency "
            "explicitly and keep the original facts; do not invent or change any dates.");
        const std::string & audit_instr = audit.fatal ? fix_instr : ok_instr;

        std::vector<llama_token> ctrl_toks, audit_toks;
        if (do_control) ctrl_toks = tokenize(e, ctrl_instr);
        if (do_norollback || do_span || do_tail) audit_toks = tokenize(e, audit_instr);
        if (do_control && ctrl_toks.empty()) {
            std::fprintf(stderr, "rollback: control tokenization failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }
        if ((do_norollback || do_span || do_tail) && audit_toks.empty()) {
            std::fprintf(stderr, "rollback: audit tokenization failed (id=%s)\n", c.id.c_str());
            std::fclose(f); e.shutdown(); return 1;
        }

        std::vector<ArmResult> arms;
        int case_span_rolled = 0;
        llama_pos span_posmax_after_rm = -2;

        auto run_arm = [&](const char * name, const std::vector<llama_token> & instr,
                           llama_pos p0, const std::string & base_text, bool uses_think,
                           int rolled, int rm_ok, double rm_us,
                           const std::string & dropped) -> ArmResult {
            ArmResult r;
            r.name = name;
            r.rolled = rolled;
            r.rm_ok = rm_ok;
            r.rm_us = rm_us;
            r.dropped_text = dropped;
            const auto tc0 = Clock::now();
            if (!e.decode_from(0, instr, p0, true)) {
                std::fprintf(stderr, "rollback: %s prompt decode failed (id=%s)\n",
                             name, c.id.c_str());
                r.cont_ms = -1.0;
                r.note = "decode_failed";
                r.total_ms = actor_ms + (uses_think ? think_ms : 0.0) + rm_us / 1000.0;
                return r;
            }
            e.sync();
            GenOut g = generate(e, 0, p0 + (llama_pos)instr.size(), cont_max, cseed);
            e.sync();
            r.cont_ms = ms_since(tc0);
            if (g.failed) {
                std::fprintf(stderr, "rollback: %s generation failed (id=%s)\n",
                             name, c.id.c_str());
                r.cont_ms = -1.0;
                r.note = "decode_failed";
                r.total_ms = actor_ms + (uses_think ? think_ms : 0.0) + rm_us / 1000.0;
                return r;
            }
            r.cont_text = g.text;
            r.gen_cont = g.n;
            r.final_text = base_text + g.text;
            r.bad = contains_needle(r.final_text, audit.quote);
            r.both_dates = has_year(r.final_text, c.y1) && has_year(r.final_text, c.y2);
            r.expl = contains_explicit(r.cont_text);
            r.invented = invented_year_count(r.final_text, c);
            r.total_ms = actor_ms + (uses_think ? think_ms : 0.0) + rm_us / 1000.0 + r.cont_ms;
            return r;
        };

        const auto restore_seq0 = [&]() {
            llama_memory_seq_rm(e.mem(), 0, pos_current, -1);
        };

        if (do_control) {
            arms.push_back(run_arm("control", ctrl_toks, pos_current, actor.text,
                                   false, 0, -1, 0.0, ""));
            restore_seq0();
        }
        if (do_norollback) {
            arms.push_back(run_arm("norollback", audit_toks, pos_current, actor.text,
                                   true, 0, -1, 0.0, ""));
            restore_seq0();
        }
        if (do_tail) {
            const bool tr = audit.fatal && actor.n >= rollback_k + 8;
            const int tk = tr ? actor.n - rollback_k : actor.n;
            const llama_pos kp = tr ? (pos_current - (llama_pos)rollback_k) : pos_current;
            const std::string kept = tr ? text_of(e, actor.toks, 0, tk) : actor.text;
            const std::string dropped = tr ? text_of(e, actor.toks, tk, actor.n) : "";
            int rm_ok = -1;
            double rm_us = 0.0;
            llama_pos pmb = -1, pma = -1;
            if (tr) {
                pmb = llama_memory_seq_pos_max(e.mem(), 0);
                const auto r0 = Clock::now();
                rm_ok = llama_memory_seq_rm(e.mem(), 0, kp, -1) ? 1 : 0;
                rm_us = us_between(r0, Clock::now());
                pma = llama_memory_seq_pos_max(e.mem(), 0);
            }
            ArmResult rr = run_arm("tailrollback", audit_toks, kp, kept, true,
                                   tr ? 1 : 0, rm_ok, rm_us, dropped);
            rr.pos_max_before = pmb;
            rr.pos_max_after = pma;
            arms.push_back(rr);
        }
        if (do_span) {
            const bool sr = audit.fatal && audit.quote_found;
            int rm_ok = -1;
            double rm_us = 0.0;
            llama_pos pmb = -1, pma = -1;
            if (sr) {
                pmb = llama_memory_seq_pos_max(e.mem(), 0);
                const auto r0 = Clock::now();
                rm_ok = llama_memory_seq_rm(e.mem(), 0, span_keep_pos, -1) ? 1 : 0;
                rm_us = us_between(r0, Clock::now());
                pma = llama_memory_seq_pos_max(e.mem(), 0);
                span_posmax_after_rm = pma;
            }
            const llama_pos kp = sr ? span_keep_pos : pos_current;
            const std::string & base = sr ? span_kept_text : actor.text;
            ArmResult rr = run_arm("spanrollback", audit_toks, kp, base, true,
                                   sr ? 1 : 0, rm_ok, rm_us,
                                   sr ? span_dropped_text : "");
            rr.pos_max_before = pmb;
            rr.pos_max_after = pma;
            arms.push_back(rr);
            if (sr) { ++n_rolled; case_span_rolled = 1; }
        }

        for (const auto & r : arms) {
            const int keep_tok = (r.name == "spanrollback" && r.rolled)
                ? span_keep_token : (r.name == "tailrollback" && r.rolled)
                ? actor.n - rollback_k : actor.n;
            const llama_pos keep_p = (r.name == "spanrollback" && r.rolled)
                ? span_keep_pos : (r.name == "tailrollback" && r.rolled)
                ? pos_current - (llama_pos)rollback_k : pos_current;
            write_row(f, c, r, actor.text, thinker_text, audit, keep_tok, keep_p,
                      pos_current, actor.n, actor_ms, think_ms);
        }
        std::fflush(f);

        const double case_ms = ms_since(t_case);
        std::printf("[%3d/%3d] %s %-13s actor=%3d fatal=%d qfound=%d rolled=%d "
                    "keep=%d/%lld q40=%d case_ms=%.0f\n",
                    ci + 1, (int)selected.size(), c.id.c_str(), c.type.c_str(),
                    actor.n, audit.fatal ? 1 : 0, audit.quote_found ? 1 : 0,
                    case_span_rolled,
                    span_keep_token, (long long)span_keep_pos,
                    audit.used_prefix40 ? 1 : 0, case_ms);
        std::fflush(stdout);

        if (verbose || smoke) {
            std::printf("  actor[%d]=\"%s\"\n", actor.n, esc(actor.text).c_str());
            std::printf("  thinker=\"%s\"\n", esc(thinker_text).c_str());
            std::printf("  fatal=%d quote_found=%d pfx40=%d char_start=%lld quote=\"%s\"\n",
                        audit.fatal ? 1 : 0, audit.quote_found ? 1 : 0,
                        audit.used_prefix40 ? 1 : 0,
                        audit.quote_found ? (long long)audit.char_start : -1LL,
                        esc(audit.quote).c_str());
            if (audit.fatal && audit.quote_found) {
                std::printf("  keep_token=%d keep_pos=%lld kept_tokens=%d\n",
                            span_keep_token, (long long)span_keep_pos, span_keep_token);
                std::printf("  retained=\"%s\"\n", esc(span_kept_text).c_str());
                std::printf("  dropped=\"%s\"\n", esc(span_dropped_text).c_str());
            }
            for (const auto & r : arms) {
                if (r.name == "spanrollback") {
                    std::printf("  span: seq_pos_max_after_rm=%lld rm_ok=%d rm_us=%.1f\n",
                                (long long)span_posmax_after_rm, r.rm_ok, r.rm_us);
                }
                std::printf("  %-12s gen=%d rolled=%d bad=%d both=%d expl=%d inv=%d "
                            "ms=%.1f total_ms=%.1f\n",
                            r.name.c_str(), r.gen_cont, r.rolled, r.bad ? 1 : 0,
                            r.both_dates ? 1 : 0, r.expl ? 1 : 0, r.invented,
                            r.cont_ms, r.total_ms);
                std::printf("  %-12s final=\"%s\"\n", r.name.c_str(),
                            esc(r.final_text).c_str());
            }
            std::fflush(stdout);
        }
    }

    std::fclose(f);
    const double run_ms = ms_since(t_run0);

    std::printf("done: cases=%d fatal=%d quote_found=%d pfx40=%d span_rolled=%d "
                "oracle_found=%d oracle_partial=%d oracle_rolled=%d "
                "map_inexact=%d skipped=%d wall=%.1fs\n",
                (int)selected.size(), n_fatal, n_quote_found, n_pfx40,
                n_rolled, n_oracle_found, n_partial, n_oracle_rolled,
                n_map_inexact, n_skipped, run_ms / 1000.0);

    {
        std::ofstream mf(out + ".meta.txt", std::ios::binary);
        if (mf) {
            mf << "model=" << model << "\n";
            mf << "data=" << data << "\n";
            mf << "out=" << out << "\n";
            mf << "cases=" << (int)selected.size() << " limit=" << limit
               << " actor=" << actor_max << " think=" << think_max
               << " cont=" << cont_max << " tail_k=" << rollback_k
               << " mode=" << mode_s << " seed=" << seed << "\n";
            mf << "chatml=explicit_im_end_boundary actor_eog_not_decoded\n";
            mf << "sampling=temp0.6_topk20_topp0.95\n";
            mf << "n_ctx=" << n_ctx << " n_seq_max=4 kv_unified=1 fa=1\n";
            mf << "fatal=" << n_fatal << " quote_found=" << n_quote_found
               << " quote_prefix40=" << n_pfx40 << " span_rolled=" << n_rolled
               << " oracle_found=" << n_oracle_found
               << " oracle_partial=" << n_partial
               << " oracle_rolled=" << n_oracle_rolled
               << " map_inexact=" << n_map_inexact
               << " skipped=" << n_skipped << " wall_s=" << (run_ms / 1000.0) << "\n";
            if (do_oracle) {
                mf << "oracle=event2_sentence char_start=snap_line_start_prefix_walk_no_step_back\n";
                mf << "arms=control,oracle_norollback,oracle_span thinker=off\n";
            }
        }
    }

    e.shutdown();
    return 0;
}
