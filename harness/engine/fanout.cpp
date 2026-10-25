// RecycleLM fan-out probe: N translations from one shared source prefix.
//
//   fanout.exe --model M --data D --rows 1-20 \
//              --langs fra,spa,deu,ita,por,rus,zho,jpn \
//              --mode both|batched|seq|warm|all --cap 512 --out F \
//              [--field 1] [--warmup 1] \
//              [--no-repeat-ngram 4] [--repeat-penalty 1.1]
//
// Outputs:
//   F                    rows: source_id,mode,lang,wall_ms,prefill_ms,ttft_ms,
//                              done_ms,generated,finish,text
//   F.runs.csv           source_id,mode,wall_all_ms,prefill_ms
//   F.steps.csv          source_id,mode,steps,step_p50_ms,step_p90_ms

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
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

static std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

struct Lang {
    std::string code;
    std::string name;
};

static bool lang_from_code(const std::string & code, Lang & out) {
    struct Pair { const char * code; const char * name; };
    static const Pair pairs[] = {
        { "fra", "French" }, { "spa", "Spanish" }, { "deu", "German" },
        { "ita", "Italian" }, { "por", "Portuguese" }, { "rus", "Russian" },
        { "zho", "Chinese" }, { "jpn", "Japanese" }, { "eng", "English" },
    };
    for (const auto & p : pairs) {
        if (code == p.code) { out.code = p.code; out.name = p.name; return true; }
    }
    return false;
}

static bool parse_langs(const std::string & s, std::vector<Lang> & out) {
    out.clear();
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        const std::string code = trim(s.substr(p, q - p));
        if (code.empty()) return false;
        Lang lang;
        if (!lang_from_code(code, lang)) return false;
        out.push_back(lang);
        if (q == s.size()) break;
        p = q + 1;
    }
    return !out.empty();
}

static std::string escape_tsv(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (char c : s) {
        if      (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c == '\r') o += "\\r";
        else                o += c;
    }
    return o;
}

static bool utf8_at(const std::string & s, size_t i, size_t & n) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { n = 1; return true; }
    unsigned cp = 0;
    if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
    else { return false; }
    if (i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k) {
        const unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (cc & 0x3F);
    }
    if (n == 2 && cp < 0x80) return false;
    if (n == 3 && cp < 0x800) return false;
    if (n == 4 && cp < 0x10000) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    if (cp > 0x10FFFF) return false;
    return true;
}

static std::string utf8_sanitize(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        size_t n = 0;
        if (utf8_at(s, i, n)) {
            o.append(s, i, n);
            i += n;
            continue;
        }
        const unsigned char c = (unsigned char)s[i];
        size_t need = 0;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        if (need > 0 && i + need > s.size()) break;
        o += '?';
        ++i;
    }
    return o;
}

static std::string csv_text(const std::string & s) {
    const std::string e = escape_tsv(utf8_sanitize(s));
    std::string o;
    o.reserve(e.size() + 2);
    o += '"';
    for (char c : e) {
        if (c == '"') o += "\"\"";
        else          o += c;
    }
    o += '"';
    return o;
}

static llama_token argmax_token(const float * lg, int nv) {
    int best = 0;
    float bestv = lg[0];
    for (int i = 1; i < nv; ++i) {
        if (lg[i] > bestv) { bestv = lg[i]; best = i; }
    }
    return (llama_token)best;
}

// Divide positive logits and multiply negative logits for tokens already in the
// recent generated window (generated tokens only, never the prompt).
static void apply_repeat_penalty(float * lg, int nv,
                                 const std::vector<llama_token> & toks,
                                 float pen, int window) {
    if (pen == 1.0f || window <= 0 || toks.empty()) return;
    const int begin = std::max(0, (int)toks.size() - window);
    for (int i = begin; i < (int)toks.size(); ++i) {
        const llama_token t = toks[(size_t)i];
        if (t < 0 || t >= nv) continue;
        bool seen = false;
        for (int k = begin; k < i; ++k) {
            if (toks[(size_t)k] == t) { seen = true; break; }
        }
        if (seen) continue;
        float & v = lg[t];
        v = (v > 0.0f) ? (v / pen) : (v * pen);
    }
}

// Standard no-repeat-ngram blocking: if the last n-1 generated tokens match an
// earlier occurrence, forbid the token that followed that occurrence.
static void apply_no_repeat_ngram(float * lg, int nv,
                                  const std::vector<llama_token> & toks, int n) {
    if (n <= 1) return;
    const int k = (int)toks.size();
    if (k < n) return;
    const int m = n - 1;
    for (int i = 0; i + m < k; ++i) {
        bool match = true;
        for (int t = 0; t < m; ++t) {
            if (toks[(size_t)(i + t)] != toks[(size_t)(k - m + t)]) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        const llama_token b = toks[(size_t)(i + m)];
        if (b >= 0 && b < nv) lg[b] = -INFINITY;
    }
}

static bool parse_rows(const std::string & s, int & a, int & b) {
    const size_t dash = s.find('-');
    if (dash == std::string::npos) return false;
    const std::string sa = trim(s.substr(0, dash));
    const std::string sb = trim(s.substr(dash + 1));
    if (sa.empty() || sb.empty()) return false;
    char * ea = nullptr;
    char * eb = nullptr;
    const long la = std::strtol(sa.c_str(), &ea, 10);
    const long lb = std::strtol(sb.c_str(), &eb, 10);
    if (*ea != '\0' || *eb != '\0' || la < 1 || lb < la) return false;
    a = (int)la;
    b = (int)lb;
    return true;
}

static bool field_at(const std::string & line, int field, std::string & out) {
    if (field < 1) return false;
    size_t start = 0;
    for (int f = 1; f < field; ++f) {
        const size_t t = line.find('\t', start);
        if (t == std::string::npos) return false;
        start = t + 1;
    }
    const size_t end = line.find('\t', start);
    out = line.substr(start, end == std::string::npos ? std::string::npos : end - start);
    return true;
}

static bool apply_chat(const char * tmpl, const std::string & sys,
                       const std::string & user, std::string & out) {
    llama_chat_message msgs[2] = {
        { "system", sys.c_str() },
        { "user",   user.c_str() },
    };
    size_t cap = std::max<size_t>(4096, sys.size() * 2 + user.size() * 2 + 1024);
    std::vector<char> buf(cap);
    int n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
    if (n < 0) return false;
    if (n > (int)buf.size()) {
        buf.resize((size_t)n);
        n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
        if (n < 0) return false;
    }
    out.assign(buf.data(), (size_t)n);
    return true;
}

static std::string detokenize(const llama_vocab * vocab,
                              const std::vector<llama_token> & toks,
                              size_t begin, size_t end) {
    std::string o;
    char buf[256];
    for (size_t i = begin; i < end; ++i) {
        const int n = llama_token_to_piece(vocab, toks[i], buf, (int)sizeof(buf), 0, true);
        if (n > 0) o.append(buf, (size_t)n);
    }
    return o;
}

static bool tokenize(const llama_vocab * vocab, const std::string & s,
                     std::vector<llama_token> & out) {
    std::vector<llama_token> t(s.size() + 32);
    int n = llama_tokenize(vocab, s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
    if (n < 0) {
        t.resize((size_t)(-n));
        n = llama_tokenize(vocab, s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
    }
    if (n < 0) return false;
    t.resize((size_t)n);
    out.swap(t);
    return !out.empty();
}

struct SourceTask {
    int id = 0;
    std::string text;
    int common_len = 0;
    std::vector<std::vector<llama_token>> prompts;
    std::vector<int> suffix_len;
};

struct Gen : Process {
    Engine & e;
    int M;
    int cap;
    int n_vocab;
    std::vector<llama_seq_id> seqs;
    std::vector<llama_pos> pos;
    std::vector<llama_token> next;
    std::vector<std::string> text;
    std::vector<bool> fin;
    std::vector<std::string> finish;
    std::vector<int> gen;
    std::vector<double> ttft;
    std::vector<double> done_ms;
    std::vector<int> active_order;
    std::vector<std::vector<llama_token>> gen_toks;
    std::vector<float> scratch;
    Clock::time_point t0;
    int base_ = 0;
    int no_repeat = 0;
    float rep_pen = 1.0f;
    int rep_window = 64;

    Gen(Engine & en, int m, int c, int nv,
        const std::vector<llama_token> & seeds,
        const std::vector<double> & ttft_ms,
        Clock::time_point t0_,
        int no_repeat_ngram, float repeat_penalty, int seq_start = 0)
        : e(en), M(m), cap(c), n_vocab(nv), t0(t0_),
          no_repeat(no_repeat_ngram), rep_pen(repeat_penalty) {
        min_slots = 0;
        seqs.resize((size_t)M);
        pos.resize((size_t)M);
        next.resize((size_t)M);
        text.resize((size_t)M);
        fin.assign((size_t)M, false);
        finish.resize((size_t)M);
        gen.assign((size_t)M, 0);
        gen_toks.resize((size_t)M);
        scratch.resize((size_t)nv);
        ttft = ttft_ms;
        done_ms.assign((size_t)M, -1.0);
        for (int j = 0; j < M; ++j) {
            seqs[j] = (llama_seq_id)(seq_start + j);
            pos[j] = 0;
            next[j] = seeds[(size_t)j];
        }
    }

    llama_token pick_next(int j, const float * lg) {
        std::copy(lg, lg + n_vocab, scratch.begin());
        apply_repeat_penalty(scratch.data(), n_vocab, gen_toks[(size_t)j],
                             rep_pen, rep_window);
        apply_no_repeat_ngram(scratch.data(), n_vocab, gen_toks[(size_t)j],
                              no_repeat);
        return argmax_token(scratch.data(), n_vocab);
    }

    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }

    void start(const std::vector<llama_pos> & p) {
        for (int j = 0; j < M; ++j) pos[(size_t)j] = p[(size_t)j];
    }

    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        active_order.clear();
        for (int j = 0; j < M; ++j) {
            if (fin[(size_t)j]) continue;
            if ((int)out.size() - base >= budget) break;
            out.push_back({ seqs[(size_t)j], pos[(size_t)j], next[(size_t)j], true });
            active_order.push_back(j);
        }
    }

    void on_logits(int idx, const float * lg) override {
        const int k = idx - base_;
        if (k < 0 || k >= (int)active_order.size()) return;
        const int j = active_order[(size_t)k];
        if (j < 0 || j >= M || fin[(size_t)j]) return;

        const llama_token tok = next[(size_t)j];
        pos[(size_t)j] += 1;
        if (llama_vocab_is_eog(e.vocab(), tok)) {
            fin[(size_t)j] = true;
            finish[(size_t)j] = "eos";
            done_ms[(size_t)j] = ms_since(t0);
            return;
        }

        char buf[256];
        const int np = llama_token_to_piece(e.vocab(), tok, buf, (int)sizeof(buf), 0, false);
        if (np > 0) text[(size_t)j].append(buf, (size_t)np);
        gen[(size_t)j] += 1;
        gen_toks[(size_t)j].push_back(tok);

        if (gen[(size_t)j] >= cap) {
            fin[(size_t)j] = true;
            finish[(size_t)j] = "cap";
            done_ms[(size_t)j] = ms_since(t0);
            return;
        }
        next[(size_t)j] = pick_next(j, lg);
    }

    bool done() const override {
        for (bool f : fin) if (!f) return false;
        return true;
    }
};

static void write_row(FILE * f, int id, const char * mode, const char * lang,
                      double wall_ms, double prefill_ms, double ttft_ms,
                      double done_ms, int generated, const char * finish,
                      const std::string & text) {
    std::fprintf(f, "%d,%s,%s,%.3f,%.3f,%.3f,%.3f,%d,%s,%s\n",
                 id, mode, lang, wall_ms, prefill_ms, ttft_ms, done_ms,
                 generated, finish, csv_text(text).c_str());
}

struct StepStat {
    int count = 0;
    double p50 = 0.0;
    double p90 = 0.0;
};

static StepStat step_stat(std::vector<double> & v) {
    StepStat s;
    s.count = (int)v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t i50 = (size_t)(0.50 * (double)(v.size() - 1));
    const size_t i90 = (size_t)(0.90 * (double)(v.size() - 1));
    s.p50 = v[i50];
    s.p90 = v[i90];
    return s;
}

struct ArmResult {
    double wall_ms = 0.0;
    double prefill_ms = 0.0;
    StepStat steps;
};

static bool run_batched_arm(Engine & e, const SourceTask & t, const std::vector<Lang> & langs,
                            int cap, int no_repeat, float rep_pen,
                            bool record, FILE * frows, ArmResult & res) {
    const int N = (int)langs.size();
    e.reset_processes();
    e.clear();
    const auto t0 = Clock::now();

    if (t.common_len > 0) {
        std::vector<llama_token> prefix(t.prompts[0].begin(),
                                        t.prompts[0].begin() + t.common_len);
        if (!e.prefill(0, prefix, false)) return false;
        e.sync();
        for (int j = 1; j < N; ++j) e.seq_cp(0, (llama_seq_id)j);
    }

    std::vector<llama_token> seeds((size_t)N, 0);
    std::vector<double> ttft((size_t)N, 0.0);
    std::vector<llama_token> suffix;
    for (int j = 0; j < N; ++j) {
        const auto & p = t.prompts[(size_t)j];
        suffix.assign(p.begin() + t.common_len, p.end());
        if (!e.decode_from((llama_seq_id)j, suffix, (llama_pos)t.common_len, true)) return false;
        e.sync();
        ttft[(size_t)j] = ms_since(t0);
        const float * lg = llama_get_logits_ith(e.ctx(), -1);
        if (!lg) return false;
        seeds[(size_t)j] = argmax_token(lg, e.n_vocab());
    }
    res.prefill_ms = ms_since(t0);

    std::vector<llama_pos> pos((size_t)N);
    for (int j = 0; j < N; ++j) pos[(size_t)j] = (llama_pos)t.prompts[(size_t)j].size();

    Gen g(e, N, cap, e.n_vocab(), seeds, ttft, t0, no_repeat, rep_pen);
    g.start(pos);
    e.add(&g);

    std::vector<double> steps;
    while (!g.done()) {
        int active = 0;
        const double dt = e.tick(N, active);
        if (dt < 0) return false;
        if (dt > 0) steps.push_back(dt);
    }
    e.sync();
    res.wall_ms = ms_since(t0);
    res.steps = step_stat(steps);

    if (record) {
        for (int j = 0; j < N; ++j) {
            write_row(frows, t.id, "batched", langs[(size_t)j].code.c_str(),
                      res.wall_ms, res.prefill_ms, ttft[(size_t)j],
                      g.done_ms[(size_t)j] < 0 ? res.wall_ms : g.done_ms[(size_t)j],
                      g.gen[(size_t)j],
                      g.finish[(size_t)j].empty() ? "cap" : g.finish[(size_t)j].c_str(),
                      g.text[(size_t)j]);
        }
        std::fprintf(stderr, "[fanout] id=%d batched wall=%.1fms prefill=%.1fms steps=%d p50=%.2fms\n",
                     t.id, res.wall_ms, res.prefill_ms, res.steps.count, res.steps.p50);
    }

    e.reset_processes();
    e.clear();
    return true;
}

static bool run_seq_arm(Engine & e, const SourceTask & t, const std::vector<Lang> & langs,
                        int cap, int no_repeat, float rep_pen,
                        bool record, FILE * frows, ArmResult & res) {
    const int N = (int)langs.size();
    std::vector<double> steps;
    res.wall_ms = 0.0;
    res.prefill_ms = 0.0;

    for (int j = 0; j < N; ++j) {
        e.reset_processes();
        e.clear();
        const auto t0 = Clock::now();

        if (!e.prefill(0, t.prompts[(size_t)j], true)) return false;
        e.sync();
        const double prefill_ms = ms_since(t0);
        const float * lg = llama_get_logits_ith(e.ctx(), -1);
        if (!lg) return false;
        const llama_token seed = argmax_token(lg, e.n_vocab());

        std::vector<llama_token> seeds{ seed };
        std::vector<double> ttft{ prefill_ms };
        Gen g(e, 1, cap, e.n_vocab(), seeds, ttft, t0, no_repeat, rep_pen);
        std::vector<llama_pos> pos{ (llama_pos)t.prompts[(size_t)j].size() };
        g.start(pos);
        e.add(&g);

        while (!g.done()) {
            int active = 0;
            const double dt = e.tick(1, active);
            if (dt < 0) return false;
            if (dt > 0) steps.push_back(dt);
        }
        e.sync();
        const double wall_ms = ms_since(t0);

        if (record) {
            write_row(frows, t.id, "seq", langs[(size_t)j].code.c_str(),
                      wall_ms, prefill_ms, prefill_ms,
                      g.done_ms[0] < 0 ? wall_ms : g.done_ms[0],
                      g.gen[0],
                      g.finish[0].empty() ? "cap" : g.finish[0].c_str(),
                      g.text[0]);
        }
        res.wall_ms += wall_ms;
        res.prefill_ms += prefill_ms;
        e.reset_processes();
    }
    res.steps = step_stat(steps);
    if (record) {
        std::fprintf(stderr, "[fanout] id=%d seq wall=%.1fms prefill=%.1fms steps=%d p50=%.2fms\n",
                     t.id, res.wall_ms, res.prefill_ms, res.steps.count, res.steps.p50);
    }
    e.clear();
    return true;
}

// Warm sequential: share the common-prefix prefill once, then for each language
// copy the prefix into a scratch sequence, decode that language's suffix through
// the same path as the batched arm, and decode the translation single-stream.
// Sum of per-language walls plus the one shared prefill. Isolates decode
// batching from the prefill/suffix path.
static bool run_warm_arm(Engine & e, const SourceTask & t, const std::vector<Lang> & langs,
                         int cap, int no_repeat, float rep_pen,
                         bool record, FILE * frows, ArmResult & res) {
    const int N = (int)langs.size();
    const llama_seq_id scratch = (llama_seq_id)N;
    std::vector<double> steps;
    res.wall_ms = 0.0;
    res.prefill_ms = 0.0;

    e.reset_processes();
    e.clear();
    const auto t0 = Clock::now();

    if (t.common_len > 0) {
        std::vector<llama_token> prefix(t.prompts[0].begin(),
                                        t.prompts[0].begin() + t.common_len);
        if (!e.prefill(0, prefix, false)) return false;
        e.sync();
    }
    res.prefill_ms = ms_since(t0);
    double lang_sum = 0.0;

    for (int j = 0; j < N; ++j) {
        const auto tj = Clock::now();
        e.seq_cp(0, scratch);

        const auto & p = t.prompts[(size_t)j];
        std::vector<llama_token> suffix(p.begin() + t.common_len, p.end());
        if (!e.decode_from(scratch, suffix, (llama_pos)t.common_len, true)) return false;
        e.sync();
        const double ttft = ms_since(tj);
        const float * lg = llama_get_logits_ith(e.ctx(), -1);
        if (!lg) return false;
        const llama_token seed = argmax_token(lg, e.n_vocab());

        std::vector<llama_token> seeds{ seed };
        std::vector<double> ttft_v{ ttft };
        Gen g(e, 1, cap, e.n_vocab(), seeds, ttft_v, tj, no_repeat, rep_pen,
              (int)scratch);
        std::vector<llama_pos> pos{ (llama_pos)p.size() };
        g.start(pos);
        const int h = e.add(&g);

        while (!g.done()) {
            int active = 0;
            const double dt = e.tick(1, active);
            if (dt < 0) return false;
            if (dt > 0) steps.push_back(dt);
        }
        e.sync();
        const double wall_ms = ms_since(tj);

        if (record) {
            write_row(frows, t.id, "warm", langs[(size_t)j].code.c_str(),
                      wall_ms, ttft, ttft,
                      g.done_ms[0] < 0 ? wall_ms : g.done_ms[0],
                      g.gen[0],
                      g.finish[0].empty() ? "cap" : g.finish[0].c_str(),
                      g.text[0]);
        }
        lang_sum += wall_ms;
        e.retire(h);
        e.reset_processes();
    }
    res.wall_ms = res.prefill_ms + lang_sum;
    res.steps = step_stat(steps);

    if (record) {
        std::fprintf(stderr, "[fanout] id=%d warm wall=%.1fms prefill=%.1fms steps=%d p50=%.2fms\n",
                     t.id, res.wall_ms, res.prefill_ms, res.steps.count, res.steps.p50);
    }
    e.clear();
    return true;
}

static bool run_source(Engine & e, const SourceTask & t, const std::vector<Lang> & langs,
                       int cap, int no_repeat, float rep_pen,
                       bool do_batched, bool do_seq, bool do_warm, bool batched_first,
                       bool record, FILE * frows, FILE * fruns, FILE * fsteps) {
    ArmResult batched;
    ArmResult seq;
    ArmResult warm;

    auto run_b = [&]() -> bool {
        return run_batched_arm(e, t, langs, cap, no_repeat, rep_pen, record, frows, batched);
    };
    auto run_s = [&]() -> bool {
        return run_seq_arm(e, t, langs, cap, no_repeat, rep_pen, record, frows, seq);
    };
    auto run_w = [&]() -> bool {
        return run_warm_arm(e, t, langs, cap, no_repeat, rep_pen, record, frows, warm);
    };

    std::vector<int> arms;
    if (do_batched && do_seq && do_warm) {
        if (batched_first) arms = { 0, 1, 2 };
        else               arms = { 2, 1, 0 };
    } else if (do_batched && do_seq) {
        if (batched_first) arms = { 0, 1 };
        else               arms = { 1, 0 };
    } else {
        if (do_batched) arms.push_back(0);
        if (do_seq)     arms.push_back(1);
        if (do_warm)    arms.push_back(2);
    }
    if (arms.empty()) return false;

    for (int a : arms) {
        if (a == 0 && !run_b()) return false;
        if (a == 1 && !run_s()) return false;
        if (a == 2 && !run_w()) return false;
    }

    if (record && fruns) {
        if (do_batched) {
            std::fprintf(fruns, "%d,batched,%.3f,%.3f\n",
                         t.id, batched.wall_ms, batched.prefill_ms);
        }
        if (do_seq) {
            std::fprintf(fruns, "%d,seq,%.3f,%.3f\n",
                         t.id, seq.wall_ms, seq.prefill_ms);
        }
        if (do_warm) {
            std::fprintf(fruns, "%d,warm,%.3f,%.3f\n",
                         t.id, warm.wall_ms, warm.prefill_ms);
        }
        std::fflush(fruns);
    }
    if (record && fsteps) {
        if (do_batched) {
            std::fprintf(fsteps, "%d,batched,%d,%.3f,%.3f\n",
                         t.id, batched.steps.count, batched.steps.p50, batched.steps.p90);
        }
        if (do_seq) {
            std::fprintf(fsteps, "%d,seq,%d,%.3f,%.3f\n",
                         t.id, seq.steps.count, seq.steps.p50, seq.steps.p90);
        }
        if (do_warm) {
            std::fprintf(fsteps, "%d,warm,%d,%.3f,%.3f\n",
                         t.id, warm.steps.count, warm.steps.p50, warm.steps.p90);
        }
        std::fflush(fsteps);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model, data, langs_s, mode = "both", out;
    std::string rows_s;
    std::string dump_prompt;
    int field = 1;
    int cap = 512;
    int warmup = 1;
    int warmup_cap = 32;
    int no_repeat = 4;
    float rep_pen = 1.1f;

    auto bad = [](const std::string & msg) -> int {
        std::fprintf(stderr, "fanout: %s\n", msg.c_str());
        return 2;
    };
    auto need_value = [&](int i) -> bool { return i + 1 < argc; };

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") {
            if (!need_value(i)) return bad("--model needs a value");
            model = argv[++i];
        } else if (k == "--data") {
            if (!need_value(i)) return bad("--data needs a value");
            data = argv[++i];
        } else if (k == "--rows") {
            if (!need_value(i)) return bad("--rows needs a value");
            rows_s = argv[++i];
        } else if (k == "--langs") {
            if (!need_value(i)) return bad("--langs needs a value");
            langs_s = argv[++i];
        } else if (k == "--mode") {
            if (!need_value(i)) return bad("--mode needs a value");
            mode = argv[++i];
        } else if (k == "--cap") {
            if (!need_value(i)) return bad("--cap needs a value");
            cap = std::atoi(argv[++i]);
        } else if (k == "--out") {
            if (!need_value(i)) return bad("--out needs a value");
            out = argv[++i];
        } else if (k == "--field") {
            if (!need_value(i)) return bad("--field needs a value");
            field = std::atoi(argv[++i]);
        } else if (k == "--dump-prompt") {
            if (!need_value(i)) return bad("--dump-prompt needs a value");
            dump_prompt = argv[++i];
        } else if (k == "--warmup") {
            if (!need_value(i)) return bad("--warmup needs a value");
            warmup = std::atoi(argv[++i]);
        } else if (k == "--warmup-cap") {
            if (!need_value(i)) return bad("--warmup-cap needs a value");
            warmup_cap = std::atoi(argv[++i]);
        } else if (k == "--no-repeat-ngram") {
            if (!need_value(i)) return bad("--no-repeat-ngram needs a value");
            no_repeat = std::atoi(argv[++i]);
        } else if (k == "--repeat-penalty") {
            if (!need_value(i)) return bad("--repeat-penalty needs a value");
            rep_pen = (float)std::atof(argv[++i]);
        } else {
            return bad("unknown argument: " + k);
        }
    }

    if (model.empty()) return bad("--model is required");
    if (data.empty()) return bad("--data is required");
    if (langs_s.empty()) return bad("--langs is required");
    if (out.empty()) return bad("--out is required");
    if (mode != "both" && mode != "batched" && mode != "seq" &&
        mode != "warm" && mode != "all") {
        return bad("--mode must be both|batched|seq|warm|all");
    }
    if (cap < 1) return bad("--cap must be >= 1");
    if (field < 1) return bad("--field must be >= 1");
    if (warmup < 0) return bad("--warmup must be >= 0");
    if (warmup_cap < 1) warmup_cap = cap;
    if (no_repeat < 0) return bad("--no-repeat-ngram must be >= 0");
    if (!(rep_pen > 0.0f)) return bad("--repeat-penalty must be > 0");

    std::vector<Lang> langs;
    if (!parse_langs(langs_s, langs)) {
        return bad("--langs must be a comma-separated list from "
                   "fra|spa|deu|ita|por|rus|zho|jpn|eng");
    }
    const int N = (int)langs.size();

    int row_a = 1;
    int row_b = 0x7fffffff;
    if (!rows_s.empty() && !parse_rows(rows_s, row_a, row_b)) {
        return bad("--rows must be A-B with 1 <= A <= B");
    }

    std::vector<std::string> lines;
    {
        std::ifstream fin(data, std::ios::binary);
        if (!fin) return bad("cannot open --data: " + data);
        std::string line;
        while (std::getline(fin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
    }
    if (lines.empty()) return bad("no lines in --data");
    if (rows_s.empty()) {
        row_a = 1;
        row_b = (int)lines.size();
    }
    if (row_b > (int)lines.size()) {
        return bad("--rows " + std::to_string(row_a) + "-" + std::to_string(row_b) +
                   " exceeds data lines " + std::to_string(lines.size()));
    }

    std::vector<SourceTask> tasks;
    for (int i = row_a; i <= row_b; ++i) {
        const std::string & line = lines[(size_t)(i - 1)];
        std::string text;
        if (!field_at(line, field, text)) {
            return bad("data line " + std::to_string(i) + " has no field " + std::to_string(field));
        }
        if (text.empty()) {
            return bad("data line " + std::to_string(i) + " field " + std::to_string(field) + " is empty");
        }
        SourceTask t;
        t.id = i;
        t.text = text;
        tasks.push_back(std::move(t));
    }
    if (tasks.empty()) return bad("no sources selected");

    const std::string sys_prompt =
        "You are a professional translator. Translate the text inside the <source> tags "
        "into the requested language. Output only the translation, with no notes.";

    std::string dump_text;
    std::vector<llama_token> dump_tokens;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.progress_callback = [](float, void *) { return true; };
    llama_model * tm = llama_model_load_from_file(model.c_str(), mp);
    if (!tm) {
        llama_backend_free();
        return bad("tokenizer model load failed: " + model);
    }
    const llama_vocab * tv = llama_model_get_vocab(tm);
    const char * tmpl = llama_model_chat_template(tm, nullptr);
    if (!tmpl) {
        llama_model_free(tm);
        llama_backend_free();
        return bad("model has no chat template");
    }

    int max_need = 0;
    for (auto & t : tasks) {
        t.prompts.resize((size_t)N);
        t.suffix_len.resize((size_t)N);
        for (int j = 0; j < N; ++j) {
            const std::string user = "<source>\n" + t.text + "\n</source>\n\n"
                                     "Target language: " + langs[(size_t)j].name + ".";
            std::string prompt;
            if (!apply_chat(tmpl, sys_prompt, user, prompt)) {
                llama_model_free(tm);
                llama_backend_free();
                return bad("chat template failed for source " + std::to_string(t.id) +
                           " lang " + langs[(size_t)j].code);
            }
            if (!tokenize(tv, prompt, t.prompts[(size_t)j])) {
                llama_model_free(tm);
                llama_backend_free();
                return bad("tokenization failed for source " + std::to_string(t.id) +
                           " lang " + langs[(size_t)j].code);
            }
            if (!dump_prompt.empty() && t.id == tasks[0].id && j == 0) {
                dump_text = prompt;
                dump_tokens = t.prompts[(size_t)j];
            }
        }

        int min_len = (int)t.prompts[0].size();
        for (int j = 1; j < N; ++j) min_len = std::min(min_len, (int)t.prompts[(size_t)j].size());
        int common = min_len;
        for (int j = 1; j < N && common > 0; ++j) {
            int c = 0;
            const auto & a = t.prompts[0];
            const auto & b = t.prompts[(size_t)j];
            while (c < (int)a.size() && c < (int)b.size() && a[(size_t)c] == b[(size_t)c]) ++c;
            common = std::min(common, c);
        }
        if (common >= min_len) common = min_len - 1;
        if (common < 0) common = 0;
        t.common_len = common;

        if (!dump_prompt.empty() && t.id == tasks[0].id) {
            std::ofstream df(dump_prompt, std::ios::binary);
            df << "source_id=" << t.id << "\n"
               << "chars=" << dump_text.size() << "\n"
               << "tokens=" << dump_tokens.size() << "\n"
               << "common_len=" << common << "\n"
               << "shared_prefix_tokens=" << common << "\n"
               << "=== full prompt (lang " << langs[0].code << ") ===\n"
               << dump_text
               << "\n=== suffix after shared prefix (per language) ===\n";
            for (int j = 0; j < N; ++j) {
                const auto & p = t.prompts[(size_t)j];
                const std::string suffix = detokenize(tv, p, (size_t)common, p.size());
                const std::string expected = " " + langs[(size_t)j].name +
                                             ".<|im_end|>\n<|im_start|>assistant\n";
                const bool has_name = suffix.find(langs[(size_t)j].name) != std::string::npos;
                const bool has_eot = suffix.find("<|im_end|>") != std::string::npos;
                const bool has_asst = suffix.find("<|im_start|>assistant") != std::string::npos;
                df << "[" << langs[(size_t)j].code << "] len="
                   << (int)p.size() - common
                   << " check=" << (has_name && has_eot && has_asst ? "PASS" : "FAIL")
                   << " exact=" << (suffix == expected ? "yes" : "no")
                   << " text=" << suffix << "\n";
            }
        }

        int need = common;
        for (int j = 0; j < N; ++j) {
            t.suffix_len[(size_t)j] = (int)t.prompts[(size_t)j].size() - common;
            need += t.suffix_len[(size_t)j] + cap + 1;
        }
        int seq_need = 0;
        for (int j = 0; j < N; ++j) {
            seq_need = std::max(seq_need, (int)t.prompts[(size_t)j].size() + cap + 1);
        }
        max_need = std::max(max_need, std::max(need, seq_need));
    }

    llama_model_free(tm);
    llama_backend_free();

    const int n_ctx = std::max(512, max_need + 64);
    for (const auto & t : tasks) {
        int need = t.common_len;
        for (int v : t.suffix_len) need += v + cap + 1;
        int seq_need = 0;
        for (const auto & p : t.prompts) {
            seq_need = std::max(seq_need, (int)p.size() + cap + 1);
        }
        if (need > n_ctx || seq_need > n_ctx) {
            return bad("context capacity failed for source " + std::to_string(t.id) +
                       " (batched_need=" + std::to_string(need) +
                       " seq_need=" + std::to_string(seq_need) +
                       " n_ctx=" + std::to_string(n_ctx) + ")");
        }
    }

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = n_ctx;
    cfg.n_seq_max = N + 2;
    cfg.max_batch = 2048;
    cfg.fa = true;
    cfg.kv_unified = true;

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "fanout: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    FILE * frows = std::fopen(out.c_str(), "w");
    if (!frows) {
        std::fprintf(stderr, "fanout: cannot open --out: %s\n", out.c_str());
        e.shutdown();
        return 1;
    }
    const std::string runs_path = out + ".runs.csv";
    const std::string steps_path = out + ".steps.csv";
    FILE * fruns = std::fopen(runs_path.c_str(), "w");
    FILE * fsteps = std::fopen(steps_path.c_str(), "w");
    if (!fruns || !fsteps) {
        std::fprintf(stderr, "fanout: cannot open sidecar CSVs\n");
        std::fclose(frows);
        if (fruns) std::fclose(fruns);
        if (fsteps) std::fclose(fsteps);
        e.shutdown();
        return 1;
    }

    std::fprintf(frows, "source_id,mode,lang,wall_ms,prefill_ms,ttft_ms,done_ms,generated,finish,text\n");
    std::fprintf(fruns, "source_id,mode,wall_all_ms,prefill_ms\n");
    std::fprintf(fsteps, "source_id,mode,steps,step_p50_ms,step_p90_ms\n");

    const bool do_batched = (mode == "both" || mode == "batched" || mode == "all");
    const bool do_seq = (mode == "both" || mode == "seq" || mode == "all");
    const bool do_warm = (mode == "warm" || mode == "all");

    std::fprintf(stderr, "[fanout] model=%s data=%s rows=%d-%d field=%d langs=%d cap=%d mode=%s\n",
                 model.c_str(), data.c_str(), row_a, row_b, field, N, cap, mode.c_str());
    std::fprintf(stderr, "[fanout] constraints: no_repeat_ngram=%d repeat_penalty=%.3f window=64 (generated tokens)\n",
                 no_repeat, rep_pen);
    std::fprintf(stderr, "[fanout] n_ctx_requested=%d n_ctx_actual=%u n_seq_max=%d sm_clock=%u\n",
                 n_ctx, llama_n_ctx(e.ctx()), N + 2, e.clock_sm());
    for (int j = 0; j < N; ++j) {
        std::fprintf(stderr, "[fanout] lang[%d]=%s (%s)\n",
                     j, langs[(size_t)j].code.c_str(), langs[(size_t)j].name.c_str());
    }
    for (const auto & t : tasks) {
        std::fprintf(stderr, "[fanout] source=%d common=%d prompt0=%d suffix_sum=%d\n",
                     t.id, t.common_len, (int)t.prompts[0].size(),
                     [&]() { int s = 0; for (int v : t.suffix_len) s += v; return s; }());
    }

    if (warmup > 0) {
        const int wc = std::min(cap, warmup_cap);
        std::fprintf(stderr, "[fanout] warmup on source=%d cap=%d (discarded)\n", tasks[0].id, wc);
        if (!run_source(e, tasks[0], langs, wc, no_repeat, rep_pen,
                        do_batched, do_seq, do_warm,
                        tasks[0].id % 2 == 1, false, nullptr, nullptr, nullptr)) {
            std::fprintf(stderr, "fanout: warmup failed\n");
            std::fclose(frows); std::fclose(fruns); std::fclose(fsteps);
            e.shutdown();
            return 1;
        }
    }

    const auto all0 = Clock::now();
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto & t = tasks[i];
        const bool batched_first = (t.id % 2 == 1);
        std::fprintf(stderr, "[fanout] source=%d start order=%s sm_clock=%u\n",
                     t.id, batched_first ? "batched,seq" : "seq,batched", e.clock_sm());
        if (!run_source(e, t, langs, cap, no_repeat, rep_pen,
                        do_batched, do_seq, do_warm, batched_first,
                        true, frows, fruns, fsteps)) {
            std::fprintf(stderr, "fanout: source %d failed\n", t.id);
            std::fclose(frows); std::fclose(fruns); std::fclose(fsteps);
            e.shutdown();
            return 1;
        }
        std::fflush(frows);
    }
    const double all_ms = ms_since(all0);

    std::fclose(frows);
    std::fclose(fruns);
    std::fclose(fsteps);

    std::fprintf(stderr, "[fanout] done sources=%zu wall_all=%.1fms sm_clock=%u\n",
                 tasks.size(), all_ms, e.clock_sm());
    e.shutdown();
    return 0;
}
