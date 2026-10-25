// RecycleLM Context-Aware Decoding (CAD) conflict probe.
//
// One model, one Engine. Each run decodes the same token on a synchronized
// pair of sequences in one llama_decode:
//   seq 0: system + "Context:\n<C>\n\nQuestion: <Q>"
//   seq 1: system + "Question: <Q>"
// Per step the two logit rows are combined with an adaptive plausibility
// constraint (APC):
//   thr = max(logits_ctx) + ln(beta)      (beta > 0; else -inf)
//   logits*[i] = (1+alpha)*logits_ctx[i] - alpha*logits_prior[i]  if ctx[i] >= thr
//              = -inf                                             otherwise
// The sampled token is fed to BOTH sequences at their own positions.
//
// Arms:
//   P  prior-only baseline: stream 1 only, sample its logits
//   A  RAG baseline:        stream 0 only, sample its logits
//   B  CAD alpha=0.5, beta=0   (no APC)
//   C  CAD alpha=0.5, beta=0.1
//   D  CAD alpha=1.0, beta=0.1
//
// Usage:
//   cad_cli.exe --model M --probes P.json --arm all --max-tokens 40 \
//               --temp 0 --seed 1234 --out F.json
//   --seeds 1234,1235,... runs every arm/probe once per seed in one process.
//   --probe P1 restricts runs to one probe (smoke).
//
// Writes {"ok":true,...,"runs":[...]} to --out and stdout. One run record:
//   arm probe seed temp alpha beta text n_tokens finish prefill_ms decode_ms
//   step_ms[]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "backends/llama_backend.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
using parallel::detail::Engine;
using parallel::detail::Request;

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

static std::string fmt(double v, const char * spec) {
    char b[64];
    std::snprintf(b, sizeof(b), spec, v);
    return b;
}

// ---------------------------------------------------------------- JSON reader
// Small recursive parser for the probe file: objects, arrays, strings, numbers,
// booleans, null. Enough for data/_cad_probes.json without an extra dependency.

struct JVal {
    enum Type { NUL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT } type = NUL;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    const JVal * find(const std::string & k) const {
        for (const auto & kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct JParser {
    const std::string & s;
    size_t i = 0;
    std::string err;
    explicit JParser(const std::string & src) : s(src) {}

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    bool fail(const char * m) { if (err.empty()) err = m; return false; }

    bool parse(JVal & v) {
        ws();
        if (!value(v)) return false;
        ws();
        return true;
    }

    bool value(JVal & v) {
        ws();
        if (i >= s.size()) return fail("unexpected end");
        const char c = s[i];
        if (c == '{') return object(v);
        if (c == '[') return array(v);
        if (c == '"') { v.type = JVal::STRING; return string(v.str); }
        if (s.compare(i, 4, "true") == 0)  { i += 4; v.type = JVal::BOOLEAN; v.b = true;  return true; }
        if (s.compare(i, 5, "false") == 0) { i += 5; v.type = JVal::BOOLEAN; v.b = false; return true; }
        if (s.compare(i, 4, "null") == 0)  { i += 4; v.type = JVal::NUL; return true; }
        return number(v);
    }

    bool number(JVal & v) {
        const size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) ++i;
        if (i == start) return fail("expected value");
        v.type = JVal::NUMBER;
        v.num = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return true;
    }

    bool string(std::string & out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i;
        out.clear();
        while (i < s.size()) {
            const unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c != '\\') { out += (char)c; continue; }
            if (i >= s.size()) return fail("truncated escape");
            const char e = s[i++];
            switch (e) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (i + 1 >= s.size() || s[i] != '\\' || s[i + 1] != 'u') return fail("bad surrogate");
                        i += 2;
                        unsigned lo = 0;
                        if (!hex4(lo)) return false;
                        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    }
                    utf8(cp, out);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    bool hex4(unsigned & cp) {
        if (i + 4 > s.size()) return fail("short \\u escape");
        cp = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            unsigned d = 0;
            if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a') + 10;
            else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A') + 10;
            else return fail("bad hex digit");
            cp = (cp << 4) | d;
        }
        return true;
    }

    static void utf8(unsigned cp, std::string & out) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    bool array(JVal & v) {
        v.type = JVal::ARRAY;
        ++i;
        ws();
        if (i < s.size() && s[i] == ']') { ++i; return true; }
        while (true) {
            JVal e;
            if (!value(e)) return false;
            v.arr.push_back(std::move(e));
            ws();
            if (i >= s.size()) return fail("unterminated array");
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == ']') { ++i; return true; }
            return fail("expected , or ]");
        }
    }

    bool object(JVal & v) {
        v.type = JVal::OBJECT;
        ++i;
        ws();
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        while (true) {
            ws();
            std::string key;
            if (!string(key)) return false;
            ws();
            if (i >= s.size() || s[i] != ':') return fail("expected :");
            ++i;
            JVal val;
            if (!value(val)) return false;
            v.obj.emplace_back(std::move(key), std::move(val));
            ws();
            if (i >= s.size()) return fail("unterminated object");
            if (s[i] == ',') { ++i; continue; }
            if (s[i] == '}') { ++i; return true; }
            return fail("expected , or }");
        }
    }
};

static std::string jstr(const JVal & o, const char * k) {
    const JVal * v = o.find(k);
    return v && v->type == JVal::STRING ? v->str : std::string();
}

static std::vector<std::string> jstrs(const JVal & o, const char * k) {
    std::vector<std::string> out;
    const JVal * v = o.find(k);
    if (!v || v->type != JVal::ARRAY) return out;
    for (const auto & e : v->arr) if (e.type == JVal::STRING) out.push_back(e.str);
    return out;
}

// ---------------------------------------------------------------- probes

struct Probe {
    std::string id, context, question, prior, target;
    std::vector<std::string> prior_aliases, target_aliases;
};

static bool load_probes(const std::string & path, std::vector<Probe> & probes, std::string & err) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) { err = "cannot open " + path; return false; }
    std::string text((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    JVal root;
    JParser p(text);
    if (!p.parse(root)) { err = path + ": " + (p.err.empty() ? "parse error" : p.err); return false; }

    const JVal * list = nullptr;
    if (root.type == JVal::ARRAY) list = &root;
    else if (root.type == JVal::OBJECT) list = root.find("probes");
    if (!list || list->type != JVal::ARRAY) { err = path + ": top level must be an array or {\"probes\":[...]}"; return false; }

    for (const auto & e : list->arr) {
        if (e.type != JVal::OBJECT) { err = path + ": probe entry is not an object"; return false; }
        Probe pr;
        pr.id       = jstr(e, "id");
        pr.context  = jstr(e, "context");
        pr.question = jstr(e, "question");
        pr.prior    = jstr(e, "prior");
        pr.target   = jstr(e, "target");
        pr.prior_aliases  = jstrs(e, "prior_aliases");
        pr.target_aliases = jstrs(e, "target_aliases");
        if (pr.id.empty() || pr.context.empty() || pr.question.empty()) {
            err = path + ": probe needs id, context, question";
            return false;
        }
        if (pr.prior_aliases.empty()) pr.prior_aliases.push_back(pr.prior);
        if (pr.target_aliases.empty()) pr.target_aliases.push_back(pr.target);
        probes.push_back(std::move(pr));
    }
    if (probes.empty()) { err = path + ": no probes"; return false; }
    return true;
}

// ---------------------------------------------------------------- model side

static std::string format_chat(Engine & eng, const std::string & sys, const std::string & user) {
    llama_chat_message msgs[2] = {
        { "system", sys.c_str() },
        { "user",   user.c_str() },
    };
    std::vector<char> buf(1 << 16);
    int n = llama_chat_apply_template(eng.model_chat_template(), msgs, 2, true,
                                      buf.data(), (int)buf.size());
    if (n > (int)buf.size()) {
        buf.resize((size_t)n);
        n = llama_chat_apply_template(eng.model_chat_template(), msgs, 2, true,
                                      buf.data(), (int)buf.size());
    }
    if (n > 0) return std::string(buf.data(), std::min((size_t)n, buf.size()));

    // Template unavailable (or not recognised): plain two-message fallback.
    return sys + "\n\nUser: " + user + "\nAssistant:";
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), nullptr, 0, false, true);
    if (n < 0) n = -n;
    std::vector<llama_token> toks((size_t)n);
    if (n == 0) return toks;
    const int m = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), n, false, true);
    toks.resize(m > 0 ? (size_t)m : 0);
    return toks;
}

static std::string token_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, false);
    if (n < 0) {
        std::vector<char> big((size_t)(-n));
        const int m = llama_token_to_piece(vocab, tok, big.data(), (int32_t)big.size(), 0, false);
        return m > 0 ? std::string(big.data(), (size_t)m) : std::string();
    }
    return n > 0 ? std::string(buf, (size_t)n) : std::string();
}

static void combine(const float * ctx, const float * prior, int n,
                    float alpha, float beta, std::vector<float> & out) {
    out.resize((size_t)n);
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) mx = std::max(mx, ctx[i]);
    const float thr = beta > 0.0f ? mx + std::log(beta)
                                  : -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (ctx[i] >= thr) out[(size_t)i] = (1.0f + alpha) * ctx[i] - alpha * prior[i];
        else               out[(size_t)i] = -std::numeric_limits<float>::infinity();
    }
}

static int pick(const std::vector<float> & v, float temp, std::mt19937 & rng) {
    if (temp <= 0.0f) {
        int best = 0;
        float bv = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i] > bv) { bv = v[i]; best = (int)i; }
        return best;
    }

    float mx = -std::numeric_limits<float>::infinity();
    for (float x : v) if (std::isfinite(x) && x > mx) mx = x;
    if (!std::isfinite(mx)) return 0;

    std::vector<double> p(v.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) continue;
        p[i] = std::exp(((double)v[i] - (double)mx) / (double)temp);
        sum += p[i];
    }
    if (!(sum > 0.0)) return 0;

    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double r = U(rng) * sum;
    double acc = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        acc += p[i];
        if (r <= acc) return (int)i;
    }
    return (int)p.size() - 1;
}

enum class Mode { Prior, Base, Cad };

struct RunOut {
    std::string text;
    int         n_tokens  = 0;
    std::string finish    = "max";
    double      prefill_ms = 0.0;
    double      decode_ms  = 0.0;
    std::vector<double> step_ms;
    std::string err;
};

static bool generate(Engine & eng, Mode mode,
                     const std::vector<llama_token> & ctx_toks,
                     const std::vector<llama_token> & qry_toks,
                     float alpha, float beta, int max_tokens, float temp,
                     uint32_t seed, RunOut & out) {
    const llama_vocab * vocab = eng.vocab();
    const int nv = eng.n_vocab();
    std::vector<float> lc, lp, mod;

    const auto t_pre = Clock::now();
    if (mode != Mode::Prior) {
        if (!eng.prefill(0, ctx_toks, true)) { out.err = "prefill ctx failed"; return false; }
        const float * p = llama_get_logits_ith(eng.ctx(), -1);
        if (!p) { out.err = "no ctx logits"; return false; }
        lc.assign(p, p + nv);
    }
    if (mode != Mode::Base) {
        const llama_seq_id seq = mode == Mode::Prior ? 0 : 1;
        if (!eng.prefill(seq, qry_toks, true)) { out.err = "prefill query failed"; return false; }
        const float * p = llama_get_logits_ith(eng.ctx(), -1);
        if (!p) { out.err = "no query logits"; return false; }
        lp.assign(p, p + nv);
    }
    out.prefill_ms = ms_since(t_pre);

    int pos_ctx = (int)ctx_toks.size();
    int pos_qry = (int)qry_toks.size();
    std::mt19937 rng(seed);

    auto sample_now = [&]() -> int {
        if (mode == Mode::Prior) return pick(lp, temp, rng);
        if (mode == Mode::Base)  return pick(lc, temp, rng);
        combine(lc.data(), lp.data(), nv, alpha, beta, mod);
        return pick(mod, temp, rng);
    };

    int tok = sample_now();
    for (;;) {
        if (llama_vocab_is_eog(vocab, tok)) { out.finish = "eos"; break; }
        out.text += token_piece(vocab, tok);
        ++out.n_tokens;
        if (out.n_tokens >= max_tokens) { out.finish = "max"; break; }

        std::vector<Request> reqs;
        if (mode != Mode::Prior)
            reqs.push_back({ (llama_seq_id)0, (llama_pos)pos_ctx++, tok, true });
        if (mode != Mode::Base)
            reqs.push_back({ (llama_seq_id)(mode == Mode::Prior ? 0 : 1),
                             (llama_pos)pos_qry++, tok, true });

        const auto t0 = Clock::now();
        if (!eng.decode_requests(reqs)) { out.err = "decode failed"; return false; }
        eng.sync();
        out.step_ms.push_back(ms_since(t0));

        const float * c = nullptr;
        const float * q = nullptr;
        if (mode == Mode::Cad) {
            c = llama_get_logits_ith(eng.ctx(), 0);
            q = llama_get_logits_ith(eng.ctx(), 1);
        } else if (mode == Mode::Base) {
            c = llama_get_logits_ith(eng.ctx(), 0);
        } else {
            q = llama_get_logits_ith(eng.ctx(), 0);
        }
        if ((mode != Mode::Prior && !c) || (mode != Mode::Base && !q)) {
            out.err = "missing step logits";
            return false;
        }
        if (c) lc.assign(c, c + nv);
        if (q) lp.assign(q, q + nv);
        tok = sample_now();
    }

    for (double ms : out.step_ms) out.decode_ms += ms;
    return true;
}

// ---------------------------------------------------------------- CLI

struct ArmSpec {
    char  id    = 'A';
    float alpha = 0.0f;
    float beta  = 0.0f;
    Mode  mode  = Mode::Base;
};

static bool arm_spec(char c, ArmSpec & a) {
    a.id = c;
    switch (c) {
        case 'P': a.alpha = 0.0f; a.beta = 0.0f; a.mode = Mode::Prior; return true;
        case 'A': a.alpha = 0.0f; a.beta = 0.0f; a.mode = Mode::Base;  return true;
        case 'B': a.alpha = 0.5f; a.beta = 0.0f; a.mode = Mode::Cad;   return true;
        case 'C': a.alpha = 0.5f; a.beta = 0.1f; a.mode = Mode::Cad;   return true;
        case 'D': a.alpha = 1.0f; a.beta = 0.1f; a.mode = Mode::Cad;   return true;
        default:  return false;
    }
}

static bool parse_seeds(const std::string & s, std::vector<uint32_t> & out) {
    out.clear();
    size_t start = 0;
    while (start <= s.size()) {
        const size_t c = s.find(',', start);
        const std::string tok = s.substr(start, c == std::string::npos ? std::string::npos : c - start);
        if (tok.empty()) return false;
        char * end = nullptr;
        const unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || tok[0] == '-') return false;
        out.push_back((uint32_t)v);
        if (c == std::string::npos) break;
        start = c + 1;
    }
    return !out.empty();
}

static std::string run_json(const ArmSpec & arm, const std::string & probe, uint32_t seed,
                            float temp, const RunOut & r) {
    std::string js = "{";
    js += "\"arm\":\"" + std::string(1, arm.id) + "\"";
    js += ",\"probe\":\"" + esc(probe) + "\"";
    js += ",\"seed\":" + std::to_string(seed);
    js += ",\"temp\":" + fmt((double)temp, "%.6g");
    js += ",\"alpha\":" + fmt((double)arm.alpha, "%.3f");
    js += ",\"beta\":" + fmt((double)arm.beta, "%.3f");
    js += ",\"text\":\"" + esc(r.text) + "\"";
    js += ",\"n_tokens\":" + std::to_string(r.n_tokens);
    js += ",\"finish\":\"" + esc(r.finish) + "\"";
    js += ",\"prefill_ms\":" + fmt(r.prefill_ms, "%.4f");
    js += ",\"decode_ms\":" + fmt(r.decode_ms, "%.4f");
    js += ",\"step_ms\":[";
    for (size_t i = 0; i < r.step_ms.size(); ++i) {
        if (i) js += ",";
        js += fmt(r.step_ms[i], "%.4f");
    }
    js += "]}";
    return js;
}

static const char * SYS_MSG = "You are a helpful assistant. Answer the question.";

int main(int argc, char ** argv) {
    std::string model, probes_path, out, arm_arg = "all", probe_filter;
    std::string seeds_arg;
    int max_tokens = 40;
    float temp = 0.0f;
    uint32_t seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--probes") probes_path = nx();
        else if (k == "--out") out = nx();
        else if (k == "--arm") arm_arg = nx();
        else if (k == "--probe") probe_filter = nx();
        else if (k == "--max-tokens") max_tokens = std::atoi(nx().c_str());
        else if (k == "--temp") temp = std::strtof(nx().c_str(), nullptr);
        else if (k == "--seed") seed = (uint32_t)std::strtoul(nx().c_str(), nullptr, 10);
        else if (k == "--seeds") seeds_arg = nx();
        else { std::fprintf(stderr, "cad_cli: unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (model.empty() || probes_path.empty() || out.empty()) {
        std::fprintf(stderr, "cad_cli: --model --probes --out are required\n");
        return 2;
    }

    for (char & c : arm_arg) c = (char)std::toupper((unsigned char)c);
    std::vector<ArmSpec> arms;
    if (arm_arg == "ALL") {
        for (char c : std::string("PABCD")) {
            ArmSpec a;
            if (!arm_spec(c, a)) return 2;
            arms.push_back(a);
        }
    } else {
        if (arm_arg.size() != 1) { std::fprintf(stderr, "cad_cli: --arm P|A|B|C|D|all\n"); return 2; }
        ArmSpec a;
        if (!arm_spec(arm_arg[0], a)) { std::fprintf(stderr, "cad_cli: --arm P|A|B|C|D|all\n"); return 2; }
        arms.push_back(a);
    }

    std::vector<uint32_t> seeds;
    if (!seeds_arg.empty()) {
        if (!parse_seeds(seeds_arg, seeds)) {
            std::fprintf(stderr, "cad_cli: --seeds must be a comma-separated integer list\n");
            return 2;
        }
    } else {
        seeds.push_back(seed);
    }

    if (max_tokens < 1) max_tokens = 1;

    std::vector<Probe> probes;
    {
        std::string err;
        if (!load_probes(probes_path, probes, err)) {
            std::fprintf(stderr, "cad_cli: %s\n", err.c_str());
            return 1;
        }
    }
    if (!probe_filter.empty()) {
        std::vector<Probe> keep;
        for (auto & p : probes) if (p.id == probe_filter) keep.push_back(p);
        if (keep.empty()) { std::fprintf(stderr, "cad_cli: probe %s not found\n", probe_filter.c_str()); return 2; }
        probes = std::move(keep);
    }

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine::Config cfg;
    cfg.model     = model;
    cfg.n_ctx     = 8192;
    cfg.n_seq_max = 4;
    cfg.ngl       = 99;
    cfg.max_batch = 2048;
    cfg.kv_unified = true;
    cfg.fa        = true;

    Engine eng;
    if (!eng.init(cfg)) {
        std::fprintf(stderr, "cad_cli: model/context init failed\n");
        return 1;
    }

    std::string runs = "[";
    size_t n_runs = 0;
    for (const auto & pr : probes) {
        const std::string ctx_user = "Context:\n" + pr.context + "\n\nQuestion: " + pr.question;
        const std::string qry_user = "Question: " + pr.question;
        const std::vector<llama_token> ctx_toks = tokenize(eng.vocab(), format_chat(eng, SYS_MSG, ctx_user));
        const std::vector<llama_token> qry_toks = tokenize(eng.vocab(), format_chat(eng, SYS_MSG, qry_user));

        for (const auto & arm : arms) {
            for (uint32_t sd : seeds) {
                eng.clear();
                RunOut r;
                if (!generate(eng, arm.mode, ctx_toks, qry_toks, arm.alpha, arm.beta,
                              max_tokens, temp, sd, r)) {
                    std::fprintf(stderr, "cad_cli: arm %c probe %s seed %u: %s\n",
                                 arm.id, pr.id.c_str(), sd, r.err.c_str());
                    eng.shutdown();
                    return 1;
                }
                if (n_runs++) runs += ",";
                runs += run_json(arm, pr.id, sd, temp, r);
                std::fprintf(stderr, "[cad] arm=%c probe=%s seed=%u n=%d finish=%s prefill=%.1fms decode=%.1fms\n",
                             arm.id, pr.id.c_str(), sd, r.n_tokens, r.finish.c_str(),
                             r.prefill_ms, r.decode_ms);
            }
        }
    }
    runs += "]";

    std::string js = "{\"ok\":true";
    js += ",\"model\":\"" + esc(model) + "\"";
    js += ",\"probes\":\"" + esc(probes_path) + "\"";
    js += ",\"max_tokens\":" + std::to_string(max_tokens);
    js += ",\"temp\":" + fmt((double)temp, "%.6g");
    js += ",\"runs\":" + runs + "}";

    eng.shutdown();

    FILE * f = std::fopen(out.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cad_cli: cannot write %s\n", out.c_str()); return 1; }
    std::fwrite(js.data(), 1, js.size(), f);
    std::fclose(f);
    std::printf("%s\n", js.c_str());
    return 0;
}
