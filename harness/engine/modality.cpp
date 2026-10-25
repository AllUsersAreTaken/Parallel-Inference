// RecycleLM modality probe 2: clean cross-substrate generator.
//
// N parallel streams, one mode per stream (cot|cotw|fwd|bwd|table|subq|restate|
// units|check|code). Every stream gets its own chat prompt (system + user), is
// prefilled on its own sequence id, seeded from the argmax of its own last
// logits, and then all live streams decode together in ONE Engine::tick per step.
//
//   modality.exe --model M --data D --out F --limit N \
//                --modes cot,cot,...,code,... --max-trace 1024 --seed 1234
//   modality.exe ... --seeds 1234,1335,...   # one RNG seed per stream
//
//   modality.exe --model M --data D --out F --limit N --modes cot,... \
//                --exemplars FILE   # one escaped exemplar per stream
//   modality.exe ... --wrapper     # format-match every stream, no exemplar
//
// The "cotw" mode uses the same system prompt as "cot" but wraps its user
// message as "Question: " + <question> + "\nSolution:" (the s4 --wrapper text
// with no exemplar), so the format lever can be tested inside one batch.
//
// With --seeds the list must have exactly one integer per stream; the default
// (absent flag) keeps the historical per-stream series seed + 101*j.
//
// --swap-parity swaps the first half and second half of the --modes list on
// problems with an ODD id before building prompts, so a mode's position in the
// batch alternates across problems and position effects cancel. Seeds stay
// keyed by stream index.
//
// With --exemplars (or --wrapper) the user message for stream j is
//   <exemplar_j> + "\n\n" + "Question: " + <question> + "\nSolution:"
// and an empty exemplar line yields just the wrapper. The exemplar file uses
// the same escaping as the TSV text column (\n, \t, \\).
//
//   modality.exe --selftest      # no model load; checks stop + unescape rules
//
// Output TSV (tab separated, one row per stream):
//   id  stream  mode  finish  generated  ttft_ms  done_ms  text
// text is escaped: \ -> \\, newline -> \n, tab -> \t, CR -> \r.
// A sidecar <out>.run.json records per-problem timings, the step_ms trace, and
// the SHA-256 of the output TSV.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::string SYS_COT =
    "You are a careful math solver. Think step by step and show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number>";
static std::string SYS_FWD =
    "You are a careful math solver. Work forward from the quantities given in the "
    "problem: list the knowns, then compute the intermediate results in order until "
    "you reach the asked quantity. Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_BWD =
    "You are a careful math solver. Work backward from the quantity the problem "
    "asks for: first write the equation that produces it, then fill in each input, "
    "tracing every input back to the given quantities. Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_TABLE =
    "You are a careful math solver. First build a table of every quantity in the "
    "problem, with its unit and meaning. Then compute the answer from the table. "
    "Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_SUBQ =
    "You are a careful math solver. Break the problem into the smallest "
    "sub-questions, answer each sub-question explicitly in order, then combine the "
    "sub-answers into the final result. Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_RESTATE =
    "You are a careful math solver. First restate the problem in your own words and "
    "state exactly what is asked. Then solve it step by step. Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_UNITS =
    "You are a careful math solver. Track the units and the magnitude of every "
    "quantity at each step, and sanity-check the size of the final result. Show your "
    "reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_CHECK =
    "You are a careful math solver. Solve the problem step by step, then verify by "
    "substituting your answer back into the original conditions; if the check fails, "
    "correct your answer. Show your reasoning. "
    "End with a final line exactly of the form: The answer is: <number> "
    "The final answer may be boxed.";
static std::string SYS_CODE =
    "You are a Python programmer. Write one short Python program that computes and prints "
    "the answer to the problem. Use only the Python standard library. Do not read input, "
    "files, or the network. The last statement must be print(<answer>). Reply with exactly "
    "one ```python fenced code block and nothing else.";

// Every named mode other than "code" is a CoT-family strategy: it samples text,
// stops on the balanced-brace/answer-line rule, and is scored by extract_cot.
static bool is_cot_mode(const std::string & m) { return m != "code"; }

static const std::string * sys_prompt_for(const std::string & mode) {
    if (mode == "cot")     return &SYS_COT;
    if (mode == "cotw")    return &SYS_COT;
    if (mode == "fwd")     return &SYS_FWD;
    if (mode == "bwd")     return &SYS_BWD;
    if (mode == "table")   return &SYS_TABLE;
    if (mode == "subq")    return &SYS_SUBQ;
    if (mode == "restate") return &SYS_RESTATE;
    if (mode == "units")   return &SYS_UNITS;
    if (mode == "check")   return &SYS_CHECK;
    if (mode == "code")    return &SYS_CODE;
    return nullptr;
}

static double ms_since(const Clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::string lower_copy(const std::string & s) {
    std::string o = s;
    for (char & c : o) c = (char)std::tolower((unsigned char)c);
    return o;
}

// End of the think block, if the model emitted one.
static size_t think_block_end(const std::string & text) {
    size_t best = std::string::npos;
    const std::string deepseek_end =
        "<" "\xEF\xBD\x9C" "end" "\xE2\x96\x81" "of" "\xE2\x96\x81" "thinking"
        "\xEF\xBD\x9C" ">";
    const std::string markers[] = {
        std::string("</") + "think>",
        deepseek_end,
    };
    for (const auto & m : markers) {
        size_t p = text.rfind(m);
        if (p != std::string::npos) best = std::max(best, p + m.size());
    }
    return best;
}

// Result of scanning the first \boxed{...} after the think block (if any).
struct BoxScan {
    bool present = false;   // an opening "\boxed{" was seen
    bool complete = false;  // its brace depth returned to zero
};

// Brace-depth scan: \boxed{...} is complete only when nested {} balance out.
// This lets \boxed{\frac{104}{4}} and \boxed{160 \text{ miles }} close on the
// outer brace instead of the first '}'.
static BoxScan scan_box(const std::string & text) {
    BoxScan info;
    size_t start = think_block_end(text);
    if (start == std::string::npos) start = 0;
    size_t b = text.find("\\boxed{", start);
    if (b == std::string::npos) return info;
    info.present = true;
    int depth = 1;
    for (size_t i = b + 7; i < text.size(); ++i) {
        if (text[i] == '{') {
            ++depth;
        } else if (text[i] == '}') {
            if (--depth == 0) { info.complete = true; break; }
        }
    }
    return info;
}

// True only when a \boxed{...} is present AND balanced.
static bool cot_answer_complete(const std::string & text) {
    return scan_box(text).complete;
}

// "The answer is:" followed by a number and a real terminator. A digit run is
// complete only when it ends at newline, CR, '.', or ',' (after optional
// trailing spaces); internal ',' or '.' is accepted only when another digit
// follows (1,234 / 3.14). An unterminated digit run at end-of-text is still
// being generated, so generation continues until EOG or a completed \boxed{}.
static bool cot_has_answer_line(const std::string & text) {
    const std::string low = lower_copy(text);
    const std::string key = "the answer is:";
    size_t p = low.rfind(key);
    if (p == std::string::npos) return false;
    size_t i = p + key.size();
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    if (i < text.size() && text[i] == '$') ++i;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    if (i < text.size() && (text[i] == '-' || text[i] == '+')) ++i;
    if (i >= text.size() || !std::isdigit((unsigned char)text[i])) return false;
    while (i < text.size()) {
        if (std::isdigit((unsigned char)text[i])) { ++i; continue; }
        if ((text[i] == ',' || text[i] == '.') &&
            i + 1 < text.size() && std::isdigit((unsigned char)text[i + 1])) {
            ++i;
            continue;
        }
        break;
    }
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    if (i >= text.size()) return false;                      // unterminated run
    return text[i] == '\n' || text[i] == '\r' || text[i] == '.' || text[i] == ',';
}

// Stop on a complete box. Keep generating while a box is open so a truncated
// box is never labelled "answer". With no box, stop on the requested final
// answer line.
static bool cot_stop(const std::string & text) {
    BoxScan box = scan_box(text);
    if (box.complete) return true;
    if (box.present) return false;
    return cot_has_answer_line(text);
}

// Opening ``` fence with a matching closing ```.
static bool code_stop(const std::string & text) {
    size_t o = text.find("```");
    if (o == std::string::npos) return false;
    return text.find("```", o + 3) != std::string::npos;
}

// Inverse of the file escaping used for exemplar lines: \\ -> \, \n -> newline,
// \t -> tab, \r -> CR. Unknown escapes keep both characters.
static std::string unescape_escaped(const std::string & s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            o += s[i];
            continue;
        }
        const char c = s[i + 1];
        if      (c == '\\') { o += '\\'; ++i; }
        else if (c == 'n')  { o += '\n'; ++i; }
        else if (c == 't')  { o += '\t'; ++i; }
        else if (c == 'r')  { o += '\r'; ++i; }
        else                { o += s[i]; }
    }
    return o;
}

// One escaped exemplar per line. Blank lines are preserved; a trailing newline
// at end of file does not add a line.
static bool load_escaped_lines(const std::string & path, std::vector<std::string> & out) {
    out.clear();
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return false;
    std::string line;
    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(unescape_escaped(line));
    }
    return true;
}

// True when a stream's user message carries the "Question: ...\nSolution:"
// wrapper: the explicit cotw mode always does, and --wrapper / --exemplars
// format-match every stream.
static bool mode_wraps(const std::string & mode, bool use_wrapper) {
    return use_wrapper || mode == "cotw";
}

// Swap the first and second halves of a list; a no-op for odd-length lists.
static std::vector<std::string> swap_halves(const std::vector<std::string> & v) {
    std::vector<std::string> o = v;
    if (o.size() >= 2 && o.size() % 2 == 0) {
        const size_t h = o.size() / 2;
        for (size_t i = 0; i < h; ++i) std::swap(o[i], o[i + h]);
    }
    return o;
}

// --swap-parity alternates the mode halves across problems while seeds stay
// keyed by stream index. To keep mode from being confounded with seed, stream
// j and its swapped partner j + N/2 must share one seed. N must be even.
static bool seeds_have_parity(const std::vector<uint32_t> & seeds) {
    const size_t n = seeds.size();
    if (n < 2 || n % 2 != 0) return false;
    for (size_t j = 0; j < n / 2; ++j) {
        if (seeds[j] != seeds[j + n / 2]) return false;
    }
    return true;
}

// ---------------------------------------------------------------- SHA-256 ----
// Sidecar provenance: a streaming SHA-256 of the output TSV, so a later
// analysis can prove it read the bytes the run actually produced.

struct Sha256Ctx {
    uint32_t h[8];
    uint64_t len;
    unsigned char buf[64];
    size_t buflen;
};

static uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_init(Sha256Ctx & c) {
    static const uint32_t iv[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    for (int i = 0; i < 8; ++i) c.h[i] = iv[i];
    c.len = 0;
    c.buflen = 0;
}

static void sha256_transform(Sha256Ctx & c, const unsigned char * block) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3];
    uint32_t e = c.h[4], f = c.h[5], g = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & cc) ^ (b & cc);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d;
    c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += h;
}

static void sha256_update(Sha256Ctx & c, const unsigned char * data, size_t n) {
    c.len += (uint64_t)n * 8u;
    while (n > 0) {
        const size_t take = std::min(n, (size_t)64 - c.buflen);
        std::memcpy(c.buf + c.buflen, data, take);
        c.buflen += take;
        data += take;
        n -= take;
        if (c.buflen == 64) {
            sha256_transform(c, c.buf);
            c.buflen = 0;
        }
    }
}

static std::string sha256_final_hex(Sha256Ctx & c) {
    const uint64_t bits = c.len;
    unsigned char pad[72];
    const size_t padlen = (c.buflen < 56) ? (56 - c.buflen) : (120 - c.buflen);
    pad[0] = 0x80;
    for (size_t i = 1; i < padlen; ++i) pad[i] = 0;
    for (int i = 0; i < 8; ++i) {
        pad[padlen + i] = (unsigned char)(bits >> (56 - 8 * i));
    }
    sha256_update(c, pad, padlen + 8);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            out += hex[(c.h[i] >> shift) & 0xfu];
        }
    }
    return out;
}

static std::string sha256_hex(const void * data, size_t n) {
    Sha256Ctx c;
    sha256_init(c);
    sha256_update(c, (const unsigned char *)data, n);
    return sha256_final_hex(c);
}

static std::string sha256_file_hex(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    Sha256Ctx c;
    sha256_init(c);
    char buf[1 << 16];
    while (in) {
        in.read(buf, sizeof(buf));
        const std::streamsize got = in.gcount();
        if (got > 0) sha256_update(c, (const unsigned char *)buf, (size_t)got);
    }
    return sha256_final_hex(c);
}

// CoT stop-rule self test. No model, no GPU: pure string logic.
static int run_selftest() {
    const struct { const char * text; bool want; } box_cases[] = {
        {"\\boxed{55}", true},
        {"\\boxed{\\frac{104}{4}}", true},
        {"\\boxed{160 \\text{ miles westward}}", true},
        {"\\boxed{55", false},
        {"\\boxed{\\frac{104}{4}", false},
        {"no box here", false},
        {"\\boxed{}", true},
        {"\\boxed{a{b}c}", true},
    };
    const struct { const char * text; bool want; } stop_cases[] = {
        {"\\boxed{55}", true},
        {"\\boxed{\\frac{104}{4}}", true},
        {"\\boxed{55 incomplete", false},
        {"The answer is: 42", false},
        {"The answer is: 1", false},
        {"The answer is: 42\n", true},
        {"The answer is: 1234\nNext", true},
        {"The answer is: 1,234.\n", true},
        {"The answer is: 42, and that is final.", true},
        {"\\boxed{55 incomplete\nThe answer is: 42", false},
        {"just some text", false},
    };
    const struct { const char * in; const char * want; } unescape_cases[] = {
        {"Question: q\\nSolution: x", "Question: q\nSolution: x"},
        {"a\\tb", "a\tb"},
        {"back\\\\slash", "back\\slash"},
        {"cr\\rlf\\n", "cr\rlf\n"},
        {"no escapes", "no escapes"},
        {"", ""},
    };
    int fails = 0;
    for (const auto & c : box_cases) {
        const bool got = cot_answer_complete(c.text);
        if (got != c.want) {
            std::fprintf(stderr, "selftest FAIL cot_answer_complete(\"%s\") = %d, want %d\n",
                         c.text, (int)got, (int)c.want);
            ++fails;
        }
    }
    for (const auto & c : stop_cases) {
        const bool got = cot_stop(c.text);
        if (got != c.want) {
            std::fprintf(stderr, "selftest FAIL cot_stop(\"%s\") = %d, want %d\n",
                         c.text, (int)got, (int)c.want);
            ++fails;
        }
    }
    for (const auto & c : unescape_cases) {
        const std::string got = unescape_escaped(c.in);
        if (got != c.want) {
            std::fprintf(stderr, "selftest FAIL unescape_escaped(\"%s\") = \"%s\", want \"%s\"\n",
                         c.in, got.c_str(), c.want);
            ++fails;
        }
    }
    {
        const std::vector<std::string> in = {"a", "b", "c", "d"};
        const std::vector<std::string> want = {"c", "d", "a", "b"};
        const std::vector<std::string> got = swap_halves(in);
        if (got != want) { std::fprintf(stderr, "selftest FAIL swap_halves\n"); ++fails; }
    }
    {
        const std::vector<std::string> in = {"a", "b", "c"};
        const std::vector<std::string> want = {"a", "b", "c"};
        const std::vector<std::string> got = swap_halves(in);
        if (got != want) { std::fprintf(stderr, "selftest FAIL swap_halves(odd)\n"); ++fails; }
    }
    {
        const std::vector<std::string> in = {"cot", "cotw"};
        const std::vector<std::string> want = {"cotw", "cot"};
        const std::vector<std::string> got = swap_halves(in);
        if (got != want) { std::fprintf(stderr, "selftest FAIL swap_halves(cot/cotw)\n"); ++fails; }
    }
    {
        const struct { std::vector<uint32_t> seeds; bool want; } seed_cases[] = {
            {{10, 20, 10, 20}, true},
            {{10, 20, 20, 10}, false},
            {{10, 20, 30, 40}, false},
            {{10, 10}, true},
            {{10, 20, 10}, false},
        };
        for (const auto & c : seed_cases) {
            const bool got = seeds_have_parity(c.seeds);
            if (got != c.want) {
                std::fprintf(stderr, "selftest FAIL seeds_have_parity(%zu streams) = %d, want %d\n",
                             c.seeds.size(), (int)got, (int)c.want);
                ++fails;
            }
        }
    }
    {
        const std::string got = sha256_hex("abc", 3);
        const std::string want = "ba7816bf8f01cfea414140de5dae2223"
                                 "b00361a396177a9cb410ff61f20015ad";
        if (got != want) {
            std::fprintf(stderr, "selftest FAIL sha256_hex(\"abc\") = %s\n", got.c_str());
            ++fails;
        }
    }
    if (fails == 0) {
        std::printf("selftest: OK (%zu box cases, %zu stop cases, %zu unescape cases)\n",
                    sizeof(box_cases) / sizeof(box_cases[0]),
                    sizeof(stop_cases) / sizeof(stop_cases[0]),
                    sizeof(unescape_cases) / sizeof(unescape_cases[0]));
        return 0;
    }
    std::printf("selftest: FAILED (%d)\n", fails);
    return 1;
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

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"':  o += "\\\""; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", (unsigned char)c); o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

static llama_token argmax_token(const float * lg, int nv) {
    int b = 0; float bv = lg[0];
    for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; }
    return (llama_token)b;
}

struct Gen : Process {
    Engine & e;
    int M, max_trace;
    std::vector<llama_seq_id> seqs;
    std::vector<llama_pos>    pos;
    std::vector<llama_token>  next;
    std::vector<std::string>  mode;
    std::vector<std::string>  text;
    std::vector<bool>         fin;
    std::vector<std::string>  finish;
    std::vector<int>          gen;
    std::vector<double>       ttft;
    std::vector<double>       done_ms;
    std::vector<llama_sampler *> smpls;
    std::vector<int> active_order;
    Clock::time_point t0;
    int base_ = 0;

    Gen(Engine & en, const std::vector<std::string> & modes, int mt,
        const std::vector<uint32_t> & seeds, Clock::time_point t0_)
        : e(en), M((int)modes.size()), max_trace(mt), mode(modes), t0(t0_) {
        min_slots = 0;
        for (int j = 0; j < M; ++j) {
            seqs.push_back(j);
            pos.push_back(0);
            next.push_back(0);
            text.push_back("");
            fin.push_back(false);
            finish.push_back("");
            gen.push_back(0);
            ttft.push_back(-1.0);
            done_ms.push_back(-1.0);
            auto sp = llama_sampler_chain_default_params();
            auto * c = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(seeds[j]));
            smpls.push_back(c);
        }
    }

    ~Gen() override {
        for (auto * s : smpls) llama_sampler_free(s);
    }

    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }

    void start(const std::vector<size_t> & prompt_len, const std::vector<llama_token> & seeds) {
        for (int j = 0; j < M; ++j) {
            pos[j] = (llama_pos)prompt_len[j];
            next[j] = seeds[j];
        }
    }

    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        active_order.clear();
        for (int j = 0; j < M; ++j) {
            if (fin[j]) continue;
            if ((int)out.size() - base >= budget) break;
            out.push_back({ seqs[j], pos[j], next[j], true });
            active_order.push_back(j);
        }
    }

    void on_logits(int idx, const float * lg) override {
        int active_idx = idx - base_;
        if (active_idx < 0 || active_idx >= (int)active_order.size()) return;
        {
            int j = active_order[active_idx];
            if (j < 0 || j >= M || fin[j]) return;

            const llama_token tok = next[j];
            if (gen[j] == 0) ttft[j] = ms_since(t0);
            ++gen[j];
            const bool eog = llama_vocab_is_eog(e.vocab(), tok);
            if (!eog) {
                char buf[256];
                int np = llama_token_to_piece(e.vocab(), tok, buf, (int)sizeof(buf), 0, false);
                if (np > 0) text[j].append(buf, (size_t)np);
            }

            std::string reason;
            if (eog) reason = "eos";
            else if (is_cot_mode(mode[j]) && cot_stop(text[j])) reason = "answer";
            else if (mode[j] == "code" && code_stop(text[j]))   reason = "fence";
            else if (gen[j] >= max_trace) reason = "cap";

            if (!reason.empty()) {
                fin[j] = true;
                finish[j] = reason;
                done_ms[j] = ms_since(t0);
            } else {
                next[j] = llama_sampler_sample(smpls[j], e.ctx(), idx);
            }
            pos[j] += 1;
            (void)lg;
            return;
        }
    }

    bool done() const override {
        for (bool f : fin) if (!f) return false;
        return true;
    }
};

static bool parse_modes(const std::string & s, std::vector<std::string> & out) {
    out.clear();
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        std::string m = s.substr(p, q - p);
        if (sys_prompt_for(m) == nullptr) return false;
        out.push_back(m);
        if (q == s.size()) break;
        p = q + 1;
    }
    return !out.empty();
}

// Comma-separated non-negative 32-bit integers, one per stream.
static bool parse_seeds(const std::string & s, std::vector<uint32_t> & out) {
    out.clear();
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        const std::string v = s.substr(p, q - p);
        if (v.empty()) return false;
        for (char c : v) if (!std::isdigit((unsigned char)c)) return false;
        char * end = nullptr;
        const unsigned long long n = std::strtoull(v.c_str(), &end, 10);
        if (end == v.c_str() || *end != '\0' || n > 0xFFFFFFFFull) return false;
        out.push_back((uint32_t)n);
        if (q == s.size()) break;
        p = q + 1;
    }
    return !out.empty();
}

int main(int argc, char ** argv) {
    std::string model, data, out, modes_s, exemplars_path, seeds_s;
    int limit = 40, max_trace = 1024;
    uint32_t seed = 1234;
    bool selftest = false;
    bool wrapper = false;
    bool swap_parity = false;

    auto need_value = [&](int i) -> bool { return i + 1 < argc; };
    auto bad = [&](const std::string & msg) -> int {
        std::fprintf(stderr, "modality: %s\n", msg.c_str());
        return 2;
    };

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--selftest") {
            selftest = true;
        } else if (k == "--model") {
            if (!need_value(i)) return bad("--model needs a value");
            model = argv[++i];
        } else if (k == "--data") {
            if (!need_value(i)) return bad("--data needs a value");
            data = argv[++i];
        } else if (k == "--out") {
            if (!need_value(i)) return bad("--out needs a value");
            out = argv[++i];
        } else if (k == "--limit") {
            if (!need_value(i)) return bad("--limit needs a value");
            limit = std::atoi(argv[++i]);
        } else if (k == "--modes") {
            if (!need_value(i)) return bad("--modes needs a value");
            modes_s = argv[++i];
        } else if (k == "--max-trace") {
            if (!need_value(i)) return bad("--max-trace needs a value");
            max_trace = std::atoi(argv[++i]);
        } else if (k == "--seed") {
            if (!need_value(i)) return bad("--seed needs a value");
            seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        } else if (k == "--seeds") {
            if (!need_value(i)) return bad("--seeds needs a value");
            seeds_s = argv[++i];
        } else if (k == "--exemplars") {
            if (!need_value(i)) return bad("--exemplars needs a value");
            exemplars_path = argv[++i];
        } else if (k == "--wrapper") {
            wrapper = true;
        } else if (k == "--swap-parity") {
            swap_parity = true;
        } else {
            return bad("unknown argument: " + k);
        }
    }

    if (selftest) return run_selftest();

    if (model.empty()) return bad("--model is required");
    if (data.empty())  return bad("--data is required");
    if (out.empty())   return bad("--out is required");
    if (modes_s.empty()) return bad("--modes is required");
    if (limit < 1) return bad("--limit must be >= 1");
    if (max_trace < 1) return bad("--max-trace must be >= 1");

    std::vector<std::string> modes;
    if (!parse_modes(modes_s, modes)) {
        return bad("--modes must be a comma-separated list of "
                   "cot|cotw|fwd|bwd|table|subq|restate|units|check|code (no empty entries)");
    }
    const int N = (int)modes.size();
    const std::vector<std::string> modes_swapped = swap_halves(modes);

    std::vector<uint32_t> stream_seeds((size_t)N);
    if (!seeds_s.empty()) {
        if (!parse_seeds(seeds_s, stream_seeds)) {
            return bad("--seeds must be a comma-separated list of non-negative integers");
        }
        if ((int)stream_seeds.size() != N) {
            return bad("--seeds has " + std::to_string(stream_seeds.size()) +
                       " entries but there are " + std::to_string(N) + " streams");
        }
    } else {
        for (int j = 0; j < N; ++j) stream_seeds[j] = seed + 101u * (uint32_t)j;
    }

    // --swap-parity moves modes between halves, so paired halves must share
    // seeds or the swap confounds mode with seed.
    if (swap_parity && !seeds_have_parity(stream_seeds)) {
        return bad("--swap-parity requires an even stream count and paired seeds "
                   "(stream_seeds[j] == stream_seeds[j + N/2]); got " +
                   std::to_string(N) + " streams");
    }

    std::vector<std::string> exemplars;
    size_t max_exemplar_chars = 0;
    if (!exemplars_path.empty()) {
        if (!load_escaped_lines(exemplars_path, exemplars)) {
            return bad("cannot open --exemplars: " + exemplars_path);
        }
        if ((int)exemplars.size() != N) {
            return bad("--exemplars has " + std::to_string(exemplars.size()) +
                       " lines but there are " + std::to_string(N) + " streams");
        }
        for (const auto & e : exemplars) max_exemplar_chars = std::max(max_exemplar_chars, e.size());
    }
    // A wrapper is used for every stream when requested, and automatically when
    // exemplars are supplied, so the zero-shot arm is format-matched.
    const bool use_wrapper = wrapper || !exemplars_path.empty();

    // Read and validate the input before touching the GPU.
    std::vector<std::pair<int, std::string>> problems;
    size_t max_q_chars = 0;
    {
        std::ifstream fin(data);
        if (!fin) return bad("cannot open --data: " + data);
        std::string line;
        int lineno = 0;
        while (std::getline(fin, line) && (int)problems.size() < limit) {
            ++lineno;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                return bad("data line " + std::to_string(lineno) + " has no tab");
            }
            std::string q = line.substr(0, tab);
            problems.push_back({ lineno, q });
            max_q_chars = std::max(max_q_chars, q.size());
        }
    }
    if (problems.empty()) return bad("no problems read from --data");

    // Conservative bound: a BPE token covers at least one byte, plus room for
    // the chat template, the system message, and any exemplar.
    const int per_seq = (int)(max_q_chars + max_exemplar_chars) + 320 + 32 + max_trace + 16;
    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = N * per_seq + 512;
    cfg.n_seq_max = N + 2;
    cfg.max_batch = 1024;
    cfg.fa = true;
    cfg.kv_unified = true;

    llama_log_set([](ggml_log_level l, const char * t, void *) {
        if (l >= GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
    }, nullptr);

    Engine e;
    if (!e.init(cfg)) {
        std::fprintf(stderr, "modality: engine init failed (model=%s)\n", model.c_str());
        return 1;
    }

    auto tokenize = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 32);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) {
            t.resize((size_t)(-n));
            n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        }
        if (n < 0) { t.clear(); return t; }
        t.resize((size_t)n);
        return t;
    };

    FILE * f = std::fopen(out.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "modality: cannot open --out: %s\n", out.c_str());
        e.shutdown();
        return 1;
    }
    std::fprintf(f, "id\tstream\tmode\tfinish\tgenerated\tttft_ms\tdone_ms\ttext\n");

    struct ProbStat {
        int id;
        double prefill_ms, decode_ms, total_ms;
        int prompt_tokens_max;
        int prompt_tokens_sum;
    };
    std::vector<ProbStat> stats;
    std::vector<double> step_ms;
    size_t step_ms_total = 0;

    for (const auto & pr : problems) {
        const int pid = pr.first;
        const std::string & q = pr.second;

        // On odd ids the mode list is half-swapped so each mode's position in
        // the batch alternates across problems (seeds stay keyed by stream).
        const std::vector<std::string> & pm =
            (swap_parity && (pid % 2 == 1)) ? modes_swapped : modes;

        std::vector<std::vector<llama_token>> ptoks(N);
        for (int j = 0; j < N; ++j) {
            const std::string & sys = *sys_prompt_for(pm[j]);
            std::string user;
            if (mode_wraps(pm[j], use_wrapper)) user = "Question: " + q + "\nSolution:";
            else                                 user = q;
            if (!exemplars.empty() && !exemplars[j].empty()) {
                user = exemplars[j] + "\n\n" + user;
            }
            llama_chat_message msgs[2] = {
                { "system", sys.c_str() },
                { "user",   user.c_str() },
            };
            size_t cap = std::max<size_t>(4096, user.size() * 2 + sys.size() * 2 + 1024);
            std::vector<char> buf(cap);
            int plen = llama_chat_apply_template(e.model_chat_template(), msgs, 2, true,
                                                 buf.data(), (int)buf.size());
            if (plen < 0) {
                std::fprintf(stderr, "modality: chat template failed for id=%d stream=%d\n", pid, j);
                std::fclose(f); e.shutdown(); return 1;
            }
            if (plen > (int)buf.size()) {
                buf.resize((size_t)plen);
                plen = llama_chat_apply_template(e.model_chat_template(), msgs, 2, true,
                                                 buf.data(), (int)buf.size());
            }
            ptoks[j] = tokenize(std::string(buf.data(), (size_t)plen));
            if (ptoks[j].empty()) {
                std::fprintf(stderr, "modality: tokenization produced 0 tokens (id=%d stream=%d)\n", pid, j);
                std::fclose(f); e.shutdown(); return 1;
            }
        }

        int need = 0, ptmax = 0, ptsum = 0;
        for (int j = 0; j < N; ++j) {
            need += (int)ptoks[j].size() + max_trace;
            ptmax = std::max(ptmax, (int)ptoks[j].size());
            ptsum += (int)ptoks[j].size();
            if ((int)ptoks[j].size() + max_trace + 1 > cfg.n_ctx) {
                std::fprintf(stderr,
                    "modality: context too small for id=%d stream=%d (prompt=%zu max_trace=%d n_ctx=%d)\n",
                    pid, j, ptoks[j].size(), max_trace, cfg.n_ctx);
                std::fclose(f); e.shutdown(); return 1;
            }
        }
        if (need > cfg.n_ctx) {
            std::fprintf(stderr,
                "modality: unified KV too small for id=%d (need=%d n_ctx=%d)\n",
                pid, need, cfg.n_ctx);
            std::fclose(f); e.shutdown(); return 1;
        }

        const auto t0 = Clock::now();
        e.reset_processes();
        e.clear();

        std::vector<llama_token> seeds((size_t)N, 0);
        std::vector<size_t> plens((size_t)N, 0);
        for (int j = 0; j < N; ++j) {
            if (!e.prefill((llama_seq_id)j, ptoks[j], true)) {
                std::fprintf(stderr, "modality: prefill failed (id=%d stream=%d)\n", pid, j);
                std::fclose(f); e.shutdown(); return 1;
            }
            plens[j] = ptoks[j].size();
            seeds[j] = argmax_token(llama_get_logits_ith(e.ctx(), -1), e.n_vocab());
        }
        e.sync();
        const double prefill_ms = ms_since(t0);

        Gen g(e, pm, max_trace, stream_seeds, t0);
        g.start(plens, seeds);
        e.add(&g);

        const auto td0 = Clock::now();
        while (!g.done()) {
            int active = 0;
            double dt = e.tick(64, active);
            if (dt < 0) {
                std::fprintf(stderr, "modality: decode failed (id=%d)\n", pid);
                std::fclose(f); e.shutdown(); return 1;
            }
            if ((int)step_ms.size() < 5000) step_ms.push_back(dt);
            ++step_ms_total;
        }
        e.sync();
        const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - td0).count();
        const double total_ms = ms_since(t0);

        for (int j = 0; j < N; ++j) {
            std::string reason = g.finish[j].empty() ? "cap" : g.finish[j];
            std::fprintf(f, "%d\t%d\t%s\t%s\t%d\t%.3f\t%.3f\t%s\n",
                         pid, j, pm[j].c_str(), reason.c_str(), g.gen[j],
                         g.ttft[j] < 0 ? 0.0 : g.ttft[j],
                         g.done_ms[j] < 0 ? total_ms : g.done_ms[j],
                         escape_tsv(g.text[j]).c_str());
        }
        std::fflush(f);

        stats.push_back({ pid, prefill_ms, decode_ms, total_ms, ptmax, ptsum });
        std::printf("id=%d done\n", pid);
        std::fflush(stdout);

        e.reset_processes();
    }

    std::fclose(f);

    const std::string out_sha256 = sha256_file_hex(out);

    // Sidecar run metadata.
    {
        std::ofstream jf(out + ".run.json", std::ios::binary);
        jf << "{\n";
        jf << "  \"output\": \"" << json_escape(out) << "\",\n";
        jf << "  \"output_sha256\": \"" << out_sha256 << "\",\n";
        jf << "  \"model\": \"" << json_escape(model) << "\",\n";
        jf << "  \"data\": \"" << json_escape(data) << "\",\n";
        jf << "  \"n_streams\": " << N << ",\n";
        jf << "  \"modes\": [";
        for (int j = 0; j < N; ++j) jf << (j ? ", " : "") << "\"" << modes[j] << "\"";
        jf << "],\n";
    jf << "  \"max_trace\": " << max_trace << ",\n";
    jf << "  \"seed\": " << seed << ",\n";
    jf << "  \"seeds\": [";
    for (int j = 0; j < N; ++j) jf << (j ? ", " : "") << stream_seeds[j];
    jf << "],\n";
        jf << "  \"wrapper\": " << (use_wrapper ? "true" : "false") << ",\n";
        jf << "  \"swap_parity\": " << (swap_parity ? "true" : "false") << ",\n";
        jf << "  \"exemplars\": \"" << json_escape(exemplars_path) << "\",\n";
        jf << "  \"exemplar_streams\": " << (int)exemplars.size() << ",\n";
        jf << "  \"n_ctx\": " << cfg.n_ctx << ",\n";
        jf << "  \"problems\": [\n";
        for (size_t i = 0; i < stats.size(); ++i) {
            const auto & s = stats[i];
            jf << "    {\"id\": " << s.id
               << ", \"prefill_ms\": " << s.prefill_ms
               << ", \"decode_ms\": " << s.decode_ms
               << ", \"total_ms\": " << s.total_ms
               << ", \"prompt_tokens_max\": " << s.prompt_tokens_max
               << ", \"prompt_tokens_sum\": " << s.prompt_tokens_sum << "}"
               << (i + 1 == stats.size() ? "" : ",") << "\n";
        }
        jf << "  ],\n";
        auto arr = [&](const char * name, auto get, bool last) {
            jf << "  \"" << name << "\": [";
            for (size_t i = 0; i < stats.size(); ++i) jf << (i ? ", " : "") << get(stats[i]);
            jf << "]" << (last ? "" : ",") << "\n";
        };
        arr("prefill_ms", [](const ProbStat & s) { return s.prefill_ms; }, false);
        arr("decode_ms",  [](const ProbStat & s) { return s.decode_ms;  }, false);
        arr("total_ms",   [](const ProbStat & s) { return s.total_ms;   }, false);
        jf << "  \"step_ms_total\": " << step_ms_total << ",\n";
        jf << "  \"step_ms_recorded\": " << step_ms.size() << ",\n";
        jf << "  \"step_ms_fraction\": "
           << (step_ms_total ? (double)step_ms.size() / (double)step_ms_total : 1.0) << ",\n";
        jf << "  \"step_ms\": [";
        for (size_t i = 0; i < step_ms.size(); ++i) jf << (i ? ", " : "") << step_ms[i];
        jf << "]\n";
        jf << "}\n";
    }

    e.shutdown();
    return 0;
}
