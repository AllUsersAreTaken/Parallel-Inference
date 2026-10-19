// RecycleLM DesignStreams generator: N streams, each either a CoT solve or a
// Python-program solve. Dumps raw text; execution + voting happen in Python.
//
//   streams.exe --model M --data gsm8k.tsv --mode cot|code --N 8 --out file.tsv
// Output: id <TAB> stream <TAB> text

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::string SYS_COT =
    "You are a careful math solver. Think step by step, then end with a final line "
    "exactly of the form: The answer is: <number>";
static std::string SYS_JSON =
    "You are a precise scheduling assistant. Read the local message context and the request, "
    "then output a single JSON object describing the intended action. Use only facts present in "
    "the context. Do not invent details. If a required detail is absent, list it in "
    "missing_information and set needs_clarification to true.";
static std::string SYS_CODE =
    "You are a Python programmer. Write a short program that computes and prints the answer "
    "to the problem. Reply with only one ```python code block and nothing else.";

static const char * JSON_GRAMMAR = R"GBNF(
root ::= "{" sp "\"intended_event\"" sp ":" sp str "," sp "\"resolved_date_time\"" sp ":" sp str "," sp "\"requested_action\"" sp ":" sp str "," sp "\"supporting_message_ids\"" sp ":" sp arr "," sp "\"missing_information\"" sp ":" sp sarr "," sp "\"needs_clarification\"" sp ":" sp bool sp "}"
str ::= "\"" [^"]{0,120} "\""
sarr ::= "[" sp (str (sp "," sp str)*)? sp "]"
arr ::= "[" sp (int (sp "," sp int)*)? sp "]"
int ::= [0-9]{1,3}
bool ::= "true" | "false"
sp ::= [ \t\n]{0,3}
)GBNF";
struct Gen : Process {
    Engine & e; int M, max_trace;
    std::vector<llama_seq_id> seqs; std::vector<llama_pos> pos; std::vector<llama_token> next;
    std::vector<std::string> text; std::vector<bool> fin;
    std::vector<llama_sampler *> smpls;
    int base_ = 0; llama_pos p_start = 0;
    Gen(Engine & en, int m, int mt, uint32_t seed, const char * grammar) : e(en), M(m), max_trace(mt) {
        min_slots = 0;
        for (int j = 0; j < M; ++j) {
            seqs.push_back(j); pos.push_back(0); next.push_back(0); text.push_back(""); fin.push_back(false);
            auto sp = llama_sampler_chain_default_params(); auto * c = llama_sampler_chain_init(sp);
            if (grammar) llama_sampler_chain_add(c, llama_sampler_init_grammar(e.vocab(), grammar, "root"));
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(seed + 101u * (uint32_t)j));
            smpls.push_back(c);
        }
    }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }
    void start(llama_pos p0, llama_token seed) { for (int j = 1; j < M; ++j) e.seq_cp(0, j); p_start = p0; for (auto & p : pos) p = p0; for (auto & t : next) t = seed; }
    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        for (int j = 0; j < M; ++j) { if (fin[j]) continue; if ((int)out.size() - base >= budget) break; out.push_back({ seqs[j], pos[j], next[j], true }); }
    }
    void on_logits(int idx, const float * lg) override {
        int j = idx - base_; if (j < 0 || j >= M || fin[j]) return;
        char b[64]; int np = llama_token_to_piece(e.vocab(), next[j], b, sizeof(b), 0, false); if (np > 0) text[j].append(b, np);
        pos[j] += 1;
        if (llama_vocab_is_eog(e.vocab(), next[j]) || (int)(pos[j] - p_start) >= max_trace) fin[j] = true;
        else next[j] = llama_sampler_sample(smpls[j], e.ctx(), idx);
    }
    bool done() const override { for (bool x : fin) if (!x) return false; return true; }
};

int main(int argc, char ** argv) {
    std::string model, data, out, mode = "cot";
    int limit = 60, N = 8, max_trace = 512;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx(); else if (k == "--data") data = nx(); else if (k == "--out") out = nx();
        else if (k == "--mode") mode = nx(); else if (k == "--N") N = atoi(nx().c_str());
        else if (k == "--limit") limit = atoi(nx().c_str()); else if (k == "--max-trace") max_trace = atoi(nx().c_str());
    }
    Engine::Config cfg; cfg.model = model; cfg.n_ctx = 20 * (512 + max_trace + 16) + 256; cfg.n_seq_max = N + 2; cfg.max_batch = 1024;
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    Engine e; if (!e.init(cfg)) { fprintf(stderr, "init failed\n"); return 1; }

    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };

    std::ifstream fin(data); std::string line; int id = 0;
    FILE * f = fopen(out.c_str(), "w");
    while (std::getline(fin, line) && id < limit) {
        size_t tab = line.find('\t'); if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab); ++id;
        const std::string & sys = (mode == "code") ? SYS_CODE : (mode == "json") ? SYS_JSON : SYS_COT;
        llama_chat_message msgs[2] = { { "system", sys.c_str() }, { "user", q.c_str() } };
        const int nmsg = (mode == "cot") ? 1 : 2;
        const llama_chat_message * mp = (mode == "cot") ? &msgs[1] : msgs;
        std::vector<char> buf(std::max<size_t>(2048, q.size() * 2 + 512));
        int plen = llama_chat_apply_template(e.model_chat_template(), mp, nmsg, true, buf.data(), (int)buf.size());
        if (plen > (int)buf.size()) { buf.resize(plen); plen = llama_chat_apply_template(e.model_chat_template(), mp, nmsg, true, buf.data(), (int)buf.size()); }
        auto ptoks = tok(std::string(buf.data(), plen > 0 ? plen : 0));

        e.reset_processes(); e.clear();
        e.prefill(0, ptoks, true); e.sync();
        int nv = e.n_vocab(); const float * lg = llama_get_logits_ith(e.ctx(), -1);
        int seed = 0; float bv = lg[0]; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; seed = i; }
        auto * g = new Gen(e, N, max_trace, 1234, (mode == "json") ? JSON_GRAMMAR : nullptr); g->start((llama_pos)ptoks.size(), seed); e.add(g);
        while (!g->done()) { int a = 0; if (e.tick(64, a) < 0) break; }
        for (int j = 0; j < N; ++j) {
            std::string t = g->text[j];
            for (char & c : t) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
            fprintf(f, "%d\t%d\t%s\n", id, j, t.c_str());
        }
        fflush(f);
        printf("id=%d done\n", id); fflush(stdout);
    }
    fclose(f); e.shutdown(); return 0;
}
