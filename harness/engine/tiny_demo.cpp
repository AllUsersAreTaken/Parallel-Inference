// RecycleLM tiny-model demo: single sample vs N-way first-K-agree, prints text.
//
//   tiny_demo.exe --model M --data gsm8k_tsv --limit 5 --N 8 --K 3

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
#include <map>
#include <string>
#include <vector>

static const std::string TE = "<" + std::string("/") + "think" + ">";
static bool answer_complete(const std::string & t) {
    size_t b = t.find("\\boxed{");
    if (b == std::string::npos) return false;
    return t.find('}', b) != std::string::npos;
}
static std::string extract_answer(const std::string & t) {
    auto norm = [](const std::string & s) {
        std::string d; for (char c : s) if (std::isdigit((unsigned char)c) || c == '.' || c == '-') d += c;
        size_t i = 0; while (i < d.size() && !(std::isdigit((unsigned char)d[i]) || d[i] == '-')) ++i;
        size_t j = d.size(); while (j > i && !std::isdigit((unsigned char)d[j - 1])) --j;
        return d.substr(i, j - i);
    };
    size_t b = t.rfind("\\boxed{");
    if (b != std::string::npos) { size_t e = t.find('}', b); if (e != std::string::npos) return norm(t.substr(b + 7, e - b - 7)); }
    std::string seg = t; size_t te = t.rfind(TE); if (te != std::string::npos) seg = t.substr(te + TE.size());
    size_t last = std::string::npos;
    for (size_t i = 0; i < seg.size(); ++i) if (std::isdigit((unsigned char)seg[i]) && (i == 0 || !std::isdigit((unsigned char)seg[i - 1]))) last = i;
    if (last == std::string::npos) return "";
    size_t j = last; while (j < seg.size() && (std::isdigit((unsigned char)seg[j]) || seg[j] == '.')) ++j;
    return norm(seg.substr(last, j - last));
}

struct Ensemble : Process {
    Engine & e; int M, K, max_trace;
    std::vector<llama_seq_id> seqs; std::vector<llama_pos> pos; std::vector<llama_token> next;
    std::vector<std::string> text; std::vector<bool> fin;
    std::vector<llama_sampler *> smpls;
    std::vector<std::pair<int, std::string>> finished;
    int base_ = 0; bool complete = false; std::string answer; llama_pos p_start = 0;

    Ensemble(Engine & en, int m, int k, int mt, uint32_t seed) : e(en), M(m), K(k), max_trace(mt) {
        min_slots = 0;
        for (int j = 0; j < M; ++j) {
            seqs.push_back(j); pos.push_back(0); next.push_back(0); text.push_back(""); fin.push_back(false);
            auto sp = llama_sampler_chain_default_params(); auto * c = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(seed + 101u * (uint32_t)j));
            smpls.push_back(c);
        }
    }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }
    void start(llama_pos p0, llama_token seed) {
        for (int j = 1; j < M; ++j) e.seq_cp(0, j);
        p_start = p0; for (auto & p : pos) p = p0; for (auto & t : next) t = seed;
    }
    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        for (int j = 0; j < M; ++j) { if (fin[j]) continue; if ((int)out.size() - base >= budget) break; out.push_back({ seqs[j], pos[j], next[j], true }); }
    }
    void on_logits(int idx, const float * lg) override {
        int j = idx - base_; if (j < 0 || j >= M || fin[j]) return;
        char b[64]; int np = llama_token_to_piece(e.vocab(), next[j], b, sizeof(b), 0, false); if (np > 0) text[j].append(b, np);
        pos[j] += 1;
        if (llama_vocab_is_eog(e.vocab(), next[j]) || answer_complete(text[j]) || (int)(pos[j] - p_start) >= max_trace) {
            fin[j] = true; finished.push_back({ (int)(pos[j] - p_start), extract_answer(text[j]) }); check();
        } else next[j] = llama_sampler_sample(smpls[j], e.ctx(), idx);
    }
    void check() {
        if (complete) return;
        for (auto & f : finished) {
            if (f.second.empty()) continue;
            int c = 0; for (auto & g : finished) if (g.second == f.second) ++c;
            if (c >= K) { complete = true; answer = f.second; return; }
        }
        bool all = true; for (bool x : fin) if (!x) all = false;
        if (all) { std::string best; int bc = 0; for (auto & f : finished) { if (f.second.empty()) continue; int c = 0; for (auto & g : finished) if (g.second == f.second) ++c; if (c > bc) { bc = c; best = f.second; } } complete = true; answer = best; }
    }
    bool done() const override { return complete; }
};

int main(int argc, char ** argv) {
    std::string model, data; int limit = 5, N = 8, K = 3, max_trace = 512;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx(); else if (k == "--data") data = nx();
        else if (k == "--limit") limit = atoi(nx().c_str()); else if (k == "--N") N = atoi(nx().c_str());
        else if (k == "--K") K = atoi(nx().c_str()); else if (k == "--max-trace") max_trace = atoi(nx().c_str());
    }
    Engine::Config cfg; cfg.model = model; cfg.n_ctx = 4096 + 512; cfg.n_seq_max = N + 2; cfg.max_batch = 1024;
    llama_log_set([](ggml_log_level level, const char * text, void *) { if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }, nullptr);
    Engine e; if (!e.init(cfg)) { fprintf(stderr, "init failed\n"); return 1; }

    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };

    std::ifstream fin(data); std::string line; int id = 0;
    while (std::getline(fin, line) && id < limit) {
        size_t tab = line.find('\t'); if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab), gold = line.substr(tab + 1);
        { std::string d; for (char c : gold) if (isdigit((unsigned char)c) || c == '.' || c == '-') d += c; gold = d; }
        ++id;
        llama_chat_message msg = { "user", q.c_str() };
        std::vector<char> buf(std::max<size_t>(2048, q.size() * 2 + 512));
        int plen = llama_chat_apply_template(e.model_chat_template(), &msg, 1, true, buf.data(), (int)buf.size());
        if (plen > (int)buf.size()) { buf.resize(plen); plen = llama_chat_apply_template(e.model_chat_template(), &msg, 1, true, buf.data(), (int)buf.size()); }
        auto ptoks = tok(std::string(buf.data(), plen > 0 ? plen : 0));

        printf("\n========== Q%d  gold=%s ==========\nQ: %s\n", id, gold.c_str(), q.substr(0, 160).c_str());
        for (int M : { 1, N }) {
            e.reset_processes();
            e.clear();
            bool ok = e.prefill(0, ptoks, true); e.sync();
            int nv = e.n_vocab(); const float * lg = llama_get_logits_ith(e.ctx(), -1);
            int seed = 0; float bv = lg[0]; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; seed = i; }
            if (getenv("TINYDBG")) fprintf(stderr, "DBG Q%d M=%d ptoks=%zu prefill=%d seed=%d eog=%d\n", id, M, ptoks.size(), ok ? 1 : 0, seed, llama_vocab_is_eog(e.vocab(), seed));
            auto * pe = new Ensemble(e, M, M == 1 ? 1 : K, max_trace, 1234);
            pe->start((llama_pos)ptoks.size(), seed); e.add(pe);
            while (!pe->done()) { int a = 0; if (e.tick(64, a) < 0) break; }
            if (M == 1) {
                printf("  [single ] ans=%-10s %s\n", pe->answer.c_str(), pe->answer == gold ? "OK" : "WRONG");
                printf("           text: %s\n", pe->text[0].substr(0, 400).c_str());
            } else {
                printf("  [ensemble N=%d]\n", N);
                std::map<std::string, int> cnt; std::string vote;
                for (int j = 0; j < M; ++j) {
                    std::string a = extract_answer(pe->text[j]);
                    if (!a.empty()) cnt[a]++;
                    printf("    s%-2d ans=%-10s %s | %s\n", j, a.c_str(), a == gold ? "OK" : "..", pe->text[j].substr(0, 110).c_str());
                }
                int bc = 0; for (auto & kv : cnt) if (kv.second > bc) { bc = kv.second; vote = kv.first; }
                printf("    VOTE ans=%-10s %s (agreement %d/%d)\n", vote.c_str(), vote == gold ? "OK" : "WRONG", bc, M);
            }
        }
        fflush(stdout);
    }
    e.shutdown(); return 0;
}
