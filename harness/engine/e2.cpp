// RecycleLM E2 - multitrack thinking, live on the 4060 through the engine.
//
// For each problem, runs two arms:
//   A1: M=1 stream (single sampled trace)
//   A3: M=16 streams, first-K-agree (K=3)
// Records accuracy and per-problem wall clock (sum of decode step times).
//
//   e2.exe --model M --data heldout.tsv --out csv --limit 100 --K 3

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

static const std::string TE = "<" + std::string("/") + "think" + ">";

static bool answer_complete(const std::string & t) {
    if (t.find(TE) == std::string::npos) return false;
    size_t b = t.find("\\boxed{", t.find(TE));
    if (b == std::string::npos) return false;
    return t.find('}', b) != std::string::npos;
}

// normalized numeric answer, or empty
static std::string extract_answer(const std::string & t) {
    auto norm = [](const std::string & s) -> std::string {
        std::string d;
        for (char c : s) if (std::isdigit((unsigned char)c) || c == '.' || c == '-') d += c;
        size_t i = 0;
        while (i < d.size() && !(std::isdigit((unsigned char)d[i]) || d[i] == '-')) ++i;
        size_t j = d.size();
        while (j > i && !std::isdigit((unsigned char)d[j - 1])) --j;
        return d.substr(i, j - i);
    };
    size_t b = t.rfind("\\boxed{");
    if (b != std::string::npos) {
        size_t e = t.find('}', b);
        if (e != std::string::npos) return norm(t.substr(b + 7, e - b - 7));
    }
    std::string seg = t;
    size_t te = t.rfind(TE);
    if (te != std::string::npos) seg = t.substr(te + TE.size());
    // last number in seg
    size_t last = std::string::npos;
    for (size_t i = 0; i < seg.size(); ++i)
        if (std::isdigit((unsigned char)seg[i]) && (i == 0 || !std::isdigit((unsigned char)seg[i - 1]))) last = i;
    if (last == std::string::npos) return "";
    size_t j = last;
    while (j < seg.size() && (std::isdigit((unsigned char)seg[j]) || seg[j] == '.')) ++j;
    return norm(seg.substr(last, j - last));
}

// ---------------------------------------------------------------- ensemble

struct ProcessEnsemble : Process {
    Engine & e;
    int M, K, max_trace;
    std::vector<llama_seq_id> seqs;
    std::vector<llama_pos>    pos;
    std::vector<llama_token>  next;
    std::vector<std::string>  text;
    std::vector<bool>         fin;
    std::vector<llama_sampler *> smpls;
    std::vector<std::pair<int, std::string>> finished; // (finish_len, answer)
    int base_ = 0;
    std::vector<int> emitted;   // stream index per batch slot, in order
    llama_pos p_start = 0;      // first generated position (prompt length)
    bool complete = false;
    int finish_len = 0;
    std::string answer;
    long long total_tokens = 0;   // sum of decoded tokens across streams
    int steps = 0;

    ProcessEnsemble(Engine & en, int m, int k, int mt, uint32_t seed) : e(en), M(m), K(k), max_trace(mt) {
        min_slots = 0;
        for (int j = 0; j < M; ++j) {
            seqs.push_back(j); pos.push_back(0); next.push_back(0); text.push_back(""); fin.push_back(false);
            auto sp = llama_sampler_chain_default_params();
            auto * c = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(seed + 101u * (uint32_t)j));
            smpls.push_back(c);
        }
    }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }

    void start(llama_pos p0, llama_token seed_tok) {
        for (int j = 1; j < M; ++j) e.seq_cp(0, j);
        p_start = p0;
        for (auto & p : pos) p = p0;
        for (auto & t : next) t = seed_tok;
    }
    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        emitted.clear();
        for (int j = 0; j < M; ++j) {
            if (fin[j]) continue;
            if ((int)emitted.size() >= budget) break;
            emitted.push_back(j);
            out.push_back({ seqs[j], pos[j], next[j], true });
        }
    }
    void on_logits(int idx, const float * lg) override {
        int slot = idx - base_;
        if (slot < 0 || slot >= (int)emitted.size()) return;
        int j = emitted[slot];
        if (fin[j]) return;
        char piece[64];
        int np = llama_token_to_piece(e.vocab(), next[j], piece, sizeof(piece), 0, false);
        if (np > 0) text[j].append(piece, np);
        pos[j] += 1;
        total_tokens += 1;
        if (llama_vocab_is_eog(e.vocab(), next[j]) || answer_complete(text[j]) || (int)(pos[j] - p_start) >= max_trace) {
            fin[j] = true;
            finished.push_back({ (int)pos[j], extract_answer(text[j]) });
            check_agree();
        } else {
            next[j] = llama_sampler_sample(smpls[j], e.ctx(), idx);
        }
    }
    void check_agree() {
        if (complete) return;
        for (int a = (int)finished.size() - 1; a >= 0; --a) {
            if (finished[a].second.empty()) continue;
            int cnt = 0;
            for (auto & f : finished) if (f.second == finished[a].second) ++cnt;
            if (cnt >= K) {
                complete = true;
                answer = finished[a].second;
                // finish_len = the max finish length among the agreeing ones
                int fl = 0;
                for (auto & f : finished) if (f.second == answer) fl = std::max(fl, f.first);
                finish_len = fl;
                return;
            }
        }
        // all finished, no K-agreement -> majority
        bool all = true;
        for (bool f : fin) if (!f) all = false;
        if (all) {
            complete = true;
            std::string best; int bc = 0;
            for (auto & f : finished) {
                if (f.second.empty()) continue;
                int cnt = 0; for (auto & g : finished) if (g.second == f.second) ++cnt;
                if (cnt > bc) { bc = cnt; best = f.second; }
            }
            answer = best;
            finish_len = 0; for (auto & f : finished) finish_len = std::max(finish_len, f.first);
        }
    }
    bool done() const override { return complete; }
};

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::string model, data, out;
    int limit = 100, K = 3, max_trace = 1024, prompt_len = 0, sweep = 0;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--out") out = nx();
        else if (k == "--limit") limit = atoi(nx().c_str());
        else if (k == "--K") K = atoi(nx().c_str());
        else if (k == "--max-trace") max_trace = atoi(nx().c_str());
        else if (k == "--sweep") sweep = 1;
    }
    if (model.empty() || data.empty()) { fprintf(stderr, "need --model --data\n"); return 2; }

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = 512 + max_trace + 64;         // per sequence; unified pool set below
    cfg.n_ctx = 20 * (512 + max_trace + 16) + 256;
    cfg.n_seq_max = 24;
    cfg.max_batch = 1024;
    Engine e;
    if (!e.init(cfg)) { fprintf(stderr, "engine init failed\n"); return 1; }

    std::ifstream fin(data);
    FILE * f = out.empty() ? nullptr : fopen(out.c_str(), "w");
    if (f && sweep) fprintf(f, "id,gold,N,K,ans,correct,wall_ms,tokens\n");
    else if (f) fprintf(f, "id,gold,a1_ans,a1_correct,a1_wall_ms,a1_tok,a3_ans,a3_correct,a3_wall_ms,a3_finish,a3_tok,a2_ans,a2_correct,a2_wall_ms,a2_tok,a1r_ans,a1r_correct,a1r_wall_ms,a1r_finish,a1r_tok\n");

    std::string line;
    int id = 0;
    while (std::getline(fin, line) && id < limit) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        std::string gold = line.substr(tab + 1);
        // normalize gold
        { std::string d; for (char c : gold) if (isdigit((unsigned char)c) || c == '.' || c == '-') d += c; gold = d; }
        ++id;

        // build prompt with the model's chat template
        llama_chat_message msg = { "user", q.c_str() };
        std::vector<char> buf(std::max<size_t>(1024, q.size() * 2 + 512));
        int plen = llama_chat_apply_template(e.model_chat_template(), &msg, 1, true, buf.data(), (int)buf.size());
        if (plen > (int)buf.size()) { buf.resize(plen); plen = llama_chat_apply_template(e.model_chat_template(), &msg, 1, true, buf.data(), (int)buf.size()); }
        std::vector<llama_token> ptoks(buf.size() + 16);
        int pn = llama_tokenize(e.vocab(), buf.data(), plen, ptoks.data(), (int)ptoks.size(), false, true);
        if (pn < 0) { ptoks.resize(-pn); pn = llama_tokenize(e.vocab(), buf.data(), plen, ptoks.data(), (int)ptoks.size(), false, true); }
        ptoks.resize(pn);

        // run one ensemble arm; returns answer, wall clock, tokens, stream-0 text
        auto run_arm = [&](int M, int kk, uint32_t seed, double & wall_out, std::string & ans_out, int & fin_out,
                           long long & tok_out, std::string & text_out) {
            e.clear();
            auto t0 = std::chrono::steady_clock::now();
            e.prefill(0, ptoks, true);
            e.sync();
            double pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            llama_token seed_tok = 0;
            { const float * lg = llama_get_logits_ith(e.ctx(), -1); int nv = e.n_vocab(); float bv = lg[0]; int bi = 0; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; bi = i; } seed_tok = bi; }
            auto * pe = new ProcessEnsemble(e, M, kk, max_trace, seed);
            pe->start((llama_pos)ptoks.size(), seed_tok);
            e.add(pe);
            double wall = pre_ms;
            while (!pe->done()) { int active = 0; double dt = e.tick(64, active); if (dt < 0) break; wall += dt; }
            wall_out = wall; ans_out = pe->answer; fin_out = pe->finish_len;
            tok_out = pe->total_tokens; text_out = pe->text.empty() ? "" : pe->text[0];
        };
        // repeated-suffix loop detector
        auto is_loop = [](const std::string & t) {
            if (t.size() < 120) return false;
            std::string suf = t.substr(t.size() - 40);
            return t.find(suf) < t.size() - 40;
        };

        if (sweep) {
            for (int N : { 1, 2, 4, 8, 16 }) {
                int kk = N < 3 ? N : 3;
                double w = 0; std::string a, t; int fin = 0; long long nt = 0;
                run_arm(N, kk, 1234, w, a, fin, nt, t);
                int c = (!a.empty() && a == gold) ? 1 : 0;
                fprintf(f, "%d,%s,%d,%d,%s,%d,%.1f,%lld\n", id, gold.c_str(), N, kk, a.c_str(), c, w, nt);
            }
            fflush(f);
            printf("id=%d done\n", id);
            fflush(stdout);
            continue;
        }

        double w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        std::string a0, a1, a2, a3, t0s, t1s, t2s, t3s;
        int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
        long long n0 = 0, n1 = 0, n2 = 0, n3 = 0;
        run_arm(1, 1, 1234, w0, a0, f0, n0, t0s);          // A1: single sampled trace
        run_arm(16, K, 1234, w1, a1, f1, n1, t1s);         // A3: N=16, first-K-agree
        run_arm(16, 16, 1234, w2, a2, f2, n2, t2s);        // A2: N=16 majority vote (all finish)
        // A1r: single trace, repeated-suffix loop detection + restart (up to 3 attempts)
        for (int att = 0; att < 3; ++att) {
            double w = 0; std::string a, t; int fin = 0; long long nt = 0;
            run_arm(1, 1, 1234u + 777u * (uint32_t)att, w, a, fin, nt, t);
            w3 += w; n3 += nt; a3 = a; f3 = fin;
            if (!a.empty() && !is_loop(t) && fin < max_trace) break;
        }
        int c0 = (!a0.empty() && a0 == gold) ? 1 : 0;
        int c1 = (!a1.empty() && a1 == gold) ? 1 : 0;
        int c2 = (!a2.empty() && a2 == gold) ? 1 : 0;
        int c3 = (!a3.empty() && a3 == gold) ? 1 : 0;
        fprintf(f, "%d,%s,%s,%d,%.1f,%lld,%s,%d,%.1f,%d,%lld,%s,%d,%.1f,%lld,%s,%d,%.1f,%d,%lld\n",
                id, gold.c_str(),
                a0.c_str(), c0, w0, n0,
                a1.c_str(), c1, w1, f1, n1,
                a2.c_str(), c2, w2, n2,
                a3.c_str(), c3, w3, f3, n3);
        fflush(f);
        printf("id=%d done\n", id);
        fflush(stdout);
    }
    if (f) fclose(f);
    e.shutdown();
    return 0;
}
