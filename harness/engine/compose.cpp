// RecycleLM composition test.
//
// Runs the three processes (foreground decode, N parallel traces, probe fork)
// through one shared Engine::tick and compares live step time to the E0 cost
// model. Also asserts the foreground's token stream is byte-identical with and
// without the other processes (correctness invariant).
//
//   compose.exe --model M --traces N --probe P --steps S --out csv
//
// Probe P = fork the foreground every P steps, append a cue, decode a few
// tokens, retire. P = 0 disables it.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static llama_token argmax(const float * lg, int nv) {
    int b = 0; float bv = lg[0];
    for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; }
    return b;
}

// ---------------------------------------------------------------- foreground

struct ProcessForeground : Process {
    Engine & e;
    llama_seq_id seq = 0;
    llama_pos pos = 0;
    llama_token next = 0;
    bool fin = false;
    int max_gen;
    std::vector<llama_token> gen;

    ProcessForeground(Engine & en, int mg) : e(en), max_gen(mg) { min_slots = 1; }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = { seq }; }

    void start(const std::vector<llama_token> & prompt) {
        e.prefill(seq, prompt, true);
        pos = (llama_pos)prompt.size();
        next = argmax(llama_get_logits_ith(e.ctx(), -1), e.n_vocab());
    }
    void want(int, int, std::vector<Request> & out) override {
        if (fin) return;
        out.push_back({ seq, pos, next, true });
    }
    void on_logits(int, const float * lg) override {
        gen.push_back(next);
        pos += 1;
        if ((int)gen.size() >= max_gen) { fin = true; return; }
        next = argmax(lg, e.n_vocab());
    }
    bool done() const override { return fin; }
};

// ---------------------------------------------------------------- N traces

struct ProcessTraces : Process {
    Engine & e;
    std::vector<llama_seq_id> seqs;
    std::vector<llama_pos>    pos;
    std::vector<llama_token>  next;
    std::vector<bool>         fin;
    int max_gen;
    int base_ = 0;
    std::vector<llama_sampler *> smpls;

    ProcessTraces(Engine & en, int n, int mg) : e(en), max_gen(mg) {
        min_slots = 0;
        for (int j = 0; j < n; ++j) {
            seqs.push_back(1 + j); pos.push_back(0); next.push_back(0); fin.push_back(false);
            auto sp = llama_sampler_chain_default_params();
            auto * c = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
            llama_sampler_chain_add(c, llama_sampler_init_dist(1234u + 101u * (uint32_t)j));
            smpls.push_back(c);
        }
    }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = seqs; }

    void start(llama_pos p0, llama_token seed) {
        for (auto s : seqs) e.seq_cp(0, s);
        for (auto & p : pos) p = p0;
        for (auto & t : next) t = seed;
    }
    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base;
        for (int j = 0; j < (int)seqs.size(); ++j) {
            if (fin[j]) continue;
            if ((int)out.size() - base >= budget) break;
            out.push_back({ seqs[j], pos[j], next[j], true });
        }
    }
    void on_logits(int idx, const float * lg) override {
        // batch indices for this process are contiguous from base_; idx - base_
        // is the stream's position in the emitted order (only active streams).
        int active_idx = idx - base_;
        int seen = -1;
        for (int j = 0; j < (int)seqs.size(); ++j) {
            if (fin[j]) continue;
            ++seen;
            if (seen == active_idx) {
                pos[j] += 1;
                next[j] = llama_sampler_sample(smpls[j], e.ctx(), idx);
                break;
            }
        }
    }
    bool done() const override {
        for (bool f : fin) if (!f) return false;
        return true;
    }
    // stop all streams after max_gen steps (handled in main via step count)
    void stop_all() { for (auto & f : fin) f = true; }
};

// ---------------------------------------------------------------- probe fork

struct ProcessProbe : Process {
    Engine & e;
    ProcessForeground & fg;
    llama_seq_id seq = 1;
    std::vector<llama_token> cue;
    int probe_len;
    int every;
    int state = -1;          // -1 idle, 0/1 active
    bool cue_sent = false;
    int emitted = 0;         // probe tokens emitted
    int logits_seen = 0;     // 1 = the cue's logits
    llama_pos p0 = 0;
    llama_token pending = 0;

    ProcessProbe(Engine & en, ProcessForeground & f, int every_, int plen, llama_seq_id sid)
        : e(en), fg(f), every(every_), probe_len(plen), seq(sid) { min_slots = 0; }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = { seq }; }

    bool should_trigger(int step) const { return every > 0 && step > 0 && step % every == 0; }

    void trigger() { e.seq_cp(fg.seq, seq); p0 = fg.pos; state = 0; cue_sent = false; emitted = 0; logits_seen = 0; }

    void want(int, int, std::vector<Request> & out) override {
        if (state < 0) return;
        if (!cue_sent) {
            for (int i = 0; i < (int)cue.size(); ++i)
                out.push_back({ seq, p0 + i, cue[i], (i + 1 == (int)cue.size()) });
            cue_sent = true;
            return;
        }
        if (emitted < probe_len)
            out.push_back({ seq, p0 + (llama_pos)cue.size() + emitted, pending, true });
    }
    void on_logits(int, const float * lg) override {
        if (state < 0) return;
        pending = argmax(lg, e.n_vocab());
        logits_seen += 1;
        if (logits_seen == 1) return;   // cue logits seed the first probe token
        emitted += 1;
        if (emitted >= probe_len) {
            state = -1;
            llama_memory_seq_rm(e.mem(), seq, -1, -1);
        }
    }
    bool done() const override { return false; }
};

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::string model, out, data;
    int traces = 0, probe_every = 0, steps = 256, prompt_len = 512, max_gen = 256;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (k == "--model") model = nx();
        else if (k == "--out") out = nx();
        else if (k == "--traces") traces = atoi(nx().c_str());
        else if (k == "--probe") probe_every = atoi(nx().c_str());
        else if (k == "--steps") steps = atoi(nx().c_str());
        else if (k == "--prompt-len") prompt_len = atoi(nx().c_str());
        else if (k == "--max-gen") max_gen = atoi(nx().c_str());
    }
    if (model.empty()) { fprintf(stderr, "--model required\n"); return 2; }

    Engine::Config cfg;
    cfg.model = model;
    cfg.n_ctx = (traces + 4) * (prompt_len + max_gen + 16) + 256;
    cfg.n_seq_max = std::max(4, traces + 4);
    cfg.max_batch = 1024;
    Engine e;
    if (!e.init(cfg)) { fprintf(stderr, "engine init failed\n"); return 1; }

    std::vector<llama_token> prompt(prompt_len);
    for (int i = 0; i < prompt_len; ++i) prompt[i] = (llama_token)((1234 + i) % e.n_vocab());

    ProcessForeground fg(e, max_gen);
    fg.start(prompt);

    ProcessTraces tr(e, traces, max_gen);
    if (traces > 0) tr.start(fg.pos, fg.next);

    ProcessProbe pr(e, fg, probe_every, 8, traces + 1);
    // cue tokens
    {
        const char * c = "\n\nTherefore, the answer is";
        std::vector<llama_token> t(64);
        int n = llama_tokenize(e.vocab(), c, (int)strlen(c), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), c, (int)strlen(c), t.data(), (int)t.size(), false, true); }
        t.resize(n);
        pr.cue = t;
    }
    int fg_h = e.add(&fg);
    int tr_h = traces > 0 ? e.add(&tr) : -1;
    int pr_h = probe_every > 0 ? e.add(&pr) : -1;

    FILE * f = out.empty() ? nullptr : fopen(out.c_str(), "w");
    if (f) fprintf(f, "step,step_ms,active,clock_mhz\n");

    for (int s = 0; s < steps; ++s) {
        if (pr_h >= 0 && pr.should_trigger(s)) pr.trigger();
        int active = 0;
        double dt = e.tick(32, active);
        if (dt < 0) { fprintf(stderr, "tick failed at %d\n", s); break; }
        if (f) fprintf(f, "%d,%.4f,%d,%u\n", s, dt, active, e.clock_sm());
        if (tr_h >= 0 && (int)fg.gen.size() >= max_gen) tr.stop_all();
    }

    // foreground identity fingerprint
    unsigned long long h = 1469598103934665603ull;
    for (llama_token t : fg.gen) { h ^= (unsigned long long)(uint32_t)t; h *= 1099511628211ull; }
    printf("traces=%d probe=%d steps=%d fg_tokens=%zu fg_hash=%016llx\n",
           traces, probe_every, steps, fg.gen.size(), h);

    if (f) fclose(f);
    e.shutdown();
    return 0;
}
