// RecycleLM context-compaction demo.
//
// Shared prefix (common instructions + history + latest user message), forked:
//   seq 0: "answer the latest user message"
//   seq 1: "produce a compact memory of this conversation"
// Both task instructions are appended AFTER the shared history.
//
// Configs per conversation: FULL (answer only), SERIAL (answer then memory),
// PARALLEL (both, forked). Then a follow-up question is answered from the full
// context and from the compacted context, and scored.

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
#include <vector>

static const char * COMMON =
    "You are a helpful assistant. Use only the conversation above. Do not invent facts.";
static const char * ANSWER_TASK = "Task: answer the latest user message above.";
static const char * MEM_TASK =
    "Task: produce a compact memory of this conversation, for use as context on later turns. "
    "Use these sections and keep exact wording for facts:\n"
    "CURRENT GOAL:\nDECISIONS AND LATEST CORRECTIONS:\nCONSTRAINTS AND PREFERENCES:\n"
    "EXACT FACTS / NUMBERS:\nOPEN QUESTIONS / NEXT STEPS:\n"
    "Preserve negations, uncertainty, and changes of mind. Do not invent missing information. "
    "Omit conversational filler.";

struct Gen : Process {
    Engine & e; llama_seq_id seq; int max_trace;
    std::vector<llama_pos> pos; std::vector<llama_token> next;
    std::string text; bool fin = false;
    std::vector<llama_sampler *> smpls;
    int base_ = 0; llama_pos p_start = 0;
    Gen(Engine & en, llama_seq_id s, int mt, uint32_t seed) : e(en), seq(s), max_trace(mt) {
        min_slots = 1; pos.push_back(0); next.push_back(0);
        auto sp = llama_sampler_chain_default_params(); auto * c = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(c, llama_sampler_init_temp(0.6f));
        llama_sampler_chain_add(c, llama_sampler_init_dist(seed));
        smpls.push_back(c);
    }
    void seq_ids(std::vector<llama_seq_id> & o) const override { o = { seq }; }
    void start(llama_pos p0, llama_token seed) { p_start = p0; pos[0] = p0; next[0] = seed; }
    void want(int, int base, std::vector<Request> & out) override {
        base_ = base;
        if (fin) return;
        out.push_back({ seq, pos[0], next[0], true });
    }
    void on_logits(int idx, const float * lg) override {
        if (fin || idx != base_) return;
        char b[64]; int np = llama_token_to_piece(e.vocab(), next[0], b, sizeof(b), 0, false);
        if (np > 0) text.append(b, np);
        pos[0] += 1;
        if (llama_vocab_is_eog(e.vocab(), next[0]) || (int)(pos[0] - p_start) >= max_trace) fin = true;
        else next[0] = llama_sampler_sample(smpls[0], e.ctx(), idx);
    }
    bool done() const override { return fin; }
};

static std::string clean(std::string s) { for (char & c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' '; return s; }
static bool has(const std::string & h, const std::string & n) { return !n.empty() && h.find(n) != std::string::npos; }

int main(int argc, char ** argv) {
    std::string model, data, out;
    int limit = 10, mem_cap = 384, ans_cap = 160;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx(); else if (k == "--data") data = nx(); else if (k == "--out") out = nx();
        else if (k == "--limit") limit = atoi(nx().c_str()); else if (k == "--mem-cap") mem_cap = atoi(nx().c_str());
    }
    Engine::Config cfg; cfg.model = model; cfg.n_ctx = 8192; cfg.n_seq_max = 4; cfg.max_batch = 2048;
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(t, stderr); }, nullptr);
    Engine e; if (!e.init(cfg)) { fprintf(stderr, "init failed\n"); return 1; }

    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };
    auto apply = [&](std::vector<llama_chat_message> & msgs) {
        std::vector<char> buf(1 << 20);
        int n = llama_chat_apply_template(e.model_chat_template(), msgs.data(), msgs.size(), true, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) { buf.resize(n); n = llama_chat_apply_template(e.model_chat_template(), msgs.data(), msgs.size(), true, buf.data(), (int)buf.size()); }
        return tok(std::string(buf.data(), n > 0 ? n : 0));
    };

    // generate from a prompt on a given seq; returns text + timing
    struct R { std::string text; double ttft, total; int ntok; };
    auto gen = [&](const std::vector<llama_token> & prompt, int maxt, llama_seq_id seq, uint32_t seed) {
        R r{ "", 0, 0, 0 };
        e.clear();
        auto t0 = std::chrono::steady_clock::now();
        e.prefill(seq, prompt, true); e.sync();
        r.ttft = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int nv = e.n_vocab();
        auto amax = [&](const float * lg) { int b = 0; float bv = lg[0]; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; } return b; };
        llama_token cur = amax(llama_get_logits_ith(e.ctx(), -1));
        llama_pos pos = (llama_pos)prompt.size();
        for (int i = 0; i < maxt; ++i) {
            if (llama_vocab_is_eog(e.vocab(), cur)) break;
            char b[64]; int np = llama_token_to_piece(e.vocab(), cur, b, sizeof(b), 0, false); if (np > 0) r.text.append(b, np);
            r.ntok += 1;
            e.decode_from(seq, { cur }, pos, true); e.sync(); pos += 1;
            cur = amax(llama_get_logits_ith(e.ctx(), -1));
        }
        r.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    };

    std::ifstream fin(data); std::string line;
    FILE * f = fopen(out.c_str(), "w");
    if (f) fprintf(f, "conv,config,prefix_tok,ans_ttft_ms,ans_total_ms,ans_tok,mem_total_ms,mem_tok,both_ready_ms,"
                      "full_follow_ok,compact_follow_ok,compact_stale,full_prefix_tok,compact_prefix_tok\n");

    int id = 0;
    while (std::getline(fin, line) && id < limit) {
        std::vector<std::string> p; size_t s = 0;
        while (s <= line.size()) { size_t t = line.find('\t', s); if (t == std::string::npos) t = line.size(); p.push_back(line.substr(s, t - s)); s = t + 1; }
        if (p.size() < 6) continue;
        ++id;
        std::string history = p[1], latest = p[2], followup = p[3], gold = p[4], stale = p[5];

        // build messages: system + history turns + latest user
        std::vector<llama_chat_message> base;
        base.push_back({ "system", COMMON });
        { std::string seg; std::vector<std::string> parts;
          size_t q = 0; while (q <= history.size()) { size_t t = history.find(" | ", q); if (t == std::string::npos) t = history.size(); parts.push_back(history.substr(q, t - q)); q = t + 3; }
          for (auto & m : parts) { if (m.rfind("user: ", 0) == 0) base.push_back({ "user", strdup(m.substr(6).c_str()) }); else if (m.rfind("assistant: ", 0) == 0) base.push_back({ "assistant", strdup(m.substr(11).c_str()) }); }
        }
        base.push_back({ "user", latest.c_str() });

        auto prefix = apply(base);                       // shared prefix tokens
        std::vector<llama_chat_message> am = base; am.push_back({ "user", ANSWER_TASK });
        std::vector<llama_chat_message> mm = base; mm.push_back({ "user", MEM_TASK });
        auto ans_prompt = apply(am), mem_prompt = apply(mm);

        // ---- FULL
        R full = gen(ans_prompt, ans_cap, 0, 1234);

        // ---- SERIAL
        R sa = gen(ans_prompt, ans_cap, 0, 1234);
        R sm = gen(mem_prompt, mem_cap, 0, 1234);
        fprintf(f, "%d,SERIAL,%zu,%.1f,%.1f,%d,%.1f,%d,%.1f,-1,-1,-1,-1,-1\n",
                id, prefix.size(), sa.ttft, sa.total, sa.ntok, sm.total, (int)tok(sm.text).size(), sa.total + sm.total);

        // ---- PARALLEL: prefill shared prefix, fork, append each task suffix
        e.reset_processes(); e.clear();
        auto t0 = std::chrono::steady_clock::now();
        e.prefill(0, prefix, false); e.sync();
        e.seq_cp(0, 1);
        std::vector<llama_token> asfx(ans_prompt.begin() + prefix.size(), ans_prompt.end());
        std::vector<llama_token> msfx(mem_prompt.begin() + prefix.size(), mem_prompt.end());
        {
            std::vector<Request> rq;
            for (size_t k = 0; k < asfx.size(); ++k) rq.push_back({ 0, (llama_pos)(prefix.size() + k), asfx[k], k + 1 == asfx.size() });
            for (size_t k = 0; k < msfx.size(); ++k) rq.push_back({ 1, (llama_pos)(prefix.size() + k), msfx[k], k + 1 == msfx.size() });
            e.decode_requests(rq);
        }
        e.sync();
        double pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        llama_token aseed = 0, mseed = 0;
        { int nv = e.n_vocab(); const float * la = llama_get_logits_ith(e.ctx(), (int)asfx.size() - 1); float bv = la[0]; for (int i = 1; i < nv; ++i) if (la[i] > bv) { bv = la[i]; aseed = i; } }
        { int nv = e.n_vocab(); const float * lm = llama_get_logits_ith(e.ctx(), (int)asfx.size() + (int)msfx.size() - 1); float bv = lm[0]; for (int i = 1; i < nv; ++i) if (lm[i] > bv) { bv = lm[i]; mseed = i; } }
        auto * ga = new Gen(e, 0, ans_cap, 1234); ga->start((llama_pos)(prefix.size() + asfx.size()), aseed);
        auto * gm = new Gen(e, 1, mem_cap, 1234); gm->start((llama_pos)(prefix.size() + msfx.size()), mseed);
        e.add(ga); e.add(gm);
        double ans_done = 0, mem_done = 0, tick_sum = 0;
        while (!ga->done() || !gm->done()) {
            int act = 0; double dt = e.tick(64, act); if (dt < 0) break;
            tick_sum += dt;
            if (ans_done == 0 && ga->done()) ans_done = pre_ms + tick_sum;
            if (mem_done == 0 && gm->done()) mem_done = pre_ms + tick_sum;
        }
        if (ans_done == 0) ans_done = pre_ms + tick_sum;
        if (mem_done == 0) mem_done = pre_ms + tick_sum;
        R par = ga->text.empty() ? full : R{ ga->text, ans_done, ans_done, (int)ga->text.size() };
        (void)par;
        std::string mem = gm->text;
        fprintf(f, "%d,PARALLEL,%zu,%.1f,%.1f,%d,%.1f,%d,%.1f,-1,-1,-1,-1,-1\n",
                id, prefix.size(), pre_ms, ans_done, (int)tok(ga->text).size(), mem_done, (int)tok(mem).size(), std::max(ans_done, mem_done));

        // ---- follow-up: full context vs compacted context
        std::vector<llama_chat_message> ff = base;
        ff.push_back({ "user", ANSWER_TASK }); ff.push_back({ "assistant", strdup(ga->text.c_str()) });
        ff.push_back({ "user", followup.c_str() });
        auto ffp = apply(ff); R full_follow = gen(ffp, 120, 0, 1234);

        std::string mem_clean = mem;
        std::vector<llama_chat_message> cf;
        cf.push_back({ "system", COMMON });
        cf.push_back({ "user", strdup(("Compact memory of earlier conversation:\n" + mem_clean).c_str()) });
        cf.push_back({ "user", latest.c_str() });
        cf.push_back({ "assistant", strdup(ga->text.c_str()) });
        cf.push_back({ "user", followup.c_str() });
        auto cfp = apply(cf); R comp_follow = gen(cfp, 120, 0, 1234);

        int f_ok = has(full_follow.text, gold) ? 1 : 0;
        int c_ok = has(comp_follow.text, gold) ? 1 : 0;
        int c_stale = has(comp_follow.text, stale) ? 1 : 0;
        fprintf(f, "%d,FOLLOWUP,%zu,-1,-1,-1,-1,-1,-1,%d,%d,%d,%d,%d\n",
                id, prefix.size(), f_ok, c_ok, c_stale, (int)ffp.size(), (int)cfp.size());

        printf("conv %d done (mem %d tok, followup full=%d compact=%d stale=%d)\n", id, (int)mem.size(), f_ok, c_ok, c_stale);
        fflush(stdout); fflush(f);
    }
    if (f) fclose(f); e.shutdown(); return 0;
}
