// RecycleLM Test 2 - long-term memory with push and pull.
//
// Whole conversation goes through the model's real chat template with a system
// prompt. Arms:
//   B0 no memory (2k window, early turns evicted)
//   B1 raw BM25 retrieval at the question turn (paste top-k raw exchanges)
//   B2 memory processes: push writes a one-line bullet per exchange (keeping
//      names/numbers verbatim); at a question, BM25 over bullets + a short brief
//   B3 full history, large window (upper bound)
//
// Metrics: recall (string containment of the planted value), inserted tokens,
// TTFT, tokens/s. Diagnostics for B2: capture, retrieval, use.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "core.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static const char * SYS = "You are a helpful assistant. Answer questions directly and briefly.";

// Distractor bullets (plausible facts from other conversations) to test retrieval
// at scale. Values here are disjoint from the planted facts.
static const std::vector<std::string> distractor_bullets = {
    "the budget for Project Falcon is 21500 dollars", "the client prefers phone calls over email",
    "the launch date is April 8", "the supplier is called Brightwater",
    "the contract value is 9600 dollars", "the team lead is named Priya",
    "the office moved to Lisbon", "the review cadence is every two weeks",
    "the target is 45 percent", "the deadline is October 19",
    "the project is called Marlowe", "the client prefers weekly written updates",
    "the budget is 7300 dollars", "the launch date is January 27",
    "the supplier is called Northwind", "the team lead is named Diego",
    "the office moved to Tallinn", "the target is 62 percent",
    "the project is called Sablewood", "the client prefers afternoon calls",
    "the budget is 18400 dollars", "the launch date is August 3",
    "the supplier is called Ironbridge", "the team lead is named Mei",
    "the office moved to Porto", "the target is 38 percent",
    "the deadline is December 12", "the project is called Kestrel",
    "the client prefers async reviews", "the budget is 5200 dollars",
    "the launch date is May 30", "the supplier is called Hollowbrook",
    "the team lead is named Tomas", "the office moved to Riga",
    "the target is 71 percent", "the deadline is November 24",
    "the project is called Fernway", "the client prefers a biweekly digest",
    "the budget is 12800 dollars", "the launch date is September 6",
};

static std::vector<std::string> words(const std::string & s) {
    std::vector<std::string> w; std::string cur;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
        else { if (!cur.empty()) { w.push_back(cur); cur.clear(); } }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}
static bool contains(const std::string & h, const std::string & n) { return !n.empty() && h.find(n) != std::string::npos; }
// value may be "a|b" for multi-fact questions; require all parts
static bool contains_all(const std::string & h, const std::string & v) {
    size_t s = 0;
    while (s <= v.size()) {
        size_t t = v.find('|', s); if (t == std::string::npos) t = v.size();
        if (!contains(h, v.substr(s, t - s))) return false;
        s = t + 1;
    }
    return true;
}
// all parts of v present in the bullet pool (optionally restricted to indices)
static bool all_in(const std::string & v, const std::vector<std::string> & pool, const std::vector<int> * idx) {
    size_t s = 0;
    while (s <= v.size()) {
        size_t t = v.find('|', s); if (t == std::string::npos) t = v.size();
        std::string part = v.substr(s, t - s);
        bool found = false;
        if (idx) { for (int i : *idx) if (i >= 0 && i < (int)pool.size() && contains(pool[i], part)) { found = true; break; } }
        else { for (auto & b : pool) if (contains(b, part)) { found = true; break; } }
        if (!found) return false;
        s = t + 1;
    }
    return true;
}

struct BM25 {
    std::vector<std::vector<std::string>> docs; std::vector<int> len; std::map<std::string, int> df; double avg = 1;
    void add(const std::string & t) {
        auto w = words(t); docs.push_back(w); len.push_back((int)w.size());
        std::map<std::string, int> seen; for (auto & x : w) seen[x]++;
        for (auto & kv : seen) df[kv.first]++;
    }
    void finish() { double s = 0; for (int l : len) s += l; if (!len.empty()) avg = s / len.size(); }
    std::vector<std::pair<double, int>> query(const std::string & q, int topk) const {
        auto qw = words(q); std::vector<std::pair<double, int>> sc;
        const double k1 = 1.5, b = 0.75; int N = (int)docs.size();
        for (int d = 0; d < N; ++d) {
            std::map<std::string, int> tf; for (auto & t : docs[d]) tf[t]++;
            double s = 0;
            for (auto & t : qw) {
                auto it = tf.find(t); if (it == tf.end()) continue;
                auto dit = df.find(t); double n = dit == df.end() ? 1 : dit->second;
                double idf = std::log(1 + (N - n + 0.5) / (n + 0.5));
                double f = it->second;
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len[d] / avg));
            }
            if (s > 0) sc.push_back({ s, d });
        }
        std::sort(sc.begin(), sc.end(), [](auto & a, auto & b) { return a.first > b.first; });
        if ((int)sc.size() > topk) sc.resize(topk);
        return sc;
    }
};

struct GenR { std::string text; int ntok = 0; double ttft = 0, total = 0; };
struct Question { std::string type, value, text; };
struct Conv { int id = 0; std::string version; std::vector<std::pair<std::string, std::string>> facts; std::vector<Question> questions; std::vector<std::string> turns; std::vector<std::string> canned; };

static std::vector<Conv> load_convs(const std::string & path) {
    std::vector<Conv> out; std::ifstream f(path); std::string line; Conv * cur = nullptr;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> p; size_t s = 0;
        while (s <= line.size()) { size_t t = line.find('\t', s); if (t == std::string::npos) t = line.size(); p.push_back(line.substr(s, t - s)); s = t + 1; }
        if (p[0] == "C") { out.push_back({}); cur = &out.back(); cur->id = atoi(p[1].c_str()); cur->version = p.size() > 2 ? p[2] : ""; }
        else if (!cur) continue;
        else if (p[0] == "F") cur->facts.push_back({ p[1], p.size() > 2 ? p[2] : "" });
        else if (p[0] == "Q") cur->questions.push_back({ p[1], p.size() > 3 ? p[3] : "", p.size() > 4 ? p[4] : "" });
        else if (p[0] == "T") { cur->turns.push_back(p.size() > 1 ? p[1] : ""); cur->canned.push_back(p.size() > 2 ? p[2] : ""); }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model, data, out;
    int limit = 30, window = 2000, b3_window = 8000, max_reply = 150, topk = 4, brief_max = 120, only_arm = -1, distractors = 0;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--out") out = nx();
        else if (k == "--limit") limit = atoi(nx().c_str());
        else if (k == "--window") window = atoi(nx().c_str());
        else if (k == "--b3-window") b3_window = atoi(nx().c_str());
        else if (k == "--max-reply") max_reply = atoi(nx().c_str());
        else if (k == "--topk") topk = atoi(nx().c_str());
        else if (k == "--arm") only_arm = atoi(nx().c_str());
        else if (k == "--distractors") distractors = atoi(nx().c_str());
    }
    if (model.empty() || data.empty()) { fprintf(stderr, "need --model --data\n"); return 2; }

    Engine::Config cfg;
    cfg.model = model; cfg.n_ctx = b3_window + 2048; cfg.n_seq_max = 4; cfg.max_batch = 2048;
    Engine e;
    if (!e.init(cfg)) { fprintf(stderr, "init failed\n"); return 1; }
    const char * tmpl = e.model_chat_template();

    auto convs = load_convs(data);
    FILE * f = out.empty() ? nullptr : fopen(out.c_str(), "w");
    if (f) fprintf(f, "conv,arm,qtype,recall,inserted_tok,ttft_ms,total_ms,reply_tok,captured,retrieved,brief_has,use\\n");

    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };
    auto piece = [&](llama_token t) { char b[64]; int n = llama_token_to_piece(e.vocab(), t, b, sizeof(b), 0, false); return std::string(b, n > 0 ? n : 0); };
    auto apply = [&](std::vector<llama_chat_message> & msgs) {
        std::vector<char> buf(1 << 20);
        int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) { buf.resize(n); n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int)buf.size()); }
        return tok(std::string(buf.data(), n > 0 ? n : 0));
    };

    auto gen = [&](const std::vector<llama_token> & prompt, int maxt) {
        GenR r;
        e.clear();
        auto t0 = std::chrono::steady_clock::now();
        if (!e.prefill(0, prompt, true)) return r;
        e.sync();
        r.ttft = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int nv = e.n_vocab();
        auto amax = [&](const float * lg) { int b = 0; float bv = lg[0]; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; } return b; };
        llama_token cur = amax(llama_get_logits_ith(e.ctx(), -1));
        llama_pos pos = (llama_pos)prompt.size();
        for (int i = 0; i < maxt; ++i) {
            if (llama_vocab_is_eog(e.vocab(), cur)) break;
            r.text += piece(cur); r.ntok += 1;
            std::vector<llama_token> one = { cur };
            e.decode_from(0, one, pos, true); e.sync();
            pos += 1;
            cur = amax(llama_get_logits_ith(e.ctx(), -1));
        }
        r.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    };

    for (auto & c : convs) {
        if (c.id >= limit) continue;
        int nturn = (int)c.turns.size();
        int nq = (int)c.questions.size();

        for (int arm = 0; arm < 5; ++arm) {
            if (only_arm >= 0 && arm != only_arm) continue;
            int wnd = (arm == 3) ? b3_window : window;
            std::vector<std::pair<std::string, std::string>> history;   // (user, assistant)
            std::vector<std::string> bullets, exch_text;
            // distractor bullets (from other conversations) to test retrieval at scale
            for (int d = 0; d < distractors; ++d)
                bullets.push_back("Note: " + distractor_bullets[(c.id * 7 + d) % distractor_bullets.size()]);

            for (int ti = 0; ti < nturn + nq; ++ti) {
                bool is_q = ti >= nturn;
                const Question * q = is_q ? &c.questions[ti - nturn] : nullptr;
                std::string user_msg = is_q ? q->text : c.turns[ti];
                std::string inject;
                bool cap = false, ret = false;

                if (is_q && q) {
                    cap = all_in(q->value, bullets, nullptr);
                    std::vector<int> sel;
                    if (arm == 1) {
                        BM25 bm; for (auto & x : exch_text) bm.add(x); bm.finish();
                        auto hits = bm.query(q->text, topk);
                        for (auto & h : hits) inject += exch_text[h.second] + "\n";
                        if (!inject.empty()) inject = "Relevant earlier notes:\n" + inject + "\n";
                        // retrieval over raw exchanges: all parts present in some hit?
                        ret = false;
                        for (auto & h : hits) {
                            size_t s = 0; bool all = true;
                            while (s <= q->value.size()) {
                                size_t t = q->value.find('|', s); if (t == std::string::npos) t = q->value.size();
                                if (!contains(exch_text[h.second], q->value.substr(s, t - s))) { all = false; break; }
                                s = t + 1;
                            }
                            if (all) { ret = true; break; }
                        }
                    } else if (arm == 2 || arm == 4) {
                        BM25 bm; for (auto & x : bullets) bm.add(x); bm.finish();
                        auto hits = bm.query(q->text, arm == 4 ? 6 : topk);
                        for (auto & h : hits) sel.push_back(h.second);
                        if (arm == 4) {
                            // model reads the whole store and picks the relevant bullets
                            std::string all;
                            for (int i = 0; i < (int)bullets.size(); ++i) all += std::to_string(i) + ". " + bullets[i] + "\n";
                            std::string sel_user = "Notes:\n" + all + "\nWhich note numbers are relevant to this question: " +
                                                   q->text + "\nReply with only the numbers, comma separated.";
                            std::vector<llama_chat_message> sm = { { "system", SYS }, { "user", sel_user.c_str() } };
                            auto gs = gen(apply(sm), 24);
                            std::vector<int> msel;
                            for (auto & w : words(gs.text)) { int idx = atoi(w.c_str()); if (idx >= 0 && idx < (int)bullets.size()) msel.push_back(idx); }
                            if (!msel.empty()) sel = msel;
                        }
                        ret = all_in(q->value, bullets, &sel);
                        std::string notes;
                        for (int i : sel) notes += "- " + bullets[i] + "\n";
                        if (arm == 2 && !notes.empty()) {
                            std::string brief_user = "Using only these notes, answer briefly: " + q->text + "\n" + notes;
                            std::vector<llama_chat_message> bm_msgs = { { "system", SYS }, { "user", brief_user.c_str() } };
                            auto g2 = gen(apply(bm_msgs), brief_max);
                            inject = "Brief: " + g2.text + "\n\n";
                        } else if (arm == 4 && !notes.empty()) {
                            inject = "Relevant notes:\n" + notes + "\n";   // verbatim, no rewrite
                        }
                    }
                }

                // build messages from full history, dropping oldest until it fits the window
                std::string user_content = inject + user_msg + (is_q ? " Answer in one short sentence." : "");
                std::vector<llama_token> prompt;
                size_t drop = 0;
                while (true) {
                    std::vector<llama_chat_message> msgs;
                    msgs.push_back({ "system", SYS });
                    for (size_t h = drop; h < history.size(); ++h) {
                        msgs.push_back({ "user", history[h].first.c_str() });
                        msgs.push_back({ "assistant", history[h].second.c_str() });
                    }
                    msgs.push_back({ "user", user_content.c_str() });
                    prompt = apply(msgs);
                    if ((int)prompt.size() <= wnd || drop >= history.size()) break;
                    drop += 1;
                }

                GenR g;
                if (!is_q && ti < (int)c.canned.size() && !c.canned[ti].empty()) { g.text = c.canned[ti]; }
                else { g = gen(prompt, max_reply); }

                if (getenv("MEMDBG2")) {
                    std::string pt;
                    for (size_t z = prompt.size() > 160 ? prompt.size() - 160 : 0; z < prompt.size(); ++z) pt += piece(prompt[z]);
                    fprintf(stderr, "PROMPT_TAIL c=%d arm=%d hist=%zu drop=%zu user_content=[%s]\nTAIL=[%s]\n",
                            c.id, arm, history.size(), drop, user_content.c_str(), pt.c_str());
                }

                if (is_q && q) {
                    bool ok = contains_all(g.text, q->value);
                    bool bh = contains_all(inject, q->value);
                    if (getenv("MEMDBG")) fprintf(stderr, "PSZ=%zu DBG c=%d arm=%d %s q=[%s] v=[%s] reply=[%s]\n",
                                                  prompt.size(), c.id, arm, q->type.c_str(), q->text.c_str(), q->value.c_str(), g.text.substr(0, 200).c_str());
                    fprintf(f, "%d,%d,%s,%d,%d,%.1f,%.1f,%d,%d,%d,%d,%d\n",
                            c.id, arm, q->type.c_str(), ok ? 1 : 0, (int)tok(inject).size(), g.ttft, g.total, g.ntok,
                            cap ? 1 : 0, ret ? 1 : 0, bh ? 1 : 0, ok ? 1 : 0);
                    fflush(f);
                }

                history.push_back({ user_msg, g.text });
                exch_text.push_back("User: " + user_msg + "\nAssistant: " + g.text);

                if (arm == 2 || arm == 4) {
                    std::string push_user = "Summarize this exchange in one line. Keep names and numbers exactly as written.\nUser: "
                                            + user_msg + "\nAssistant: " + g.text;
                    std::vector<llama_chat_message> bm_msgs = { { "system", SYS }, { "user", push_user.c_str() } };
                    auto bp = apply(bm_msgs);
                    auto gb = gen(bp, 40);
                    std::string line = gb.text;
                    size_t nl = line.find('\n'); if (nl != std::string::npos) line = line.substr(0, nl);
                    bullets.push_back(line);
                }
            }
        }
        printf("conv %d done\n", c.id);
        fflush(stdout);
    }

    if (f) fclose(f);
    e.shutdown();
    return 0;
}
