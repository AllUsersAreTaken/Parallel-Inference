// RecycleLM Test 2 - long-term memory with push and pull.
//
// For each conversation, each arm is run independently:
//   B0 no memory (2k window, early turns evicted)
//   B1 raw BM25 retrieval at the question turn (paste top-k raw exchanges)
//   B2 memory processes: push writes a one-line bullet after each exchange
//      (keeping names/numbers verbatim); at a question, BM25 over bullets plus
//      a short brief is injected
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static std::vector<std::string> words(const std::string & s) {
    std::vector<std::string> w; std::string cur;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
        else { if (!cur.empty()) { w.push_back(cur); cur.clear(); } }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}
static bool contains(const std::string & hay, const std::string & needle) {
    if (needle.empty()) return false;
    return hay.find(needle) != std::string::npos;
}

struct BM25 {
    std::vector<std::vector<std::string>> docs;
    std::vector<int> len;
    std::map<std::string, int> df;
    double avg = 1;
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

struct Question { std::string type, value, text; };
struct Conv {
    int id = 0; std::string version;
    std::vector<std::pair<std::string, std::string>> facts;
    std::vector<Question> questions;
    std::vector<std::string> turns;
};

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
        else if (p[0] == "T") cur->turns.push_back(p.size() > 1 ? p[1] : "");
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model, data, out;
    int limit = 30, window = 2000, max_reply = 160, topk = 4, brief_max = 200;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if      (k == "--model") model = nx();
        else if (k == "--data") data = nx();
        else if (k == "--out") out = nx();
        else if (k == "--limit") limit = atoi(nx().c_str());
        else if (k == "--window") window = atoi(nx().c_str());
        else if (k == "--max-reply") max_reply = atoi(nx().c_str());
        else if (k == "--topk") topk = atoi(nx().c_str());
    }
    if (model.empty() || data.empty()) { fprintf(stderr, "need --model --data\n"); return 2; }

    Engine::Config cfg;
    cfg.model = model; cfg.n_ctx = 16384; cfg.n_seq_max = 4; cfg.max_batch = 1024;
    Engine e;
    if (!e.init(cfg)) { fprintf(stderr, "init failed\n"); return 1; }

    auto convs = load_convs(data);
    FILE * f = out.empty() ? nullptr : fopen(out.c_str(), "w");
    if (f) fprintf(f, "conv,arm,qtype,recall,inserted_tok,ttft_ms,total_ms,reply_tok,capture,retrieval,use\n");

    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(e.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };
    auto piece = [&](llama_token t) { char b[64]; int n = llama_token_to_piece(e.vocab(), t, b, sizeof(b), 0, false); return std::string(b, n > 0 ? n : 0); };
    auto fmt_user = [](const std::string & u) { return "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n"; };

    // generate greedily from a prefilled prompt on seq 0; returns text + timing
    auto gen = [&](const std::vector<llama_token> & prompt, int maxt) {
        struct R { std::string text; int ntok; double ttft, total; } r{ "", 0, 0, 0 };
        e.clear();
        auto t0 = std::chrono::steady_clock::now();
        e.prefill(0, prompt, true); e.sync();
        auto tpre = std::chrono::steady_clock::now();
        r.ttft = std::chrono::duration<double, std::milli>(tpre - t0).count();
        int nv = e.n_vocab();
        auto argmax = [&](const float * lg) { int b = 0; float bv = lg[0]; for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; } return b; };
        llama_token cur = argmax(llama_get_logits_ith(e.ctx(), -1));
        llama_pos pos = (llama_pos)prompt.size();
        for (int i = 0; i < maxt; ++i) {
            if (llama_vocab_is_eog(e.vocab(), cur)) break;
            r.text += piece(cur); r.ntok += 1;
            std::vector<llama_token> one = { cur };
            e.decode_from(0, one, pos, true); e.sync();
            pos += 1;
            cur = argmax(llama_get_logits_ith(e.ctx(), -1));
        }
        r.total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    };

    for (auto & c : convs) {
        if (c.id >= limit) continue;
        int nq = (int)c.questions.size();
        int first_q = (int)c.turns.size() - nq;
        if (first_q < 0) first_q = (int)c.turns.size();

        for (int arm = 0; arm < 4; ++arm) {
            int wnd = (arm == 3) ? 16000 : window;
            std::vector<llama_token> hist;             // logical full history tokens
            std::vector<std::string> bullets;          // B2 pushed bullets
            std::vector<std::string> exch_text;        // raw exchanges for B1
            std::vector<int> exch_owner;               // exchange -> turn index
            std::vector<int> bullet_src;               // bullet -> exchange index

            for (int ti = 0; ti < (int)c.turns.size(); ++ti) {
                bool is_q = ti >= first_q;
                std::string user_msg = c.turns[ti];
                std::string inject;
                bool cap = false, ret = false;
                const Question * q = is_q ? &c.questions[ti - first_q] : nullptr;

                if (is_q && q) {
                    if (arm == 1) {
                        BM25 bm; for (auto & x : exch_text) bm.add(x); bm.finish();
                        auto hits = bm.query(q->text, topk);
                        for (auto & h : hits) { inject += exch_text[h.second] + "\n"; ret = ret || contains(exch_text[h.second], q->value); }
                        inject = "Relevant earlier notes:\n" + inject + "\n";
                    } else if (arm == 2) {
                        BM25 bm; for (auto & x : bullets) bm.add(x); bm.finish();
                        auto hits = bm.query(q->text, topk);
                        std::string notes;
                        for (auto & h : hits) { notes += "- " + bullets[h.second] + "\n"; ret = ret || contains(bullets[h.second], q->value); }
                        cap = false; for (auto & b : bullets) if (contains(b, q->value)) { cap = true; break; }
                        if (!notes.empty()) {
                            std::string bp = "<|im_start|>user\nQuestion: " + q->text + "\nNotes:\n" + notes +
                                             "\nWrite a short brief answering the question using only these notes.<|im_end|>\n<|im_start|>assistant\n";
                            auto g2 = gen(tok(bp), brief_max);
                            inject = "Brief:\n" + g2.text + "\n\n";
                        }
                    }
                }

                // build the prompt: inject + last wnd tokens of history + this user turn
                std::vector<llama_token> pt;
                if (!inject.empty()) { auto it = tok(inject); pt.insert(pt.end(), it.begin(), it.end()); }
                int start = std::max(0, (int)hist.size() - (wnd - (int)tok(fmt_user(user_msg)).size() - (int)pt.size()));
                pt.insert(pt.end(), hist.begin() + start, hist.end());
                auto ut = tok(fmt_user(user_msg)); pt.insert(pt.end(), ut.begin(), ut.end());

                auto g = gen(pt, max_reply);

                if (is_q && q) {
                    bool ok = contains(g.text, q->value);
                    if (getenv("MEMDBG")) fprintf(stderr, "DBG conv=%d arm=%d %s q=[%s] value=[%s] reply=[%s]\n",
                                                  c.id, arm, q->type.c_str(), q->text.c_str(), q->value.c_str(), g.text.substr(0, 300).c_str());
                    int inserted = (int)tok(inject).size();
                    // "use": answered given the brief/notes were present
                    fprintf(f, "%d,%d,%s,%d,%d,%.1f,%.1f,%d,%d,%d,%d\n",
                            c.id, arm, q->type.c_str(), ok ? 1 : 0, inserted, g.ttft, g.total, g.ntok,
                            (arm == 2 ? (cap ? 1 : 0) : -1), (arm == 2 ? (ret ? 1 : 0) : -1),
                            (arm == 2 ? (ok ? 1 : 0) : -1));
                    fflush(f);
                }

                // update history
                auto ut2 = tok(fmt_user(user_msg)); hist.insert(hist.end(), ut2.begin(), ut2.end());
                auto rt = tok(g.text); hist.insert(hist.end(), rt.begin(), rt.end());
                auto et = tok("<|im_end|>\n"); hist.insert(hist.end(), et.begin(), et.end());

                exch_text.push_back("User: " + user_msg + "\nAssistant: " + g.text);
                exch_owner.push_back(ti);

                if (arm == 2) {
                    // push a bullet for this exchange
                    std::string cue = "<|im_start|>user\nSummarize the last exchange in one line. Keep names and numbers exactly as written.<|im_end|>\n<|im_start|>assistant\n";
                    std::vector<llama_token> bp = hist; auto ct = tok(cue); bp.insert(bp.end(), ct.begin(), ct.end());
                    auto gb = gen(bp, 32);
                    std::string line = gb.text;
                    size_t nl = line.find('\n'); if (nl != std::string::npos) line = line.substr(0, nl);
                    bullets.push_back(line);
                    bullet_src.push_back((int)exch_text.size() - 1);
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
