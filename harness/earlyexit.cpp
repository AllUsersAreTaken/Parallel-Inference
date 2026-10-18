// RecycleLM - offline early-exit probe tool
//
// For each problem: prefill the chat prompt, generate a full greedy reasoning
// trace on sequence 0, and at fixed token checkpoints fork the KV (seq_cp),
// append a forced-answer cue, decode a short probe, and record the answer.
// This is the offline measurement the live harness would only have to confirm.
//
// Outputs (prefix from --out):
//   <out>.traces.tsv    id  gold  trace_len  final_text
//   <out>.probes.tsv    id  checkpoint  gen_count  probe_text
//   <out>.logprobs.tsv  id  tok_index  logprob

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct args {
    std::string model;
    std::string data;
    std::string out = "results/ee";
    int   limit      = 100;
    std::vector<int> checkpoints = {64, 128, 256};
    int   probe_len  = 16;
    int   max_trace  = 1024;
    int   n_ctx      = 4096;
    int   ngl        = 99;
    float temp       = 0.6f;
    uint32_t seed    = 1234;
    std::string cue  = "\n\nTherefore, the answer is";
};

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); }
            return argv[++i];
        };
        if      (k == "--model") a.model = need();
        else if (k == "--data")  a.data  = need();
        else if (k == "--out")   a.out   = need();
        else if (k == "--limit") a.limit = atoi(need().c_str());
        else if (k == "--probe-len") a.probe_len = atoi(need().c_str());
        else if (k == "--max-trace") a.max_trace = atoi(need().c_str());
        else if (k == "--n-ctx") a.n_ctx = atoi(need().c_str());
        else if (k == "--ngl")   a.ngl = atoi(need().c_str());
        else if (k == "--temp")  a.temp = (float)atof(need().c_str());
        else if (k == "--seed")  a.seed = (uint32_t)strtoul(need().c_str(), nullptr, 10);
        else if (k == "--cue")   a.cue = need();
        else if (k == "--checkpoints") {
            a.checkpoints.clear();
            std::string v = need();
            size_t p = 0;
            while (p <= v.size()) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                a.checkpoints.push_back(atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return false; }
    }
    if (a.model.empty() || a.data.empty()) { fprintf(stderr, "--model and --data required\n"); return false; }
    return true;
}

// greedy sample + logprob of the chosen token
static llama_token sample_greedy(const float * logits, int32_t nv, double & logp_out) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < nv; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    double mx = logits[0];
    for (int i = 1; i < nv; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < nv; ++i) sum += std::exp((double)logits[i] - mx);
    logp_out = ((double)logits[best] - mx) - std::log(sum);
    return best;
}

static std::string clean(std::string s) {
    for (char & c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return s;
}

// raw-logit stats: argmax and its log-probability (model confidence)
static void logits_stats(const float * lg, int32_t nv, int & amax, double & maxlogp) {
    int best = 0; float bv = lg[0]; double mx = lg[0];
    for (int i = 1; i < nv; ++i) {
        if (lg[i] > bv) { bv = lg[i]; best = i; }
        if (lg[i] > mx) mx = lg[i];
    }
    double s = 0.0;
    for (int i = 0; i < nv; ++i) s += std::exp((double)lg[i] - mx);
    amax = best;
    maxlogp = (double)bv - (mx + std::log(s));
}

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) return 2;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    mp.progress_callback = [](float, void *) { return true; };

    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t nv = llama_vocab_n_tokens(vocab);

    const char * tmpl = llama_model_chat_template(model, nullptr);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = a.n_ctx; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_seq_max = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified = true;
    cp.n_threads = 8; cp.n_threads_batch = 8;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // trace sampler: temperature sampling (R1-distill degenerates under greedy)
    llama_sampler * smpl = nullptr;
    if (a.temp > 0.0f) {
        auto sp = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(a.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(a.seed));
    }

    // cue tokens
    std::vector<llama_token> cue = [&] {
        std::vector<llama_token> t(a.cue.size() + 16);
        int n = llama_tokenize(vocab, a.cue.c_str(), (int)a.cue.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, a.cue.c_str(), (int)a.cue.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n);
        return t;
    }();

    FILE * ftr = fopen((a.out + ".traces.tsv").c_str(), "w");
    FILE * fpr = fopen((a.out + ".probes.tsv").c_str(), "w");
    FILE * flp = fopen((a.out + ".logprobs.tsv").c_str(), "w");
    if (!ftr || !fpr || !flp) { fprintf(stderr, "cannot open outputs\n"); return 1; }

    std::ifstream fin(a.data);
    std::string line;
    int id = 0;
    llama_batch batch = llama_batch_init(4096, 0, 1);

    auto decode_tokens = [&](const std::vector<llama_token> & toks, llama_seq_id seq, llama_pos pos0, bool logits_last) -> bool {
        for (size_t off = 0; off < toks.size(); off += cp.n_batch) {
            int n = (int)std::min<size_t>(cp.n_batch, toks.size() - off);
            for (int i = 0; i < n; ++i) {
                batch.token[i] = toks[off + i];
                batch.pos[i] = pos0 + (llama_pos)(off + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = (logits_last && (off + i + 1 == toks.size())) ? 1 : 0;
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) return false;
        }
        return true;
    };

    while (std::getline(fin, line) && id < a.limit) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        std::string gold = line.substr(tab + 1);
        ++id;

        // chat prompt
        llama_chat_message msg = { "user", q.c_str() };
        std::vector<char> buf(std::max<size_t>(1024, q.size() * 2 + 512));
        int plen = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int)buf.size());
        if (plen > (int)buf.size()) { buf.resize(plen); plen = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int)buf.size()); }
        std::string prompt(buf.data(), plen);

        std::vector<llama_token> ptoks(prompt.size() + 16);
        int pn = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), ptoks.data(), (int)ptoks.size(), false, true);
        if (pn < 0) { ptoks.resize(-pn); pn = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), ptoks.data(), (int)ptoks.size(), false, true); }
        ptoks.resize(pn);

        llama_memory_clear(mem, true);
        if (!decode_tokens(ptoks, 0, 0, true)) { fprintf(stderr, "prefill failed id=%d\n", id); continue; }

        llama_pos L = (llama_pos)ptoks.size();
        std::string final_text;
        std::vector<llama_token> gen;
        int gen_count = 0;

        // sample the first trace token from the prefill logits
        int amax = 0;
        double logp = 0.0;
        logits_stats(llama_get_logits_ith(ctx, -1), nv, amax, logp);
        llama_token tok = smpl ? llama_sampler_sample(smpl, ctx, -1) : amax;

        for (int step = 0; step < a.max_trace; ++step) {
            if (llama_vocab_is_eog(vocab, tok)) break;

            gen.push_back(tok);
            fprintf(flp, "%d\t%d\t%.5f\n", id, gen_count, logp);

            // append to seq 0
            if (!decode_tokens({ tok }, 0, L, true)) break;
            L += 1;
            gen_count += 1;

            // sample the NEXT trace token from seq 0's logits BEFORE probing,
            // since the probe decodes on seq 1 and would clobber these logits
            logits_stats(llama_get_logits_ith(ctx, -1), nv, amax, logp);
            tok = smpl ? llama_sampler_sample(smpl, ctx, -1) : amax;

            // checkpoints
            for (int N : a.checkpoints) {
                if (gen_count % N != 0) continue;
                // fork seq 0 -> seq 1
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
                // append cue on seq 1
                if (!decode_tokens(cue, 1, L, true)) { llama_memory_seq_rm(mem, 1, -1, -1); break; }
                llama_pos p = L + (llama_pos)cue.size();
                std::string ptext;
                for (int j = 0; j < a.probe_len; ++j) {
                    double lp = 0.0;
                    llama_token pt = smpl ? llama_sampler_sample(smpl, ctx, -1)
                                          : sample_greedy(llama_get_logits_ith(ctx, -1), nv, lp);
                    if (llama_vocab_is_eog(vocab, pt)) break;
                    char piece[64];
                    int np = llama_token_to_piece(vocab, pt, piece, sizeof(piece), 0, false);
                    if (np > 0) ptext.append(piece, np);
                    if (!decode_tokens({ pt }, 1, p, true)) break;
                    p += 1;
                }
                llama_memory_seq_rm(mem, 1, -1, -1);
                fprintf(fpr, "%d\t%d\t%d\t%s\n", id, N, gen_count, clean(ptext).c_str());
            }
        }

        for (llama_token t : gen) {
            char piece[64];
            int np = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
            if (np > 0) final_text.append(piece, np);
        }
        fprintf(ftr, "%d\t%s\t%d\t%s\n", id, gold.c_str(), gen_count, clean(final_text).c_str());
        printf("id=%d gold=%s trace=%d\n", id, gold.c_str(), gen_count);
        fflush(stdout); fflush(ftr); fflush(fpr); fflush(flp);
    }

    fclose(ftr); fclose(fpr); fclose(flp);
    if (smpl) llama_sampler_free(smpl);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
