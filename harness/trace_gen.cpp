// RecycleLM - parallel trace generator for the wall-clock simulator.
//
// Per problem: prefill the prompt once, seq_cp to N streams (shared prompt),
// then decode one token per active stream per step. Each stream has its own
// sampler (temp 0.6, distinct seed). A stream retires when it emits a complete
// answer; the batch shrinks as streams finish. Records finish length + text.
//
// Output TSV: id  stream  finish_len  full_len  text

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct args {
    std::string model, data, out;
    int limit = 200;
    int streams = 16;
    int max_trace = 1024;
    int n_ctx = 4096;
    int ngl = 99;
    float temp = 0.6f;
    uint32_t seed = 1234;
};

static bool parse(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() { if (i + 1 >= argc) { fprintf(stderr, "need %s\n", k.c_str()); exit(2); } return std::string(argv[++i]); };
        if      (k == "--model") a.model = need();
        else if (k == "--data") a.data = need();
        else if (k == "--out") a.out = need();
        else if (k == "--limit") a.limit = atoi(need().c_str());
        else if (k == "--streams") a.streams = atoi(need().c_str());
        else if (k == "--max-trace") a.max_trace = atoi(need().c_str());
        else if (k == "--n-ctx") a.n_ctx = atoi(need().c_str());
        else if (k == "--ngl") a.ngl = atoi(need().c_str());
        else if (k == "--temp") a.temp = (float)atof(need().c_str());
        else if (k == "--seed") a.seed = (uint32_t)strtoul(need().c_str(), nullptr, 10);
        else { fprintf(stderr, "unknown %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.data.empty();
}

static std::string clean(std::string s) {
    for (char & c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return s;
}

// true when the text contains a completed answer after the think block
static bool answer_complete(const std::string & t) {
    static const std::string TE = "<" + std::string("/") + "think" + ">";
    if (t.find(TE) == std::string::npos) return false;
    size_t b = t.find("\\boxed{", t.find(TE));
    if (b == std::string::npos) return false;
    size_t e = t.find('}', b);
    return e != std::string::npos;
}

int main(int argc, char ** argv) {
    args a;
    if (!parse(argc, argv, a)) return 2;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    mp.progress_callback = [](float, void *) { return true; };
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * tmpl = llama_model_chat_template(model, nullptr);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = a.n_ctx; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_seq_max = a.streams;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified = true;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(2048, 0, 1);

    // per-stream sampler chains
    std::vector<llama_sampler *> smpls;
    for (int j = 0; j < a.streams; ++j) {
        auto sp = llama_sampler_chain_default_params();
        auto * c = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(c, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(c, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(c, llama_sampler_init_temp(a.temp));
        llama_sampler_chain_add(c, llama_sampler_init_dist(a.seed + 101u * (uint32_t)j));
        smpls.push_back(c);
    }

    FILE * f = fopen((a.out + ".tsv").c_str(), "w");
    if (!f) { fprintf(stderr, "cannot open out\n"); return 1; }

    std::ifstream fin(a.data);
    std::string line;
    int id = 0;
    while (std::getline(fin, line) && id < a.limit) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        ++id;

        llama_chat_message msg = { "user", q.c_str() };
        std::vector<char> buf(std::max<size_t>(1024, q.size() * 2 + 512));
        int plen = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int)buf.size());
        if (plen > (int)buf.size()) { buf.resize(plen); plen = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), (int)buf.size()); }
        std::vector<llama_token> ptoks(buf.size() + 16);
        int pn = llama_tokenize(vocab, buf.data(), plen, ptoks.data(), (int)ptoks.size(), false, true);
        if (pn < 0) { ptoks.resize(-pn); pn = llama_tokenize(vocab, buf.data(), plen, ptoks.data(), (int)ptoks.size(), false, true); }
        ptoks.resize(pn);

        llama_memory_clear(mem, true);
        for (int off = 0; off < (int)ptoks.size(); off += cp.n_batch) {
            int n = std::min((int)cp.n_batch, (int)ptoks.size() - off);
            for (int i = 0; i < n; ++i) {
                batch.token[i] = ptoks[off + i];
                batch.pos[i] = off + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (off + i + 1 == (int)ptoks.size()) ? 1 : 0;
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        for (int j = 1; j < a.streams; ++j) llama_memory_seq_cp(mem, 0, j, -1, -1);

        // common first token (usually the same for every stream)
        const float * lg = llama_get_logits_ith(ctx, -1);
        int nv = llama_vocab_n_tokens(vocab);
        int first = 0; float bv = lg[0];
        for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; first = i; }

        llama_pos pos = (llama_pos)ptoks.size();
        // seed all streams with the first token
        for (int j = 0; j < a.streams; ++j) {
            batch.token[j] = first; batch.pos[j] = pos; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = j; batch.logits[j] = 1;
        }
        batch.n_tokens = a.streams;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "first decode failed\n"); return 1; }
        pos += 1;

        std::vector<std::string> text(a.streams);
        std::vector<int> fin(a.streams, -1), ngen(a.streams, 1);
        std::vector<llama_token> nxt(a.streams);
        for (int j = 0; j < a.streams; ++j) nxt[j] = llama_sampler_sample(smpls[j], ctx, j);

        std::vector<int> active;
        for (int j = 0; j < a.streams; ++j) active.push_back(j);

        for (int step = 0; step < a.max_trace && !active.empty(); ++step) {
            // decode one token per active stream
            for (int i = 0; i < (int)active.size(); ++i) {
                int j = active[i];
                batch.token[i] = nxt[j]; batch.pos[i] = pos; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = j; batch.logits[i] = 1;
            }
            batch.n_tokens = (int)active.size();
            if (llama_decode(ctx, batch) != 0) break;
            pos += 1;

            std::vector<int> still;
            for (int i = 0; i < (int)active.size(); ++i) {
                int j = active[i];
                // append the just-decoded token's piece to text
                char piece[64];
                int np = llama_token_to_piece(vocab, nxt[j], piece, sizeof(piece), 0, false);
                if (np > 0) text[j].append(piece, np);
                ngen[j] += 1;

                if (llama_vocab_is_eog(vocab, nxt[j]) || answer_complete(text[j])) {
                    fin[j] = ngen[j];
                    continue;
                }
                nxt[j] = llama_sampler_sample(smpls[j], ctx, i);
                still.push_back(j);
            }
            active.swap(still);
        }
        for (int j : active) fin[j] = ngen[j]; // hit cap

        for (int j = 0; j < a.streams; ++j) {
            fprintf(f, "%d\t%d\t%d\t%d\t%s\n", id, j, fin[j], ngen[j], clean(text[j]).c_str());
        }
        printf("id=%d done\n", id);
        fflush(stdout); fflush(f);
    }

    fclose(f);
    for (auto * s : smpls) llama_sampler_free(s);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
