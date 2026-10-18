// RecycleLM - E0 cost curve: N parallel streams sharing a prompt, unique tails.
//
// Prefills a shared prompt, forks N sequences with seq_cp, then decodes one
// token per sequence per step (batch = N) up to --max-tail, timing each step.
// Reports step time vs tail depth for each N, so unshared-tail KV cost shows up
// before any accuracy runs.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "llama.h"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- minimal NVML clock reader ----
typedef void * nvmlDevice_t;
struct nvml_api {
    HMODULE lib = nullptr;
    int (*init_v2)(void) = nullptr;
    int (*get_handle)(unsigned, nvmlDevice_t *) = nullptr;
    int (*get_clock)(nvmlDevice_t, int, unsigned *) = nullptr;
    nvmlDevice_t dev = nullptr;
    bool ok = false;
    bool load(int idx) {
        lib = LoadLibraryA("nvml.dll");
        if (!lib) return false;
        init_v2    = (decltype(init_v2))   GetProcAddress(lib, "nvmlInit_v2");
        get_handle = (decltype(get_handle))GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2");
        get_clock  = (decltype(get_clock)) GetProcAddress(lib, "nvmlDeviceGetClockInfo");
        if (!init_v2 || !get_handle) return false;
        if (init_v2() != 0) return false;
        if (get_handle((unsigned)idx, &dev) != 0) return false;
        ok = true;
        return true;
    }
    unsigned clock_sm() {
        unsigned c = 0;
        if (ok && get_clock && get_clock(dev, 1 /*NVML_CLOCK_SM*/, &c) == 0) return c;
        return 0;
    }
};

using clk = std::chrono::steady_clock;

struct args {
    std::string model;
    std::string out;
    int prompt_len = 512;
    int max_tail   = 2048;
    std::vector<int> Ns = {1, 2, 4, 8, 16};
    int fa = 1;
    int kv_unified = 1;
    int ngl = 99;
    int warmup = 10;
    uint32_t seed = 1234;
    std::string type_k = "f16";
    std::string type_v = "f16";
    int retire_at = -1;   // tail depth at which to retire half the streams (-1 = never)
    int retire_rm = 1;    // 1 = call seq_rm on retired streams, 0 = just drop from batch
    int pad_to = -1;      // top the batch up to this many tokens with filler streams (-1 = off)
};

static ggml_type parse_type(const std::string & s) {
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    fprintf(stderr, "unknown cache type %s\n", s.c_str());
    exit(2);
}

static bool parse(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&]() { if (i + 1 >= argc) { fprintf(stderr, "need value %s\n", k.c_str()); exit(2); } return std::string(argv[++i]); };
        if      (k == "--model") a.model = need();
        else if (k == "--out")   a.out = need();
        else if (k == "--prompt-len") a.prompt_len = atoi(need().c_str());
        else if (k == "--max-tail") a.max_tail = atoi(need().c_str());
        else if (k == "--fa") a.fa = atoi(need().c_str());
        else if (k == "--kv-unified") a.kv_unified = atoi(need().c_str());
        else if (k == "--ngl") a.ngl = atoi(need().c_str());
        else if (k == "--warmup") a.warmup = atoi(need().c_str());
        else if (k == "--seed") a.seed = (uint32_t)strtoul(need().c_str(), nullptr, 10);
        else if (k == "--type-k") a.type_k = need();
        else if (k == "--type-v") a.type_v = need();
        else if (k == "--retire-at") a.retire_at = atoi(need().c_str());
        else if (k == "--retire-rm") a.retire_rm = atoi(need().c_str());
        else if (k == "--pad-to") a.pad_to = atoi(need().c_str());
        else if (k == "--Ns") {
            a.Ns.clear();
            std::string v = need();
            size_t p = 0;
            while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); a.Ns.push_back(atoi(v.substr(p, q - p).c_str())); p = q + 1; }
        } else { fprintf(stderr, "unknown %s\n", k.c_str()); return false; }
    }
    return !a.model.empty();
}

int main(int argc, char ** argv) {
    args a;
    if (!parse(argc, argv, a)) return 2;

    int maxN = 1;
    for (int n : a.Ns) maxN = std::max(maxN, n);
    if (a.pad_to > maxN) maxN = a.pad_to;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    mp.progress_callback = [](float, void *) { return true; };

    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t nv = llama_vocab_n_tokens(vocab);

    // unified KV: n_ctx is the total across sequences
    uint32_t n_ctx = (uint32_t)(maxN * (a.prompt_len + a.max_tail) + 256);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = maxN;
    cp.flash_attn_type = a.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.kv_unified = a.kv_unified != 0;
    cp.type_k = parse_type(a.type_k);
    cp.type_v = parse_type(a.type_v);
    if (cp.type_v != GGML_TYPE_F16 && cp.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        fprintf(stderr, "note: quantized V cache requires flash attention; enabling fa\n");
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    cp.n_threads = 8;
    cp.n_threads_batch = 8;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(2048, 0, 1);

    nvml_api nvml;
    nvml.load(0);

    printf("n_ctx=%u maxN=%d kv_unified=%d fa=%d prompt=%d max_tail=%d n_layer=%d n_head_kv=%d type_k=%s type_v=%s\n",
           llama_n_ctx(ctx), maxN, a.kv_unified, a.fa, a.prompt_len, a.max_tail,
           llama_model_n_layer(model), llama_model_n_head_kv(model), a.type_k.c_str(), a.type_v.c_str());

    FILE * f = a.out.empty() ? nullptr : fopen(a.out.c_str(), "w");
    if (f) fprintf(f, "N,tail_depth,step_ms,clock_mhz,active,phase\n");

    auto decode_one = [&](const std::vector<int> & active, llama_pos pos, uint32_t step_seed) -> double {
        for (int i = 0; i < (int)active.size(); ++i) {
            int j = active[i];
            batch.token[i]     = (llama_token)((step_seed * 2654435761u + j * 40503u) % (uint32_t)nv);
            batch.pos[i]       = pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = j;
            batch.logits[i]    = 1;
        }
        batch.n_tokens = (int)active.size();
        const auto t0 = clk::now();
        if (llama_decode(ctx, batch) != 0) return -1.0;
        llama_synchronize(ctx);
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    for (int N : a.Ns) {
        llama_memory_clear(mem, true);
        // shared prompt on seq 0
        for (int off = 0; off < a.prompt_len; off += cp.n_batch) {
            int n = std::min((int)cp.n_batch, a.prompt_len - off);
            for (int i = 0; i < n; ++i) {
                batch.token[i] = (llama_token)((a.seed + off + i) % (uint32_t)nv);
                batch.pos[i] = off + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        llama_synchronize(ctx);
        for (int j = 1; j < std::max(N, a.pad_to); ++j) llama_memory_seq_cp(mem, 0, j, -1, -1);

        // decode the unique tail one token per sequence per step
        std::vector<int> active(N);
        for (int j = 0; j < N; ++j) active[j] = j;
        for (int t = 0; t < a.max_tail; ++t) {
            if (a.retire_at >= 0 && t == a.retire_at) {
                for (int k = N / 2; k < N; ++k) if (a.retire_rm) llama_memory_seq_rm(mem, k, -1, -1);
                active.resize(N / 2);
            }
            std::vector<int> all = active;
            if (a.pad_to > (int)active.size()) for (int j = N; j < a.pad_to; ++j) all.push_back(j);
            double dt = decode_one(all, a.prompt_len + t, a.seed + (uint32_t)N * 131 + (uint32_t)t);
            if (dt < 0) { fprintf(stderr, "decode failed N=%d t=%d\n", N, t); break; }
            int phase = (a.retire_at >= 0 && t >= a.retire_at) ? 1 : 0;
            if (f) fprintf(f, "%d,%d,%.4f,%u,%d,%d\n", N, t, dt, nvml.clock_sm(), (int)all.size(), phase);
        }
        fflush(f);
        printf("done N=%d\n", N);
    }

    if (f) fclose(f);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
