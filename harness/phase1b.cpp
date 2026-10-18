// RecycleLM - Phase 1b knee harness
//
// Measures the cost of interleaving a C-token chunk for sequence 1 into the
// same llama_decode() batch as one decode token for sequence 0, at context
// depth D. Reports per-step time (A's inter-token latency), NVML energy,
// SM clock, power and temperature.
//
// Build: see build_harness.bat. Run against a specific CUDA build so the
// right ggml-cuda.dll (graphs on/off) is loaded.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS

#include "llama.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- NVML (dynamic)

typedef void * nvmlDevice_t;

static const int NVML_SUCCESS       = 0;
static const int NVML_CLOCK_SM      = 1;
static const int NVML_TEMPERATURE_GPU = 0;

struct nvml_api {
    HMODULE lib = nullptr;

    int (*init_v2)(void)                              = nullptr;
    int (*shutdown)(void)                             = nullptr;
    int (*get_handle)(unsigned int, nvmlDevice_t *)   = nullptr;
    int (*get_total_energy)(nvmlDevice_t, unsigned long long *) = nullptr;
    int (*get_power)(nvmlDevice_t, unsigned int *)    = nullptr;
    int (*get_clock)(nvmlDevice_t, int, unsigned int *) = nullptr;
    int (*get_temp)(nvmlDevice_t, int, unsigned int *)  = nullptr;

    nvmlDevice_t dev = nullptr;
    bool has_total_energy = false;
    bool power_ok = false;

    bool load(int index) {
        lib = LoadLibraryA("nvml.dll");
        if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        if (!lib) return false;

        init_v2          = (decltype(init_v2))         GetProcAddress(lib, "nvmlInit_v2");
        shutdown         = (decltype(shutdown))        GetProcAddress(lib, "nvmlShutdown");
        get_handle       = (decltype(get_handle))      GetProcAddress(lib, "nvmlDeviceGetHandleByIndex_v2");
        get_total_energy = (decltype(get_total_energy))GetProcAddress(lib, "nvmlDeviceGetTotalEnergyConsumption");
        get_power        = (decltype(get_power))       GetProcAddress(lib, "nvmlDeviceGetPowerUsage");
        get_clock        = (decltype(get_clock))       GetProcAddress(lib, "nvmlDeviceGetClockInfo");
        get_temp         = (decltype(get_temp))        GetProcAddress(lib, "nvmlDeviceGetTemperature");

        if (!init_v2 || !get_handle) return false;
        if (init_v2() != NVML_SUCCESS) return false;
        if (get_handle((unsigned)index, &dev) != NVML_SUCCESS) return false;

        if (get_total_energy) {
            unsigned long long e = 0;
            has_total_energy = (get_total_energy(dev, &e) == NVML_SUCCESS);
        }
        if (get_power) {
            unsigned p = 0;
            power_ok = (get_power(dev, &p) == NVML_SUCCESS);
        }
        return true;
    }

    unsigned long long total_energy_mj() {
        unsigned long long e = 0;
        if (get_total_energy && get_total_energy(dev, &e) == NVML_SUCCESS) return e;
        return 0;
    }
};

// ---------------------------------------------------------------- sampler thread

struct sampler {
    nvml_api * nv = nullptr;
    int interval_ms = 20;

    std::atomic<bool>     running{false};
    std::thread           th;

    std::atomic<double>   energy_j{0.0};
    std::atomic<double>   last_power_w{0.0};
    std::atomic<unsigned> last_clock{0};
    std::atomic<unsigned> last_temp{0};
    std::atomic<unsigned> min_clock{~0u};
    std::atomic<unsigned> max_clock{0};
    std::atomic<unsigned> min_temp{~0u};
    std::atomic<unsigned> max_temp{0};
    std::atomic<long long> n_samples{0};

    void start() {
        energy_j = 0.0;
        min_clock = ~0u; max_clock = 0;
        min_temp  = ~0u; max_temp  = 0;
        n_samples = 0;
        running = true;
        th = std::thread([this] { loop(); });
    }

    void stop() {
        running = false;
        if (th.joinable()) th.join();
    }

    void loop() {
        auto tprev = std::chrono::steady_clock::now();
        while (running.load()) {
            unsigned p = 0, c = 0, t = 0;
            if (nv->get_power && nv->get_power(nv->dev, &p) == NVML_SUCCESS) {
                const double pw = p / 1000.0;
                const auto now = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(now - tprev).count();
                tprev = now;
                energy_j = energy_j.load() + pw * dt;
                last_power_w = pw;
            }
            if (nv->get_clock && nv->get_clock(nv->dev, NVML_CLOCK_SM, &c) == NVML_SUCCESS) {
                last_clock = c;
                if (c < min_clock.load()) min_clock = c;
                if (c > max_clock.load()) max_clock = c;
            }
            if (nv->get_temp && nv->get_temp(nv->dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) {
                last_temp = t;
                if (t < min_temp.load()) min_temp = t;
                if (t > max_temp.load()) max_temp = t;
            }
            n_samples++;
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    }
};

// ---------------------------------------------------------------- helpers

using clk = std::chrono::steady_clock;

static int g_verbose = 0;

static double ms_since(const clk::time_point & t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    if (lo == hi) return v[lo];
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

struct item {
    llama_token  tok;
    llama_pos    pos;
    llama_seq_id seq;
    bool         logits;
};

// Submit items in n_batch-sized chunks. Returns true on success.
static bool submit(llama_context * ctx, llama_batch & batch, const std::vector<item> & items, int n_batch) {
    for (size_t off = 0; off < items.size(); off += n_batch) {
        const int n = (int)std::min<size_t>(n_batch, items.size() - off);
        for (int i = 0; i < n; ++i) {
            const item & it = items[off + i];
            batch.token[i]     = it.tok;
            batch.pos[i]       = it.pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = it.seq;
            batch.logits[i]    = it.logits ? 1 : 0;
        }
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) return false;
    }
    return true;
}

static llama_token argmax(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    float   bv   = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    int32_t uniform(int32_t n) { return (int32_t)(next() % (uint64_t)n); }
};

// ---------------------------------------------------------------- args

struct args {
    std::string model;
    std::string arm      = "shared";   // shared | separate
    std::string mode     = "fixed";    // fixed | vary
    int   depth          = 512;
    std::vector<int> cs  = {0, 1, 2, 4, 8, 16, 32, 64, 128};
    int   steps          = 100;
    int   warmup         = 10;
    int   idle_secs      = 30;
    int   ngl            = 99;
    int   fa             = 1;          // 1 on, 0 off
    int   recycle        = 1;
    int   b_logits       = 1;
    int   same_seq       = 0;          // 1 = put B's chunk in seq 0 too (pure batching diagnostic)
    int   kv_unified     = 0;
    int   episode        = 0;          // 1 = run the episode (workload) benchmark
    int   ep_a           = 200;        // A decode tokens
    int   ep_suffix      = 25;         // B suffix prefill tokens
    int   ep_bdecode     = 30;         // B decode tokens
    std::string episode_csv = "";
    int   check          = 0;          // if >0, run correctness check with this C
    int   gpu_index      = 0;
    int   verbose        = 0;
    std::string steps_csv = "";
    std::string runs_csv  = "";
    std::string label     = "";
};

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
            return argv[++i];
        };
        if      (k == "--model")   a.model = need("--model");
        else if (k == "--arm")     a.arm = need("--arm");
        else if (k == "--mode")    a.mode = need("--mode");
        else if (k == "--depth")   a.depth = atoi(need("--depth").c_str());
        else if (k == "--steps")   a.steps = atoi(need("--steps").c_str());
        else if (k == "--warmup")  a.warmup = atoi(need("--warmup").c_str());
        else if (k == "--idle-secs") a.idle_secs = atoi(need("--idle-secs").c_str());
        else if (k == "--ngl")     a.ngl = atoi(need("--ngl").c_str());
        else if (k == "--fa")      a.fa = atoi(need("--fa").c_str());
        else if (k == "--recycle") a.recycle = atoi(need("--recycle").c_str());
        else if (k == "--b-logits")a.b_logits = atoi(need("--b-logits").c_str());
        else if (k == "--same-seq")a.same_seq = atoi(need("--same-seq").c_str());
        else if (k == "--kv-unified") a.kv_unified = atoi(need("--kv-unified").c_str());
        else if (k == "--episode") a.episode = atoi(need("--episode").c_str());
        else if (k == "--episode-a") a.ep_a = atoi(need("--episode-a").c_str());
        else if (k == "--episode-suffix") a.ep_suffix = atoi(need("--episode-suffix").c_str());
        else if (k == "--episode-bdecode") a.ep_bdecode = atoi(need("--episode-bdecode").c_str());
        else if (k == "--episode-csv") a.episode_csv = need("--episode-csv");
        else if (k == "--check")   a.check = atoi(need("--check").c_str());
        else if (k == "--gpu")     a.gpu_index = atoi(need("--gpu").c_str());
        else if (k == "--verbose") a.verbose = 1;
        else if (k == "--steps-csv") a.steps_csv = need("--steps-csv");
        else if (k == "--runs-csv")  a.runs_csv  = need("--runs-csv");
        else if (k == "--label")   a.label = need("--label");
        else if (k == "--C") {
            a.cs.clear();
            std::string v = need("--C");
            size_t p = 0;
            while (p <= v.size()) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                a.cs.push_back(atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else {
            fprintf(stderr, "unknown arg: %s\n", k.c_str());
            return false;
        }
    }
    if (a.model.empty()) { fprintf(stderr, "--model required\n"); return false; }
    return true;
}

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) return 2;

    if (a.fa != 0 && a.fa != 1) { fprintf(stderr, "--fa must be 0 or 1\n"); return 2; }

    int maxC = 0;
    for (int c : a.cs) maxC = std::max(maxC, c);

    g_verbose = a.verbose;
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    mp.progress_callback = [](float, void *) { return true; };

    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // KV budget: non-unified -> n_ctx is split across n_seq_max sequences.
    const int per_seq = a.depth
        + (a.episode
               ? std::max(a.ep_a, a.ep_suffix + a.ep_bdecode)
               : (a.same_seq
                      ? (a.warmup + a.steps) * (1 + std::max(1, maxC))
                      : (a.recycle ? (a.warmup + a.steps + maxC) : (a.warmup + a.steps) * std::max(1, maxC))))
        + 64;
    uint32_t n_ctx = (uint32_t)(2 * per_seq);
    n_ctx = (n_ctx + 255u) & ~255u;

    const int n_batch  = 2048;
    const int n_ubatch = 512;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = 2;
    cp.flash_attn_type = a.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.kv_unified      = a.kv_unified != 0;
    cp.n_threads       = 8;
    cp.n_threads_batch = 8;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); llama_model_free(model); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // deterministic prompts
    rng r0(1234), r1(5678);
    std::vector<item> prefill0; prefill0.reserve(a.depth);
    for (int i = 0; i < a.depth; ++i)
        prefill0.push_back({ r0.uniform(n_vocab), i, 0, i == a.depth - 1 });

    std::vector<item> prefill1; prefill1.reserve(a.depth);
    for (int i = 0; i < a.depth; ++i)
        prefill1.push_back({ r1.uniform(n_vocab), i, 1, false });

    // per-step chunk tokens (seq 1); regenerate per step from a stream
    rng rc(90125);

    // ---- correctness check ----
    if (a.check > 0) {
        auto run_seq = [&](int C, std::vector<llama_token> & out) -> bool {
            llama_memory_clear(mem, true);
            rng rc2(90125);
            if (!submit(ctx, batch, prefill0, n_batch)) return false;
            llama_synchronize(ctx);
            llama_token cur = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            if (a.arm == "shared") {
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
            } else {
                if (!submit(ctx, batch, prefill1, n_batch)) return false;
                llama_synchronize(ctx);
            }
            out.clear();
            for (int s = 0; s < a.steps; ++s) {
                const llama_pos p0 = a.depth + s;
                const llama_pos p1 = a.recycle ? a.depth : a.depth + s * C;
                std::vector<item> its;
                its.push_back({ cur, p0, 0, true });
                if (C > 0) {
                    if (a.recycle) llama_memory_seq_rm(mem, 1, a.depth, a.depth + C);
                    for (int k = 0; k < C; ++k)
                        its.push_back({ rc2.uniform(n_vocab), p1 + k, 1, (a.b_logits && k == C - 1) });
                }
                if (!submit(ctx, batch, its, n_batch)) return false;
                llama_synchronize(ctx);
                out.push_back(cur);
                cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
            }
            return true;
        };

        std::vector<llama_token> base, test;
        if (!run_seq(0, base))  { fprintf(stderr, "check base run failed\n"); return 1; }
        if (!run_seq(a.check, test)) { fprintf(stderr, "check test run failed\n"); return 1; }
        int first_diff = -1;
        for (size_t i = 0; i < base.size() && i < test.size(); ++i)
            if (base[i] != test[i]) { first_diff = (int)i; break; }
        printf("check arm=%s D=%d C=%d steps=%d: %s (first_diff=%d, n_base=%zu, n_test=%zu)\n",
               a.arm.c_str(), a.depth, a.check, a.steps,
               first_diff < 0 ? "PASS (A identical)" : "FAIL", first_diff, base.size(), test.size());
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return first_diff < 0 ? 0 : 3;
    }

    // ---- NVML ----
    nvml_api nv;
    sampler samp;
    const bool nv_ok = nv.load(a.gpu_index);
    if (nv_ok) {
        samp.nv = &nv;
        samp.start();
    }
    const bool energy_ok = nv_ok && (nv.has_total_energy || nv.power_ok);
    const char * method = !energy_ok ? (nv_ok ? "unavailable" : "none")
                                     : (nv.has_total_energy ? "total_energy" : "power_integrate");
    printf("nvml=%s method=%s n_ctx=%u arm=%s mode=%s D=%d recycle=%d fa=%d\n",
           nv_ok ? "ok" : "unavailable", method, llama_n_ctx(ctx), a.arm.c_str(), a.mode.c_str(),
           a.depth, a.recycle, a.fa);
    if (nv_ok && !energy_ok)
        printf("WARNING: GPU power/energy telemetry unavailable on this device; energy fields will be nan\n");

    // ---- idle baseline ----
    double idle_w = 0.0;
    if (energy_ok) {
        const double e0 = samp.energy_j.load();
        const auto   t0 = clk::now();
        std::this_thread::sleep_for(std::chrono::seconds(a.idle_secs));
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        idle_w = (samp.energy_j.load() - e0) / dt;
        printf("idle: %.1f W over %.1f s\n", idle_w, dt);
    }

    FILE * fsteps = a.steps_csv.empty() ? nullptr : fopen(a.steps_csv.c_str(), "a");
    FILE * fruns  = a.runs_csv.empty()  ? nullptr : fopen(a.runs_csv.c_str(),  "a");
    if (fsteps && ftell(fsteps) == 0)
        fprintf(fsteps, "label,arm,mode,kv_unified,depth,C,step,step_ms,sm_clock,power_w,temp_c\n");
    if (fruns && ftell(fruns) == 0)
        fprintf(fruns, "label,arm,mode,kv_unified,depth,C,steps,method,idle_w,run_s,energy_j,consol_tokens,j_per_tok,"
                       "p50_ms,p95_ms,p99_ms,min_clock,max_clock,min_temp,max_temp\n");

    auto schedule = [&](int s) -> int {
        if (a.mode == "vary") return a.cs[s % a.cs.size()];
        return a.cs[0];
    };

    // ---- episode (workload) mode ----
    if (a.episode) {
        const int C = a.cs.empty() ? 8 : a.cs[0];
        const int n_chunks = (a.ep_suffix + C - 1) / C;

        rng rs(4242);
        std::vector<llama_token> suffix(a.ep_suffix);
        for (int i = 0; i < a.ep_suffix; ++i) suffix[i] = rs.uniform(n_vocab);

        auto run_scenario = [&](bool piggyback, double & a_total, double & a_p99, double & b_done) -> bool {
            llama_memory_clear(mem, true);
            if (!submit(ctx, batch, prefill0, n_batch)) return false;
            llama_synchronize(ctx);
            llama_token cur = argmax(llama_get_logits_ith(ctx, -1), n_vocab);

            if (a.arm == "shared") {
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
            } else {
                if (!submit(ctx, batch, prefill1, n_batch)) return false;
                llama_synchronize(ctx);
            }
            llama_pos p1 = a.depth;

            llama_token b_cur = suffix.empty() ? 0 : suffix.back();
            std::vector<double> lat; lat.reserve(a.ep_a);
            int b_last_step = -1;

            const auto t0 = clk::now();
            for (int s = 0; s < a.ep_a; ++s) {
                std::vector<item> its;
                its.push_back({ cur, a.depth + s, 0, true });
                bool b_active = false;
                int  b_idx = -1;
                if (piggyback) {
                    if (s < n_chunks) {
                        const int cnt = std::min(C, a.ep_suffix - s * C);
                        for (int k = 0; k < cnt; ++k)
                            its.push_back({ suffix[s * C + k], p1 + k, 1, (k == cnt - 1) });
                        p1 += cnt;
                        b_idx = (int)its.size() - 1;
                        b_active = true;
                    } else if (s < n_chunks + a.ep_bdecode) {
                        its.push_back({ b_cur, p1, 1, true });
                        p1 += 1;
                        b_idx = (int)its.size() - 1;
                        b_active = true;
                    }
                }
                llama_synchronize(ctx);
                const auto ts = clk::now();
                if (!submit(ctx, batch, its, n_batch)) return false;
                llama_synchronize(ctx);
                lat.push_back(ms_since(ts));
                cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
                if (b_active) {
                    b_cur = argmax(llama_get_logits_ith(ctx, b_idx), n_vocab);
                    b_last_step = s;
                }
            }
            a_total = ms_since(t0);
            a_p99   = percentile(lat, 99);

            if (piggyback) {
                double sum = 0;
                for (int i = 0; i <= b_last_step && i < (int)lat.size(); ++i) sum += lat[i];
                b_done = sum;
            } else {
                const auto tb = clk::now();
                for (int s = 0; s < n_chunks; ++s) {
                    const int cnt = std::min(C, a.ep_suffix - s * C);
                    std::vector<item> its;
                    for (int k = 0; k < cnt; ++k)
                        its.push_back({ suffix[s * C + k], p1 + k, 1, (k == cnt - 1) });
                    p1 += cnt;
                    if (!submit(ctx, batch, its, n_batch)) return false;
                    llama_synchronize(ctx);
                    b_cur = argmax(llama_get_logits_ith(ctx, (int)its.size() - 1), n_vocab);
                }
                for (int d = 0; d < a.ep_bdecode; ++d) {
                    std::vector<item> its;
                    its.push_back({ b_cur, p1, 1, true });
                    p1 += 1;
                    if (!submit(ctx, batch, its, n_batch)) return false;
                    llama_synchronize(ctx);
                    b_cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
                }
                b_done = ms_since(tb);
            }
            return true;
        };

        double a_pig = 0, a_pig_p99 = 0, b_pig = 0, a_alone = 0, a_alone_p99 = 0, b_alone = 0;
        double w = 0, wp = 0, wb = 0;
        // warm both scenarios first so the measured pair sees the same thermal state
        if (!run_scenario(true,  w, wp, wb)) { fprintf(stderr, "episode warmup failed\n"); return 1; }
        if (!run_scenario(false, w, wp, wb)) { fprintf(stderr, "episode warmup failed\n"); return 1; }
        if (!run_scenario(true,  a_pig,   a_pig_p99,   b_pig))   { fprintf(stderr, "episode piggyback failed\n"); return 1; }
        if (!run_scenario(false, a_alone, a_alone_p99, b_alone)) { fprintf(stderr, "episode standalone failed\n"); return 1; }

        const double added     = a_pig - a_alone;
        const double added_pct = a_alone > 0 ? 100.0 * added / a_alone : 0.0;
        const double total_alone = a_alone + b_alone;
        const double total_pig   = a_pig; // B finishes inside A's window
        const double savings_pct = total_alone > 0 ? 100.0 * (total_alone - total_pig) / total_alone : 0.0;

        printf("episode arm=%s kv_unified=%d D=%d C=%d A=%d suffix=%d bdecode=%d | "
               "A_alone=%.1f A_pig=%.1f added=%.1f ms (%.2f%%) | p99_pig=%.3f ms | "
               "B_pig=%.1f B_alone=%.1f | total_alone=%.1f total_pig=%.1f savings=%.2f%% | clock %u-%u MHz\n",
               a.arm.c_str(), a.kv_unified, a.depth, C, a.ep_a, a.ep_suffix, a.ep_bdecode,
               a_alone, a_pig, added, added_pct, a_pig_p99, b_pig, b_alone,
               total_alone, total_pig, savings_pct, samp.min_clock.load(), samp.max_clock.load());

        if (!a.episode_csv.empty()) {
            FILE * fe = fopen(a.episode_csv.c_str(), "a");
            if (fe) {
                if (ftell(fe) == 0)
                    fprintf(fe, "label,arm,kv_unified,depth,C,a_tokens,suffix,b_decode,"
                                "A_alone_ms,A_pig_ms,added_ms,added_pct,p99_pig_ms,B_pig_ms,B_alone_ms,"
                                "total_alone_ms,total_pig_ms,savings_pct\n");
                fprintf(fe, "%s,%s,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f\n",
                        a.label.c_str(), a.arm.c_str(), a.kv_unified, a.depth, C, a.ep_a, a.ep_suffix, a.ep_bdecode,
                        a_alone, a_pig, added, added_pct, a_pig_p99, b_pig, b_alone,
                        total_alone, total_pig, savings_pct);
                fclose(fe);
            }
        }

        if (fsteps) fclose(fsteps);
        if (fruns)  fclose(fruns);
        if (nv_ok) { samp.stop(); if (nv.shutdown) nv.shutdown(); }
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // ---- prefill + warmup + measure per C ----
    const int n_cells = (a.mode == "vary") ? 1 : (int)a.cs.size();
    for (int ci = 0; ci < n_cells; ++ci) {
        const int C = (a.mode == "vary") ? maxC : a.cs[ci];

        llama_memory_clear(mem, true);
        rng rcx(90125);

        if (!submit(ctx, batch, prefill0, n_batch)) { fprintf(stderr, "prefill0 failed\n"); return 1; }
        llama_synchronize(ctx);
        llama_token cur = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        if (!a.same_seq) {
            if (a.arm == "shared") {
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
            } else {
                if (!submit(ctx, batch, prefill1, n_batch)) { fprintf(stderr, "prefill1 failed\n"); return 1; }
                llama_synchronize(ctx);
            }
        }

        const int total_steps = a.warmup + a.steps;
        std::vector<double> times;
        times.reserve(a.steps);

        const unsigned long long e_before = (energy_ok && nv.has_total_energy) ? nv.total_energy_mj() : 0;
        const double e_int_before = energy_ok ? samp.energy_j.load() : 0.0;
        const auto   t_run0 = clk::now();

        long long consol_tokens = 0;
        llama_pos p1     = a.depth; // next free position for sequence 1 (non-recycle mode)
        llama_pos p0_run = a.depth; // running position for seq 0 (same-seq mode)

        for (int s = 0; s < total_steps; ++s) {
            const bool measured = s >= a.warmup;
            const int  cc = (a.mode == "vary") ? schedule(s) : C;
            const llama_pos p0 = a.same_seq ? p0_run : (a.depth + s);

            std::vector<item> its;
            its.push_back({ cur, p0, 0, true });
            if (a.same_seq) {
                for (int k = 0; k < cc; ++k)
                    its.push_back({ rcx.uniform(n_vocab), p0 + 1 + k, 0, (a.b_logits && k == cc - 1) });
                p0_run = p0 + 1 + cc;
            } else if (cc > 0) {
                if (a.recycle) {
                    llama_memory_seq_rm(mem, 1, a.depth, a.depth + std::max(1, maxC));
                }
                for (int k = 0; k < cc; ++k)
                    its.push_back({ rcx.uniform(n_vocab), p1 + k, 1, (a.b_logits && k == cc - 1) });
                if (!a.recycle) p1 += cc;
            }

            llama_synchronize(ctx); // flush any seq_rm work before timing
            const auto t0 = clk::now();
            if (!submit(ctx, batch, its, n_batch)) { fprintf(stderr, "step decode failed\n"); return 1; }
            llama_synchronize(ctx);
            const double dt = ms_since(t0);
            if (measured) {
                times.push_back(dt);
                consol_tokens += cc;
                if (fsteps) {
                    fprintf(fsteps, "%s,%s,%s,%d,%d,%d,%d,%.4f,%u,%.2f,%u\n",
                            a.label.c_str(), a.arm.c_str(), a.mode.c_str(), a.kv_unified, a.depth, cc, s - a.warmup,
                            dt, samp.last_clock.load(), samp.last_power_w.load(), samp.last_temp.load());
                }
            }

            cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        }

        const double run_s = std::chrono::duration<double>(clk::now() - t_run0).count();

        double energy_j = std::nan("");
        if (energy_ok) {
            if (nv.has_total_energy) {
                const unsigned long long e_after = nv.total_energy_mj();
                energy_j = (double)(e_after - e_before) / 1000.0 - idle_w * run_s;
            } else {
                energy_j = (samp.energy_j.load() - e_int_before) - idle_w * run_s;
            }
            if (energy_j < 0.0) energy_j = 0.0;
        }

        const double p50 = percentile(times, 50);
        const double p95 = percentile(times, 95);
        const double p99 = percentile(times, 99);
        const double jpt = (energy_ok && consol_tokens > 0) ? energy_j / (double)consol_tokens : std::nan("");

        printf("arm=%s mode=%s D=%d C=%d | p50=%.3f p95=%.3f p99=%.3f ms | E=%.2f J | consol=%lld | J/tok=%.5f\n",
               a.arm.c_str(), a.mode.c_str(), a.depth, C, p50, p95, p99, energy_j, consol_tokens, jpt);

        if (fruns) {
            fprintf(fruns, "%s,%s,%s,%d,%d,%d,%d,%s,%.1f,%.3f,%.3f,%lld,%.6f,%.4f,%.4f,%.4f,%u,%u,%u,%u\n",
                    a.label.c_str(), a.arm.c_str(), a.mode.c_str(), a.kv_unified, a.depth, C, a.steps, method,
                    idle_w, run_s, energy_j, consol_tokens, jpt, p50, p95, p99,
                    samp.min_clock.load(), samp.max_clock.load(),
                    samp.min_temp.load(), samp.max_temp.load());
            fflush(fruns);
        }
        if (fsteps) fflush(fsteps);
    }

    if (fsteps) fclose(fsteps);
    if (fruns)  fclose(fruns);
    if (nv_ok) { samp.stop(); if (nv.shutdown) nv.shutdown(); }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
