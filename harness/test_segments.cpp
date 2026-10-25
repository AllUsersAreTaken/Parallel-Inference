// Row-segmented LoRA checks at the llama API level.
//
// Covers the cases that the runtime tests cannot reach without a mismatched
// segment list:
//   a) matching segments: per-row split and full ubatch must equal the global adapter
//   b) non-matching segment list: must decode with base weights (never the global list)
//   c) graph reuse: changing only the segment signature must rebuild the graph
//   d) readiness check: token_embd / output / MoE adapters are rejected
//
// Usage:
//   scripts\build_tool.bat build\cuda-graphs harness\test_segments.cpp harness\test_segments.exe
//   harness\test_segments.exe <model.gguf> <adapter_a.gguf> <adapter_b.gguf> [bad_adapter.gguf]
//
// bad_adapter can be produced with:
//   python harness\make_bad_segment_lora.py <model.gguf> data\_hf_lora\sql_korg.gguf results\_segtest\bad_tokembd.gguf
// Without it the readiness test is skipped.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { std::printf("  ok: "); } else { std::printf("  FAIL: "); ++failures; } \
    std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

static llama_model * g_model = nullptr;
static int g_n_vocab = 0;

struct RowLogits {
    std::vector<std::vector<float>> rows;
    bool ok = false;
};

static std::vector<llama_token> tokenize(const char * text) {
    const llama_vocab * voc = llama_model_get_vocab(g_model);
    int n = llama_tokenize(voc, text, (int) std::strlen(text), nullptr, 0, true, true);
    if (n < 0) n = -n;
    std::vector<llama_token> toks((size_t) n + 1);
    n = llama_tokenize(voc, text, (int) std::strlen(text), toks.data(), (int) toks.size(), true, true);
    toks.resize(n > 0 ? (size_t) n : 0);
    return toks;
}

static llama_context * new_ctx() {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.kv_unified = true;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    return llama_init_from_model(g_model, cp);
}

// Decode n_rows sequences that each hold one token at position 0. Returns the
// logits of that token for every row, so the ubatch has exactly n_rows rows.
static RowLogits decode_rows(llama_context * ctx, llama_token tok, int n_rows) {
    RowLogits out;
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch b = llama_batch_init(n_rows, 0, 1);
    for (int r = 0; r < n_rows; ++r) {
        b.token[r]     = tok;
        b.pos[r]       = 0;
        b.n_seq_id[r]  = 1;
        b.seq_id[r][0] = r;
        b.logits[r]    = 1;
    }
    b.n_tokens = n_rows;

    if (llama_decode(ctx, b) == 0) {
        llama_synchronize(ctx);
        out.ok = true;
        for (int r = 0; r < n_rows; ++r) {
            const float * l = llama_get_logits_ith(ctx, r);
            if (!l) { out.ok = false; break; }
            out.rows.emplace_back(l, l + g_n_vocab);
        }
    }
    llama_batch_free(b);
    return out;
}

// Prefill the whole token list on sequence 0 and return the logits of its last
// token. Every row requests logits: with a partial request the last-layer FFN
// is pruned to the output rows and no longer has one row per token, which a
// segment list cannot describe.
static RowLogits decode_ubatch(llama_context * ctx, const std::vector<llama_token> & toks) {
    RowLogits out;
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int) toks.size();

    if (llama_decode(ctx, b) == 0) {
        llama_synchronize(ctx);
        const float * l = llama_get_logits_ith(ctx, (int32_t) toks.size() - 1);
        if (l) { out.rows.emplace_back(l, l + g_n_vocab); out.ok = true; }
    }
    llama_batch_free(b);
    return out;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) m = d;
    }
    return m;
}

static RowLogits ubatch_global(llama_adapter_lora * ad, const std::vector<llama_token> & toks) {
    llama_context * ctx = new_ctx();
    float scale = 1.0f;
    llama_set_adapters_lora(ctx, &ad, 1, &scale);
    const RowLogits out = decode_ubatch(ctx, toks);
    llama_free(ctx);
    return out;
}

static RowLogits rows_global(llama_adapter_lora * ad, llama_token tok, int n_rows) {
    llama_context * ctx = new_ctx();
    float scale = 1.0f;
    llama_set_adapters_lora(ctx, &ad, 1, &scale);
    const RowLogits out = decode_rows(ctx, tok, n_rows);
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::printf("usage: %s <model.gguf> <adapter_a.gguf> <adapter_b.gguf> [bad_adapter.gguf]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.progress_callback = [](float, void *) { return true; };
    g_model = llama_model_load_from_file(argv[1], mp);
    if (!g_model) { std::printf("cannot load model %s\n", argv[1]); return 1; }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));

    llama_adapter_lora * a = llama_adapter_lora_init(g_model, argv[2]);
    llama_adapter_lora * b = llama_adapter_lora_init(g_model, argv[3]);
    if (!a || !b) { std::printf("cannot load adapters\n"); return 1; }
    llama_adapter_lora * bad = argc > 4 ? llama_adapter_lora_init(g_model, argv[4]) : nullptr;
    if (argc > 4 && !bad) { std::printf("cannot load bad adapter %s\n", argv[4]); return 1; }

    const std::vector<llama_token> toks = tokenize(
        "Schema:\nCREATE TABLE singer (ID INTEGER, Name TEXT, Country TEXT);\n\n"
        "Question: How many singers are from France?");
    const llama_token last = toks.empty() ? 0 : toks.back();
    const int n_rows = 2;

    RowLogits base_ub, base_rows;
    {
        llama_context * ctx = new_ctx();
        base_ub   = decode_ubatch(ctx, toks);
        base_rows = decode_rows(ctx, last, n_rows);
        llama_free(ctx);
    }
    CHECK(base_ub.ok && base_rows.ok, "base prefill and %d-row decode", n_rows);

    // (a) matching segments
    {
        // whole ubatch in one segment
        llama_context * ctx = new_ctx();
        float scale = 1.0f;
        int32_t n = (int32_t) toks.size();
        CHECK(llama_set_adapter_segments(ctx, &a, &scale, &n, 1) == 0, "full-ubatch segment list accepted");
        const RowLogits seg_ub = decode_ubatch(ctx, toks);
        llama_free(ctx);

        const RowLogits glob_ub = ubatch_global(a, toks);
        if (seg_ub.ok && glob_ub.ok) {
            const double d = max_abs_diff(seg_ub.rows[0], glob_ub.rows[0]);
            CHECK(d < 1e-3, "full ubatch: segments == global adapter (max|dlogit| %.2e)", d);
            CHECK(max_abs_diff(base_ub.rows[0], seg_ub.rows[0]) > 1e-2, "full ubatch: adapter changes the logits");
        } else {
            CHECK(false, "full ubatch decode failed");
        }

        // the same rows split into two segments must give the same logits
        llama_context * ctx2 = new_ctx();
        llama_adapter_lora * ad2[2] = { a, a };
        float sc2[2] = { 1.0f, 1.0f };
        int32_t nn2[2] = { (int32_t) toks.size() / 2, (int32_t) toks.size() - (int32_t) toks.size() / 2 };
        llama_set_adapter_segments(ctx2, ad2, sc2, nn2, 2);
        const RowLogits seg2_ub = decode_ubatch(ctx2, toks);
        llama_free(ctx2);
        if (seg2_ub.ok && seg_ub.ok) {
            const double d2 = max_abs_diff(seg2_ub.rows[0], seg_ub.rows[0]);
            CHECK(d2 < 1e-6, "one segment == two segments over the same ubatch (max|dlogit| %.2e)", d2);
        } else {
            CHECK(false, "split-segment ubatch decode failed");
        }
    }
    {
        // per-row split: one token per row, one segment per row
        llama_context * ctx = new_ctx();
        llama_adapter_lora * ad[2] = { a, a };
        float scale[2] = { 1.0f, 1.0f };
        int32_t n[2] = { 1, 1 };
        CHECK(llama_set_adapter_segments(ctx, ad, scale, n, 2) == 0, "per-row segment list accepted");
        const RowLogits seg = decode_rows(ctx, last, n_rows);
        llama_free(ctx);

        const RowLogits glob = rows_global(a, last, n_rows);
        if (seg.ok && glob.ok) {
            const double d0 = max_abs_diff(seg.rows[0], glob.rows[0]);
            const double d1 = max_abs_diff(seg.rows[1], glob.rows[1]);
            CHECK(d0 < 1e-3 && d1 < 1e-3, "per-row segments == global adapter (max|dlogit| %.2e / %.2e)", d0, d1);
            CHECK(max_abs_diff(base_rows.rows[0], seg.rows[0]) > 1e-2, "per-row segments change the logits");
        } else {
            CHECK(false, "per-row segmented decode failed");
        }
    }

    // (b) a list that does not cover the rows must use base weights, not merge
    {
        llama_context * ctx = new_ctx();
        float scale = 1.0f;
        int32_t n = (int32_t) toks.size() + 1;   // one row too many
        const int rc = llama_set_adapter_segments(ctx, &a, &scale, &n, 1);
        const RowLogits mism = decode_ubatch(ctx, toks);
        llama_free(ctx);

        CHECK(rc == 0, "mismatched segment list is accepted by the setter");
        if (mism.ok) {
            const double d = max_abs_diff(base_ub.rows[0], mism.rows[0]);
            CHECK(d < 1e-6, "non-matching segments decode with base weights (max|dlogit| %.2e)", d);
        } else {
            CHECK(false, "non-matching segment decode failed");
        }
    }

    // (c) same context, two decodes: only the segment signature changes
    {
        llama_context * ctx = new_ctx();
        llama_adapter_lora * seg2[2] = { a, b };
        float scales[2] = { 1.0f, 1.0f };
        int32_t counts[2] = { 1, 1 };
        llama_set_adapter_segments(ctx, seg2, scales, counts, 2);
        const RowLogits mixed = decode_rows(ctx, last, n_rows);

        float one_scale = 1.0f;
        int32_t one_n = n_rows;
        llama_set_adapter_segments(ctx, &a, &one_scale, &one_n, 1);
        const RowLogits replay = decode_rows(ctx, last, n_rows);
        llama_free(ctx);

        const RowLogits glob = rows_global(a, last, n_rows);
        if (mixed.ok && replay.ok && glob.ok) {
            const double d0 = max_abs_diff(glob.rows[0], replay.rows[0]);
            const double d1 = max_abs_diff(glob.rows[1], replay.rows[1]);
            CHECK(d0 < 1e-3 && d1 < 1e-3,
                  "graph is rebuilt when only the segment signature changes (max|dlogit| %.2e / %.2e)", d0, d1);
        } else {
            CHECK(false, "segment signature replay failed");
        }
    }

    // (d) adapters with token_embd / output / MoE weights are refused
    if (bad) {
        const llama_adapter_lora * one[1] = { bad };
        char msg[256] = { 0 };
        const int rc = llama_adapter_segments_check(one, 1, msg, sizeof(msg));
        CHECK(rc != 0, "readiness check rejects a token_embd adapter (%s)", msg);

        llama_context * ctx = new_ctx();
        float scale = 1.0f;
        int32_t n = (int32_t) toks.size();
        const int src = llama_set_adapter_segments(ctx, &bad, &scale, &n, 1);
        CHECK(src != 0, "setter refuses a token_embd adapter");
        llama_free(ctx);
    } else {
        std::printf("  skip: no bad adapter given\n");
    }

    llama_adapter_lora_free(a);
    llama_adapter_lora_free(b);
    if (bad) llama_adapter_lora_free(bad);
    llama_model_free(g_model);
    llama_backend_free();

    std::printf(failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
