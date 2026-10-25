// parallel-inference public types.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace parallel {

enum class Role { Foreground, Auxiliary };

enum class FinishReason { Eos, TokenLimit, Agreement, Cancelled, Error };

inline const char * to_string(FinishReason r) {
    switch (r) {
        case FinishReason::Eos:        return "eos";
        case FinishReason::TokenLimit: return "token_limit";
        case FinishReason::Agreement:  return "agreement";
        case FinishReason::Cancelled:  return "cancelled";
        default:                       return "error";
    }
}

struct Sampling {
    float    temperature = 0.6f;
    int      top_k       = 20;
    float    top_p       = 0.95f;
    uint32_t seed        = 1234;
};

struct Branch {
    std::string id;
    std::string suffix;                  // appended after the shared prefix
    int         max_new_tokens = 256;    // relative to this branch's generation
    Role        role           = Role::Auxiliary;
    Sampling    sampling;
    std::string grammar;                 // optional GBNF, applied from the first token
    bool        record_probs   = false;  // capture raw-logit stats per emitted token
    bool        record_hidden  = false;  // capture last-layer embedding norm per step
    int         adapter_id     = -1;     // index into RuntimeConfig::lora_paths (-1 = none)
};

// One emitted token's raw-distribution record (record_probs only).
// All stats are computed from the RAW logits (temperature 1, before
// top-k/top-p/temperature sampling) over the FULL vocabulary -- they are
// not truncated-distribution estimates.
struct StepRec {
    int   step      = 0;     // 0-based emitted-token index in this branch
    int   pos       = 0;     // absolute context position the token occupies
    int   token     = 0;     // emitted token id
    float entropy   = 0.0f;  // full-vocab entropy of raw softmax (nats)
    float p1        = 0.0f;  // full-vocab top-1 probability (raw)
    float margin    = 0.0f;  // p1 - p2 (full-vocab, raw)
    float logp      = 0.0f;  // log-probability of the emitted token (raw)
    int   text_off  = 0;     // byte offset of the emitted piece in BranchResult::text
    int   text_len  = 0;     // byte length of the emitted piece
};

// One step's last-layer embedding summary (record_hidden only). The raw
// vector is scratch state inside the engine and is not returned.
struct HRec {
    int   step     = 0;      // emitted-token count after this step (1-based)
    float norm     = 0.0f;   // L2 norm of the last-layer embedding
    float cos_mean = -1.0f;  // mean cosine to the other live streams (-1 = n/a)
};

// Cross-stream consensus of the last-layer embeddings at one engine step
// (record_hidden only). One list per run lives in RunResult::hcons: the
// consensus is a property of the step, not of a branch, so it is stored once.
struct HCons {
    int   step     = 0;     // engine tick index within this run
    float cos_mean = 1.0f;  // mean pairwise cosine similarity
    float cos_min  = 1.0f;  // minimum pairwise cosine similarity
    float cos_max  = 1.0f;  // maximum pairwise cosine similarity
    int   n        = 0;     // streams that reported an embedding this step
};

struct RuntimeConfig {
    std::string model_path;
    int  max_context_tokens = 8192;
    int  max_branches       = 4;
    int  n_gpu_layers       = 99;
    bool flash_attn         = true;
    std::vector<std::string> lora_paths;      // adapters loaded into the context
    std::vector<float>       lora_scales;     // parallel to lora_paths (default 1.0)
    bool lora_after_prefill = false;          // prefill the shared prefix with base weights
    bool prefix_add_special = true;           // tokenize the shared prefix with BOS/special
    // A shared prefix is prefilled once with base weights, so it is only an
    // approximation when a per-row run mixes adapters. Off by default: such a
    // run fails instead of silently decoding from an approximated prefix.
    bool allow_approx_prefix = false;
};

struct RunRequest {
    std::string          shared_prefix;  // pre-formatted text; the runtime tokenizes it
    std::vector<Branch>  branches;
    int                  agree_k = 0;    // >0: stop when K branches emit the same extracted answer
};

struct BranchResult {
    std::string  id;
    std::string  text;
    int          suffix_tokens    = 0;
    int          generated_tokens = 0;
    double       ttft_ms          = 0.0;
    double       completion_ms    = 0.0;
    FinishReason finish           = FinishReason::Error;
    std::vector<StepRec> probs;          // filled only when Branch::record_probs
    std::vector<HRec>    hidden;         // filled only when Branch::record_hidden

    bool completed() const { return finish == FinishReason::Eos || finish == FinishReason::Agreement; }
    bool truncated() const { return finish == FinishReason::TokenLimit; }
};

struct RunResult {
    std::vector<BranchResult> branches;
    int    shared_prefix_tokens = 0;
    double prefill_ms           = 0.0;
    double all_done_ms          = 0.0;
    long long steps             = 0;     // engine ticks for this run
    long long joint_steps       = 0;     // ticks with two or more branches active
    std::vector<long long> active_hist;  // active_hist[n] = ticks with n active branches
    std::vector<HCons> hcons;            // cross-stream consensus per step (record_hidden)
    std::string model_id;                // model + backend build identifier
};

struct Event {
    enum Kind { BranchToken, BranchDone, RunDone, Error };
    Kind        kind = RunDone;
    std::string branch_id;
    std::string token;      // for BranchToken
    std::string message;    // for Error
};

struct MemoryArtifact {
    std::string text;
    std::vector<int> covered_message_ids;   // which messages the artifact covers
    std::string status;                     // complete | truncated | failed
    std::string model_id;                   // provenance; NOT a verification claim
};

}  // namespace parallel
