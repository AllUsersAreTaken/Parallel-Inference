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
};

struct RuntimeConfig {
    std::string model_path;
    int  max_context_tokens = 8192;
    int  max_branches       = 4;
    int  n_gpu_layers       = 99;
    bool flash_attn         = true;
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

    bool completed() const { return finish == FinishReason::Eos || finish == FinishReason::Agreement; }
    bool truncated() const { return finish == FinishReason::TokenLimit; }
};

struct RunResult {
    std::vector<BranchResult> branches;
    int    shared_prefix_tokens = 0;
    double prefill_ms           = 0.0;
    double all_done_ms          = 0.0;
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
