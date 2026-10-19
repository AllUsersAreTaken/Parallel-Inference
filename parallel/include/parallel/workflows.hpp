// parallel-inference workflows built on the run primitive.
#pragma once

#include "parallel/runtime.hpp"

#include <string>
#include <utility>
#include <vector>

namespace parallel {

struct MemoryOptions {
    int         max_new_tokens = 384;
    int         target_tokens  = 256;   // soft target (prototype setting)
    std::string instruction;            // empty -> default sectioned instruction
    int         covered_message_ids_to = -1;  // -1 -> all messages except the last
};

struct ReasoningOptions {
    int         streams        = 8;
    int         agree_k        = 3;
    int         max_new_tokens = 512;
    std::string instruction;            // empty -> default solve instruction
};

// Two branches over the conversation prefix: a foreground answer and an
// auxiliary memory artifact.
Run answer_with_memory(Runtime & runtime,
                       const std::vector<std::pair<std::string, std::string>> & messages,
                       const MemoryOptions & options = {});

// N independent reasoning branches with an agreement stop rule.
Run reason(Runtime & runtime,
           const std::vector<std::pair<std::string, std::string>> & messages,
           const ReasoningOptions & options = {});

// Build a memory artifact (text + coverage + status + provenance). A completed
// generation is not automatically faithful; this reports provenance, it does
// not label the memory "verified".
MemoryArtifact memory_artifact(const RunResult & result, const MemoryOptions & options = {});

// Extract a trailing numeric answer (used by the reasoning agreement rule).
std::string extract_answer(const std::string & text);

}  // namespace parallel
