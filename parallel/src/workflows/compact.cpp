// Workflow: answer a message and, in parallel, produce a compact memory.
#include "parallel/workflows.hpp"

namespace parallel {

static const char * DEFAULT_MEMORY_INSTRUCTION =
    "Task: produce a compact memory of this conversation, for use as context on later turns. "
    "Use these sections and keep exact wording for facts:\n"
    "CURRENT GOAL:\nDECISIONS AND LATEST CORRECTIONS:\nCONSTRAINTS AND PREFERENCES:\n"
    "EXACT FACTS / NUMBERS:\nOPEN QUESTIONS / NEXT STEPS:\n"
    "Preserve negations, uncertainty, and changes of mind. Do not invent missing information. "
    "Omit conversational filler.";

Run answer_with_memory(Runtime & runtime,
                       const std::vector<std::pair<std::string, std::string>> & messages,
                       const MemoryOptions & options) {
    const std::string prefix = runtime.format_chat(messages);
    auto tail = [&](const std::string & task) {
        auto m = messages;
        m.push_back({ "user", task });
        std::string full = runtime.format_chat(m);
        return full.substr(prefix.size());
    };

    RunRequest req;
    req.shared_prefix = prefix;

    Branch answer;
    answer.id = "answer";
    answer.suffix = tail("Task: answer the latest user message above.");
    answer.max_new_tokens = 160;
    answer.role = Role::Foreground;

    Branch memory;
    memory.id = "memory";
    memory.suffix = tail(options.instruction.empty() ? DEFAULT_MEMORY_INSTRUCTION : options.instruction);
    memory.max_new_tokens = options.max_new_tokens;
    memory.role = Role::Auxiliary;

    req.branches = { answer, memory };
    return runtime.submit(req);
}

MemoryArtifact memory_artifact(const RunResult & result, const MemoryOptions & options) {
    MemoryArtifact a;
    a.model_id = result.model_id;
    for (const auto & b : result.branches) {
        if (b.id != "memory") continue;
        a.text = b.text;
        a.status = b.truncated() ? "truncated" : (b.completed() ? "complete" : "failed");
    }
    if (a.status.empty()) a.status = "failed";
    // coverage: messages [0, to). -1 -> all messages except the last.
    int to = options.covered_message_ids_to;
    if (to < 0) to = 0;   // caller sets the real count; 0 means "unspecified"
    for (int i = 0; i < to; ++i) a.covered_message_ids.push_back(i);
    return a;
}

}  // namespace parallel
