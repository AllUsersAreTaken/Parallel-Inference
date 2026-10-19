// Example: answer a conversation and produce a compact memory in parallel.
#include "parallel/runtime.hpp"
#include "parallel/workflows.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace parallel;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: answer_and_memory <model.gguf>\n"); return 2; }

    RuntimeConfig cfg;
    cfg.model_path = argv[1];
    cfg.max_context_tokens = 8192;
    Runtime rt(cfg);
    if (!rt.ok()) { fprintf(stderr, "%s\n", rt.error().c_str()); return 1; }

    std::vector<std::pair<std::string, std::string>> messages = {
        { "system", "You are a helpful assistant. Use only the conversation. Do not invent facts." },
        { "user", "Initial target city is Lisbon, kickoff March 14, budget 40k." },
        { "assistant", "Noted: Lisbon, March 14, 40k." },
        { "user", "Change of plan: move the city to Porto and the kickoff to March 21. Budget 55k." },
        { "assistant", "Updated: Porto, March 21, 55k." },
        { "user", "Remind me which city we settled on." },
    };

    Run run = answer_with_memory(rt, messages);
    while (auto e = run.next())
        if (e->kind == Event::BranchToken) fputs(e->token.c_str(), stdout);
    printf("\n");

    RunResult r = run.result();
    MemoryArtifact mem = memory_artifact(r);
    printf("\n--- memory (%s) ---\n%s\n", mem.status.c_str(), mem.text.c_str());
    printf("\nprefix %d tokens, all done %.0f ms\n", r.shared_prefix_tokens, r.all_done_ms);
    return 0;
}
