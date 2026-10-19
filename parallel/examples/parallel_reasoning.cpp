// Example: N parallel reasoning branches with an agreement stop rule.
#include "parallel/runtime.hpp"
#include "parallel/workflows.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace parallel;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: parallel_reasoning <model.gguf> [problem]\n"); return 2; }
    std::string problem = argc > 2 ? argv[2] : "A shop sells 5 apples a day for 3 days, then 7 a day for 2 days. How many apples in total?";

    RuntimeConfig cfg;
    cfg.model_path = argv[1];
    cfg.max_context_tokens = 4096;
    cfg.max_branches = 8;
    Runtime rt(cfg);
    if (!rt.ok()) { fprintf(stderr, "%s\n", rt.error().c_str()); return 1; }

    ReasoningOptions opt;
    opt.streams = 8;
    opt.agree_k = 3;

    Run run = reason(rt, { { "user", problem } }, opt);
    while (auto e = run.next()) { (void)e; }

    RunResult r = run.result();
    for (auto & b : r.branches)
        printf("  %s -> %-8s (%s)\n", b.id.c_str(), extract_answer(b.text).c_str(), to_string(b.finish));
    printf("all done %.0f ms\n", r.all_done_ms);
    return 0;
}
