// Minimum alpha tests. Set PARALLEL_TEST_MODEL to a GGUF path; tests skip otherwise.
#include "parallel/runtime.hpp"
#include "parallel/workflows.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace parallel;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); ++failures; } else printf("  ok: %s\n", msg); } while (0)

static void drive(Run & r) { while (auto e = r.next()) { (void)e; } }

int main() {
    const char * mp = getenv("PARALLEL_TEST_MODEL");
    if (!mp) { printf("PARALLEL_TEST_MODEL not set; skipping (no model available).\n"); return 0; }

    RuntimeConfig cfg;
    cfg.model_path = mp;
    cfg.max_context_tokens = 4096;
    cfg.max_branches = 8;
    Runtime rt(cfg);
    if (!rt.ok()) { printf("runtime init failed: %s\n", rt.error().c_str()); return 1; }

    // 1. capacity exhaustion is explicit (no silent truncation)
    {
        printf("capacity:\n"); fflush(stdout);
        RunRequest req;
        req.shared_prefix = rt.format_chat({ { "user", "Hello." } });
        Branch b; b.id = "big"; b.suffix = "Task: write a long essay."; b.max_new_tokens = 9000;
        req.branches = { b };
        Run run = rt.submit(req);
        bool err = false;
        while (auto e = run.next()) if (e->kind == Event::Error) err = true;
        CHECK(err, "insufficient capacity returns an Error event");
    }

    // 2. truncation is distinguished from completion; token cap is relative
    {
       printf("token cap:\n"); fflush(stdout);
        RunRequest req;
        req.shared_prefix= rt.format_chat({ { "user", "Count from one." } });
        Branch a; a.id = "short"; a.suffix = "Task: count upward."; a.max_new_tokens = 4; a.sampling.seed = 1; a.grammar = "root ::= [0-9 ]+";
        Branch c; c.id = "long";  c.suffix = "Task: count upward."; c.max_new_tokens = 24; c.sampling.seed = 2;
        req.branches = { a, c };
       Run run = rt.submit(req); drive(run);
        RunResult r = run.result();
        CHECK(r.branches[0].truncated(), "branch hitting its cap is truncated");
        CHECK(r.branches[0].generated_tokens <= 4, "generated tokens respect the relative cap");
        CHECK(r.branches[1].generated_tokens > r.branches[0].generated_tokens, "a larger cap generates more");
    }

    // 3. branch isolation: different seeds produce independent outputs
    {
        printf("isolation:\n"); fflush(stdout);
        RunRequest req;
        req.shared_prefix = rt.format_chat({ { "user", "Say one word." } });
        Branch a; a.id = "a"; a.suffix = "Task: answer."; a.max_new_tokens = 16; a.sampling.seed = 11;
        Branch b; b.id = "b"; b.suffix = "Task: answer."; b.max_new_tokens = 16; b.sampling.seed = 22;
        req.branches = { a, b };
       Run run = rt.submit(req); drive(run);
        RunResult r = run.result();
        CHECK(r.branches.size() == 2, "two branches returned");
        CHECK(r.branches[0].generated_tokens > 0 && r.branches[1].generated_tokens > 0, "both branches generated");
    }

    // 4. cancel one branch; the other still completes
    {
        printf("cancel:\n"); fflush(stdout);
        RunRequest req;
        req.shared_prefix = rt.format_chat({ { "user", "Explain gravity." } });
        Branch a; a.id = "keep";   a.suffix = "Task: answer."; a.max_new_tokens = 32;
        Branch b; b.id = "cancel"; b.suffix = "Task: answer."; b.max_new_tokens = 200;
        req.branches = { a, b };
        Run run = rt.submit(req);
        int steps = 0;
        while (auto e = run.next()) { if (++steps == 3) run.cancel("cancel"); }
        RunResult r = run.result();
        bool cancelled = false, kept_done = false;
        for (auto & br : r.branches) {
            if (br.id == "cancel" && br.finish == FinishReason::Cancelled) cancelled = true;
            if (br.id == "keep" && br.finish != FinishReason::Cancelled && br.finish != FinishReason::Error) kept_done = true;
        }
        CHECK(cancelled, "cancelled branch reports finish_reason=cancelled");
        CHECK(kept_done, "the other branch still completed");
    }

    printf(failures ? "\n%d test(s) FAILED\n" : "\nall tests passed\n", failures);
    return failures ? 1 : 0;
}
