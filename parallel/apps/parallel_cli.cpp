// parallel-inference CLI demo.
//
//   parallel_cli compact   --model M --conversation conv.txt [--report run.json]
//   parallel_cli reasoning --model M --problem prob.txt    [--streams 8] [--agree 3]
//
// Conversation / problem files are plain text, one turn per line:
//   user: <text>
//   assistant: <text>
#include "parallel/runtime.hpp"
#include "parallel/workflows.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace parallel;

static std::vector<std::pair<std::string, std::string>> read_turns(const std::string & path) {
    std::vector<std::pair<std::string, std::string>> msgs;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t c = line.find(": ");
        if (c == std::string::npos) { msgs.push_back({ "user", line }); continue; }
        msgs.push_back({ line.substr(0, c), line.substr(c + 2) });
    }
    return msgs;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') {}
        else if (c == '\t') o += "\\t";
        else o += c;
    }
    return o;
}

static void write_report(const std::string & path, const RunResult & r) {
    if (path.empty()) return;
    FILE * f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "{\n  \"shared_prefix_tokens\": %d,\n  \"prefill_ms\": %.1f,\n  \"all_done_ms\": %.1f,\n",
            r.shared_prefix_tokens, r.prefill_ms, r.all_done_ms);
    fprintf(f, "  \"model\": \"%s\",\n  \"branches\": [\n", json_escape(r.model_id).c_str());
    for (size_t i = 0; i < r.branches.size(); ++i) {
        const auto & b = r.branches[i];
        fprintf(f, "    {\"id\": \"%s\", \"suffix_tokens\": %d, \"generated_tokens\": %d, "
                   "\"ttft_ms\": %.1f, \"completion_ms\": %.1f, \"finish_reason\": \"%s\"}%s\n",
                json_escape(b.id).c_str(), b.suffix_tokens, b.generated_tokens, b.ttft_ms, b.completion_ms,
                to_string(b.finish), i + 1 < r.branches.size() ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

static void drive(Run & run, const std::string & stream_only = "") {
    while (auto e = run.next()) {
        if (e->kind == Event::BranchToken) {
            if (stream_only.empty() || e->branch_id == stream_only) { fputs(e->token.c_str(), stdout); fflush(stdout); }
        } else if (e->kind == Event::BranchDone) {
            if (stream_only.empty() || e->branch_id == stream_only) { printf("\n[%s done]\n", e->branch_id.c_str()); fflush(stdout); }
        } else if (e->kind == Event::Error) {
            fprintf(stderr, "error: %s\n", e->message.c_str());
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: parallel_cli <compact|reasoning> --model M ...\n"); return 2; }
    std::string cmd = argv[1], model, conv, report, problem;
    int streams = 8, agree = 3;
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--conversation") conv = nx();
        else if (k == "--problem") problem = nx();
        else if (k == "--report") report = nx();
        else if (k == "--streams") streams = atoi(nx().c_str());
        else if (k == "--agree") agree = atoi(nx().c_str());
    }
    if (model.empty()) { fprintf(stderr, "--model required\n"); return 2; }

    RuntimeConfig cfg;
    cfg.model_path = model;
    cfg.max_context_tokens = 8192;
    cfg.max_branches = std::max(4, streams);
    Runtime rt(cfg);
    if (!rt.ok()) { fprintf(stderr, "runtime error: %s\n", rt.error().c_str()); return 1; }

    if (cmd == "compact") {
        if (conv.empty()) { fprintf(stderr, "--conversation required\n"); return 2; }
        auto messages = read_turns(conv);
        MemoryOptions opt;
        opt.max_new_tokens = 384;
        Run run = answer_with_memory(rt, messages, opt);
        printf("=== answer ===\n");
        drive(run, "answer");
        RunResult r = run.result();
        MemoryArtifact mem = memory_artifact(r, opt);
        printf("\n=== memory (%s) ===\n%s\n", mem.status.c_str(), mem.text.c_str());
        printf("\n[prefix %d tok, prefill %.0f ms, all done %.0f ms]\n", r.shared_prefix_tokens, r.prefill_ms, r.all_done_ms);
        fflush(stdout);
        write_report(report, r);
        return 0;
    }
    if (cmd == "reasoning") {
        std::vector<std::pair<std::string, std::string>> messages;
        if (!problem.empty()) messages.push_back({ "user", problem });
        else if (!conv.empty()) messages = read_turns(conv);
        else { fprintf(stderr, "--problem or --conversation required\n"); return 2; }
        ReasoningOptions opt;
        opt.streams = streams;
        opt.agree_k = agree;
        Run run = reason(rt, messages, opt);
        drive(run);
        RunResult r = run.result();
        printf("\n=== candidates ===\n");
        for (auto & b : r.branches)
            printf("  %s: %s (%s, %d tok)\n", b.id.c_str(), extract_answer(b.text).c_str(),
                   to_string(b.finish), b.generated_tokens);
        printf("[prefix %d tok, all done %.0f ms]\n", r.shared_prefix_tokens, r.all_done_ms);
        write_report(report, r);
        return 0;
    }
    fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    return 2;
}
