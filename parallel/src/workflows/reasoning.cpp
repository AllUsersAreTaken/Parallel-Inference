// Workflow: N independent reasoning branches with an agreement stop rule.
#include "parallel/workflows.hpp"

namespace parallel {

static const char * DEFAULT_SOLVE_INSTRUCTION =
    "Task: solve the problem above. Think step by step, then end with a final line "
    "exactly of the form: The answer is: <number>";

Run reason(Runtime & runtime,
           const std::vector<std::pair<std::string, std::string>> & messages,
           const ReasoningOptions & options) {
    const std::string prefix = runtime.format_chat(messages);
    const std::string task = options.instruction.empty() ? DEFAULT_SOLVE_INSTRUCTION : options.instruction;

    auto m = messages;
    m.push_back({ "user", task });
    const std::string suffix = runtime.format_chat(m).substr(prefix.size());

    RunRequest req;
    req.shared_prefix = prefix;
    req.agree_k = options.agree_k;
    for (int i = 0; i < options.streams; ++i) {
        Branch b;
        b.id = "s" + std::to_string(i);
        b.suffix = suffix;
        b.max_new_tokens = options.max_new_tokens;
        b.role = Role::Auxiliary;
        b.sampling.seed = 1234u + 101u * (uint32_t)i;
        req.branches.push_back(b);
    }
    return runtime.submit(req);
}

std::string extract_answer(const std::string & text) {
    size_t last = std::string::npos;
    for (size_t i = 0; i < text.size(); ++i)
        if (std::isdigit((unsigned char)text[i]) && (i == 0 || !std::isdigit((unsigned char)text[i - 1]))) last = i;
    if (last == std::string::npos) return "";
    size_t j = last;
    while (j < text.size() && (std::isdigit((unsigned char)text[j]) || text[j] == '.')) ++j;
    return text.substr(last, j - last);
}

}  // namespace parallel
