// parallel-inference public runtime API.
#pragma once

#include "parallel/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace parallel {

// A run is the unit of execution: one shared prefix, several branches.
class Run {
public:
    ~Run();
    Run(Run &&) noexcept;
    Run & operator=(Run &&) noexcept;

    // Drives work until the next event. Returns nullopt only after RunDone.
    std::optional<Event> next();

    // Partial results are valid at any time; final after RunDone.
    RunResult result();

    // Cancel one branch, or all when branch_id is empty.
    void cancel(const std::string & branch_id = "");

    bool done() const;

private:
    friend class Runtime;
    struct Impl;
    explicit Run(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Runtime {
public:
    explicit Runtime(const RuntimeConfig & config);
    ~Runtime();

    bool ok() const;
    const std::string & error() const;

    Run submit(const RunRequest & request);

    // Apply the model's chat template to (role, content) pairs.
    std::string format_chat(const std::vector<std::pair<std::string, std::string>> & messages) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace parallel
