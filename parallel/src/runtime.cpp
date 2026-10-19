// parallel-inference runtime: shared-prefix runs with branches.
#include "parallel/runtime.hpp"

#include "backends/llama_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <map>

namespace parallel {

using namespace parallel::detail;

namespace {

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

llama_token argmax(const float * lg, int nv) {
    int b = 0; float bv = lg[0];
    for (int i = 1; i < nv; ++i) if (lg[i] > bv) { bv = lg[i]; b = i; }
    return b;
}

static std::string extract_number(const std::string & s) {
    size_t last = std::string::npos;
    for (size_t i = 0; i < s.size(); ++i)
        if (std::isdigit((unsigned char)s[i]) && (i == 0 || !std::isdigit((unsigned char)s[i - 1]))) last = i;
    if (last == std::string::npos) return "";
    size_t j = last; while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
    return s.substr(last, j - last);
}

// One branch's live state.
struct BranchState {
    Branch       cfg;
    llama_seq_id seq = 0;
    llama_sampler * smpl = nullptr;
    std::vector<llama_token> next;
    llama_pos    pos = 0;
    llama_pos    p_start = 0;
    std::string  text;
    int          generated = 0;
    double       ttft = 0.0;
    double       done_ms = 0.0;
    bool         fin = false;
    FinishReason finish = FinishReason::Error;
    int          suffix_tokens = 0;
};

// Drives all branches of a run through one engine tick per step.
struct RunProcess : Process {
    Engine & eng;
    std::vector<BranchState> br;
    std::deque<Event> * events;
    std::chrono::steady_clock::time_point t0;
    int  agree_k;
    int  base_ = 0;
    std::vector<int> emitted;

    RunProcess(Engine & e, std::deque<Event> * ev, int k)
        : eng(e), events(ev), t0(std::chrono::steady_clock::now()), agree_k(k) { min_slots = 0; }

    void seq_ids(std::vector<llama_seq_id> & o) const override {
        o.clear();
        for (auto & b : br) o.push_back(b.seq);
    }
    void want(int budget, int base, std::vector<Request> & out) override {
        base_ = base; emitted.clear();
        for (int j = 0; j < (int)br.size(); ++j) {
            if (br[j].fin) continue;
            if ((int)emitted.size() >= budget) break;
            emitted.push_back(j);
            out.push_back({ br[j].seq, br[j].pos, br[j].next.empty() ? 0 : br[j].next[0], true });
        }
    }
    void on_logits(int idx, const float * lg) override {
        int slot = idx - base_;
        if (slot < 0 || slot >= (int)emitted.size()) return;
        int j = emitted[slot];
        BranchState & b = br[j];
        if (b.fin) return;
        llama_token tok = b.next.empty() ? 0 : b.next[0];
        char p[64]; int np = llama_token_to_piece(eng.vocab(), tok, p, sizeof(p), 0, false);
        if (np > 0) b.text.append(p, np);
        b.pos += 1; b.generated += 1;
        if (b.ttft == 0.0) b.ttft = ms_since(t0);
        if (!b.text.empty()) events->push_back({ Event::BranchToken, b.cfg.id, b.text.substr(b.text.size() - (np > 0 ? np : 0)), "" });

        if (llama_vocab_is_eog(eng.vocab(), tok)) { finish_branch(j, FinishReason::Eos); return; }
        if (b.generated >= b.cfg.max_new_tokens) { finish_branch(j, FinishReason::TokenLimit); return; }
        b.next.assign(1, llama_sampler_sample(b.smpl, eng.ctx(), idx));
    }
    void finish_branch(int j, FinishReason r) {
        BranchState & b = br[j];
        b.fin = true; b.finish = r; b.done_ms = ms_since(t0);
        events->push_back({ Event::BranchDone, b.cfg.id, "", "" });
        check_agree();
    }
    void check_agree() {
        if (agree_k <= 0) return;
        std::map<std::string, int> cnt;
        for (auto & b : br) if (b.fin && b.finish != FinishReason::Cancelled) { std::string a = extract_number(b.text); if (!a.empty()) cnt[a]++; }
        for (auto & kv : cnt) {
            if (kv.second < agree_k) continue;
            for (auto & b : br) if (b.fin && extract_number(b.text) == kv.first) b.finish = FinishReason::Agreement;
            for (auto & b : br) if (!b.fin) { b.fin = true; b.finish = FinishReason::Cancelled; b.done_ms = ms_since(t0); eng.retire_seq(b.seq); events->push_back({ Event::BranchDone, b.cfg.id, "", "" }); }
            return;
        }
    }
    bool done() const override {
        for (auto & b : br) if (!b.fin) return false;
        return true;
    }
};

}  // namespace

// ---------------------------------------------------------------- Runtime

struct Runtime::Impl {
    RuntimeConfig cfg;
    detail::Engine eng;
    std::string error;
    std::string assistant_marker;
    bool ok = false;
};

Runtime::Runtime(const RuntimeConfig & config) : impl_(new Impl()) {
    impl_->cfg = config;
    detail::Engine::Config c;
    c.model = config.model_path;
    c.n_ctx = config.max_context_tokens;
    c.n_seq_max = std::max(2, config.max_branches);
    c.ngl = config.n_gpu_layers;
    c.fa = config.flash_attn;
    c.max_batch = 2048;
    if (!impl_->eng.init(c)) { impl_->error = "failed to load model or create context"; return; }
    { std::vector<llama_chat_message> m = { { "assistant", "MMM" } };
      std::vector<char> b(4096);
      int n = llama_chat_apply_template(impl_->eng.model_chat_template(), m.data(), 1, false, b.data(), (int)b.size());
      std::string s(b.data(), n > 0 ? n : 0);
      size_t p = s.find("MMM");
      impl_->assistant_marker = p == std::string::npos ? std::string() : s.substr(0, p); }
    impl_->ok = true;
}

Runtime::~Runtime() { if (impl_ && impl_->ok) impl_->eng.shutdown(); }

bool Runtime::ok() const { return impl_ && impl_->ok; }
const std::string & Runtime::error() const { static std::string e; return impl_ ? impl_->error : e; }

std::string Runtime::format_chat(const std::vector<std::pair<std::string, std::string>> & messages) const {
    std::vector<llama_chat_message> msgs;
    for (auto & m : messages) msgs.push_back({ m.first.c_str(), m.second.c_str() });
    std::vector<char> buf(1 << 20);
    int n = llama_chat_apply_template(impl_->eng.model_chat_template(), msgs.data(), msgs.size(), false, buf.data(), (int)buf.size());
    if (n > (int)buf.size()) { buf.resize(n); n = llama_chat_apply_template(impl_->eng.model_chat_template(), msgs.data(), msgs.size(), false, buf.data(), (int)buf.size()); }
    return std::string(buf.data(), n > 0 ? n : 0);
}

// ---------------------------------------------------------------- Run

struct Run::Impl {
    detail::Engine * eng = nullptr;
    RuntimeConfig cfg;
    RunRequest req;
    std::deque<Event> events;
    std::vector<BranchState> branches;
    RunProcess * proc = nullptr;
    bool done_flag = false;
    bool started = false;
    long long steps = 0;
    std::string capacity_error;
    int shared_prefix_tokens = 0;
    double prefill_ms = 0;
    std::string model_id;

    ~Impl() {
        delete proc;                            // proc->br shares sampler pointers, not ownership
        for (auto & b : branches) if (b.smpl) llama_sampler_free(b.smpl);
    }
};

Run::Run(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Run::~Run() = default;
Run::Run(Run &&) noexcept = default;
Run & Run::operator=(Run &&) noexcept = default;

bool Run::done() const { return !impl_ || impl_->done_flag; }

Run Runtime::submit(const RunRequest & request) {
    std::unique_ptr<Run::Impl> im(new Run::Impl());
    im->eng = &impl_->eng;
    im->cfg = impl_->cfg;
    im->req = request;
    im->model_id = std::string("llama.cpp build ") + llama_version();

    if (!impl_->ok) { im->capacity_error = "runtime not initialized"; im->done_flag = true; return Run(std::move(im)); }
    if (request.branches.empty()) { im->capacity_error = "no branches"; im->done_flag = true; return Run(std::move(im)); }

    // capacity check: prefix + sum(suffix + max_new) must fit, no silent truncation
    auto tok = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(impl_->eng.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(impl_->eng.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), false, true); }
        t.resize(n); return t;
    };
    auto prefix = tok(request.shared_prefix);
    im->shared_prefix_tokens = (int)prefix.size();
    long long needed = (long long)prefix.size();
    for (auto & b : request.branches) needed += (long long)tok(b.suffix + impl_->assistant_marker).size() + b.max_new_tokens + 1;
    if (needed > (long long)impl_->cfg.max_context_tokens) {
        im->capacity_error = "context capacity insufficient: need " + std::to_string(needed) +
                             " tokens, have " + std::to_string(impl_->cfg.max_context_tokens);
        im->done_flag = true;
        return Run(std::move(im));
    }

    // prefill the shared prefix once on seq 0
    auto tp0 = std::chrono::steady_clock::now();
    if (getenv("PARALLEL_DEBUG")) fprintf(stderr, "submit: prefix=%zu branches=%zu\n", prefix.size(), request.branches.size());
    impl_->eng.clear();
    if (!impl_->eng.prefill(0, prefix, false)) { im->capacity_error = "prefill failed"; im->done_flag = true; return Run(std::move(im)); }
    impl_->eng.sync();
    im->prefill_ms = ms_since(tp0);

    // fork + append each branch's suffix
    // copy the shared prefix to every branch BEFORE any suffix decode
    for (int i = 1; i < (int)request.branches.size(); ++i) impl_->eng.seq_cp(0, i);

    for (int i = 0; i < (int)request.branches.size(); ++i) {
        const Branch & bcfg = request.branches[i];
        BranchState bs;
        bs.cfg = bcfg;
        bs.seq = i;
        auto st = tok(bcfg.suffix + impl_->assistant_marker);
        if (getenv("PARALLEL_DEBUG")) fprintf(stderr, "submit: branch %d seq=%d suffix=%zu\n", i, i, st.size());
        bs.suffix_tokens = (int)st.size();
        impl_->eng.decode_from(i, st, (llama_pos)prefix.size(), true);
        impl_->eng.sync();
        llama_token seed = argmax(llama_get_logits_ith(impl_->eng.ctx(), (int)st.size() - 1), impl_->eng.n_vocab());
        bs.next.assign(1, seed);
        bs.pos = (llama_pos)(prefix.size() + st.size());
        bs.p_start = bs.pos;

        auto sp = llama_sampler_chain_default_params();
        bs.smpl = llama_sampler_chain_init(sp);
        if (!bcfg.grammar.empty()) llama_sampler_chain_add(bs.smpl, llama_sampler_init_grammar(impl_->eng.vocab(), bcfg.grammar.c_str(), "root"));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_top_k(bcfg.sampling.top_k));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_top_p(bcfg.sampling.top_p, 1));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_temp(bcfg.sampling.temperature));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_dist(bcfg.sampling.seed));
        im->branches.push_back(std::move(bs));
    }
    im->started = true;
    return Run(std::move(im));
}

std::optional<Event> Run::next() {
    Impl & im = *impl_;
    if (!im.events.empty()) { Event e = im.events.front(); im.events.pop_front(); return e; }

    if (!im.capacity_error.empty()) {
        Event e; e.kind = Event::Error; e.message = im.capacity_error;
        im.events.push_back(e);
        im.events.push_back({ Event::RunDone, "", "", "" });
        im.capacity_error.clear();
        im.done_flag = true;
        return next();
    }
    if (im.done_flag) return std::nullopt;
    if (!im.started) { im.done_flag = true; return std::nullopt; }

    if (!im.proc) {
        im.proc = new RunProcess(*im.eng, &im.events, im.req.agree_k);
        im.proc->br = im.branches;              // copy (samplers owned by im.branches)
        im.eng->reset_processes();
        im.eng->add(im.proc);
    }
    int active = 0;
    im.eng->tick(64, active);
    if (getenv("PARALLEL_DEBUG")) {
        fprintf(stderr, "tick step=%lld active=%d:", im.steps, active);
        for (auto & b : im.proc->br) fprintf(stderr, " [%s g=%d fin=%d r=%d]", b.cfg.id.c_str(), b.generated, b.fin ? 1 : 0, (int)b.finish);
        fprintf(stderr, "\n");
    }
    if (++im.steps > 100000) {
        Event e; e.kind = Event::Error; e.message = "step limit exceeded (branch did not terminate)";
        im.events.push_back(e);
        im.events.push_back({ Event::RunDone, "", "", "" });
        im.done_flag = true;
        return next();
    }
    if (im.proc->done()) {
        im.branches = im.proc->br;              // pull back final state
        im.events.push_back({ Event::RunDone, "", "", "" });
        im.done_flag = true;
    }
    return next();
}

RunResult Run::result() {
    Impl & im = *impl_;
    RunResult r;
    r.shared_prefix_tokens = im.shared_prefix_tokens;
    r.prefill_ms = im.prefill_ms;
    r.model_id = im.model_id;
    double all = 0;
    for (auto & b : im.branches) {
        BranchResult br;
        br.id = b.cfg.id;
        br.text = b.text;
        br.suffix_tokens = b.suffix_tokens;
        br.generated_tokens = b.generated;
        br.ttft_ms = b.ttft;
        br.completion_ms = b.done_ms;
        br.finish = b.finish;
        if (!b.fin) br.finish = FinishReason::Error;
        r.branches.push_back(br);
        all = std::max(all, b.done_ms);
    }
    r.all_done_ms = all;
    return r;
}

void Run::cancel(const std::string & branch_id) {
    Impl & im = *impl_;
    std::vector<BranchState> & brs = im.proc ? im.proc->br : im.branches;
    for (auto & b : brs) {
        if (b.fin) continue;
        if (branch_id.empty() || b.cfg.id == branch_id) {
            b.fin = true; b.finish = FinishReason::Cancelled;
            b.done_ms = im.proc ? ms_since(im.proc->t0) : 0.0;
            im.eng->retire_seq(b.seq);
        }
    }
}

}  // namespace parallel
