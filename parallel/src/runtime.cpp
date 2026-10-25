// parallel-inference runtime: shared-prefix runs with branches.
#include "parallel/runtime.hpp"

#include "backends/llama_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>

namespace parallel {

using namespace parallel::detail;

namespace {

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::string extract_number(const std::string & s) {
    size_t last = std::string::npos;
    for (size_t i = 0; i < s.size(); ++i)
        if (std::isdigit((unsigned char)s[i]) && (i == 0 || !std::isdigit((unsigned char)s[i - 1]))) last = i;
    if (last == std::string::npos) return "";
    size_t j = last; while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
    return s.substr(last, j - last);
}

static std::string tok_piece(const llama_vocab * voc, llama_token tok) {
    char p[64];
    int np = llama_token_to_piece(voc, tok, p, sizeof(p), 0, false);
    if (np >= 0) return std::string(p, np);
    std::vector<char> big((size_t)(-np) + 1);
    np = llama_token_to_piece(voc, tok, big.data(), (int)big.size(), 0, false);
    return np > 0 ? std::string(big.data(), (size_t)np) : std::string();
}

// Full-vocabulary stats of one raw-logit row (temperature 1, before any
// top-k/top-p/temperature sampler). One pass for max/sum/sum(z*e^z) plus
// top-2 tracking; entropy in nats.
struct RawStats {
    float logz = 0.0f, p1 = 0.0f, margin = 0.0f, entropy = 0.0f;
    bool  valid = false;
};

static RawStats raw_stats(const std::vector<float> & lg) {
    RawStats s;
    if (lg.empty()) return s;
    float mx = lg[0];
    for (float v : lg) if (v > mx) mx = v;
    double sum = 0.0, sz = 0.0;
    for (float v : lg) { double e = exp((double)v - mx); sum += e; sz += (double)v * e; }
    if (!(sum > 0.0)) return s;
    const double logz = (double)mx + log(sum);
    float b1 = -1.0e30f, b2 = -1.0e30f;
    for (float v : lg) {
        if (v > b1) { b2 = b1; b1 = v; }
        else if (v > b2) { b2 = v; }
    }
    s.logz    = (float)logz;
    s.p1      = (float)exp((double)b1 - logz);
    s.margin  = s.p1 - (float)exp((double)b2 - logz);
    s.entropy = (float)(logz - sz / sum);
    s.valid   = true;
    return s;
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
    std::vector<StepRec> probs;    // record_probs only
    std::vector<float>   rawbuf;   // record_probs only: raw logits copy (logp)
    std::vector<HRec>    hidden;   // record_hidden only
    std::vector<float>   hvec;     // record_hidden only: last embedding vector
    float                hnorm = 0.0f;
    bool                 hfresh = false;  // embedding captured in the current tick
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
    std::vector<HCons> hcons;      // record_hidden only: one row per consensus step
    long long tick = -1;           // engine tick counter within this run
    bool any_hidden = false;       // some branch asked for embeddings

    RunProcess(Engine & e, std::deque<Event> * ev, int k)
        : eng(e), events(ev), t0(std::chrono::steady_clock::now()), agree_k(k) { min_slots = 0; }

    void seq_ids(std::vector<llama_seq_id> & o) const override {
        o.clear();
        for (auto & b : br) o.push_back(b.seq);
    }
    void want(int budget, int base, std::vector<Request> & out) override {
        ++tick;
        base_ = base; emitted.clear();
        any_hidden = false;
        for (auto & b : br) b.hfresh = false;
        for (int j = 0; j < (int)br.size(); ++j) {
            if (br[j].fin) continue;
            if ((int)emitted.size() >= budget) break;
            emitted.push_back(j);
            if (br[j].cfg.record_hidden) any_hidden = true;
            out.push_back({ br[j].seq, br[j].pos, br[j].next.empty() ? 0 : br[j].next[0], true, br[j].cfg.adapter_id });
        }
    }
    void on_logits(int idx, const float * lg) override {
        int slot = idx - base_;
        if (slot < 0 || slot >= (int)emitted.size()) return;
        int j = emitted[slot];
        BranchState & b = br[j];
        if (b.fin) return;
        llama_token tok = b.next.empty() ? 0 : b.next[0];
        std::string piece = tok_piece(eng.vocab(), tok);
        b.text += piece;
        b.pos += 1; b.generated += 1;
        if (b.ttft == 0.0) b.ttft = ms_since(t0);
        if (!piece.empty()) events->push_back({ Event::BranchToken, b.cfg.id, piece, "" });

        // Embeddings are produced for the same rows that requested logits.
        // Capture before the EOS/limit returns so every active stream reports
        // on every tick and the consensus step ids stay contiguous.
        if (b.cfg.record_hidden) {
            const float * e = llama_get_embeddings_ith(eng.ctx(), idx);
            if (!e) {
                hidden_fail("hidden capture: llama_get_embeddings_ith returned null (embeddings unsupported)");
                return;
            }
            const int ne = eng.n_embd_out();
            b.hvec.assign(e, e + ne);
            double ss = 0.0;
            for (float v : b.hvec) ss += (double)v * (double)v;
            b.hnorm = (float)std::sqrt(ss);
            b.hfresh = true;
            HRec h; h.step = b.generated; h.norm = b.hnorm;
            b.hidden.push_back(h);
        }
        if (any_hidden && slot + 1 == (int)emitted.size()) step_consensus();

        if (llama_vocab_is_eog(eng.vocab(), tok)) { finish_branch(j, FinishReason::Eos); return; }
        if (b.generated >= b.cfg.max_new_tokens) { finish_branch(j, FinishReason::TokenLimit); return; }
        // lg predicts the next token; snapshot it BEFORE the sampler chain
        // mutates anything, so stats stay raw-logit (pre temp/top-k/top-p).
        const bool rec = b.cfg.record_probs;
        RawStats rs;
        if (rec) {
            b.rawbuf.assign(lg, lg + llama_n_vocab(eng.vocab()));
            rs = raw_stats(b.rawbuf);
        }
        b.next.assign(1, llama_sampler_sample(b.smpl, eng.ctx(), idx));
        if (rec && rs.valid) {
            StepRec r;
            r.step     = b.generated;
            r.pos      = (int)b.pos;
            r.token    = b.next[0];
            r.entropy  = rs.entropy;
            r.p1       = rs.p1;
            r.margin   = rs.margin;
            r.logp     = (float)((double)b.rawbuf[(size_t)r.token] - (double)rs.logz);
            r.text_off = (int)b.text.size();          // piece appends next tick
            r.text_len = (int)tok_piece(eng.vocab(), r.token).size();
            b.probs.push_back(r);
        }
    }
    // Pairwise cosine over the streams that reported an embedding this tick.
    // Also stores each stream's mean cosine to the others on that step's HRec.
    void step_consensus() {
        std::vector<int> live;
        for (int j : emitted) if (br[j].hfresh && !br[j].hvec.empty()) live.push_back(j);
        if (live.size() < 2) return;
        std::vector<double> per(live.size(), 0.0);
        std::vector<int>    pn(live.size(), 0);
        double sum = 0.0;
        float mn = 1.0f, mx = -1.0f;
        int np = 0;
        for (size_t a = 0; a < live.size(); ++a) {
            for (size_t c = a + 1; c < live.size(); ++c) {
                const BranchState & x = br[live[a]];
                const BranchState & y = br[live[c]];
                if (x.hvec.size() != y.hvec.size()) continue;
                double dot = 0.0;
                for (size_t k = 0; k < x.hvec.size(); ++k) dot += (double)x.hvec[k] * (double)y.hvec[k];
                const double den = (double)x.hnorm * (double)y.hnorm;
                if (!(den > 0.0)) continue;
                float cs = (float)(dot / den);
                if (cs > 1.0f) cs = 1.0f;
                if (cs < -1.0f) cs = -1.0f;
                sum += cs;
                if (cs < mn) mn = cs;
                if (cs > mx) mx = cs;
                ++np;
                per[a] += cs; ++pn[a];
                per[c] += cs; ++pn[c];
            }
        }
        if (np == 0) return;
        HCons c;
        c.step = (int)tick;
        c.cos_mean = (float)(sum / np);
        c.cos_min = mn;
        c.cos_max = mx;
        c.n = (int)live.size();
        hcons.push_back(c);
        for (size_t a = 0; a < live.size(); ++a) {
            if (pn[a] == 0) continue;
            BranchState & b = br[live[a]];
            if (!b.hidden.empty() && b.hidden.back().step == b.generated)
                b.hidden.back().cos_mean = (float)(per[a] / pn[a]);
        }
    }
    void hidden_fail(const char * msg) {
        events->push_back({ Event::Error, "", "", msg });
        for (auto & x : br) if (!x.fin) {
            x.fin = true; x.finish = FinishReason::Error; x.done_ms = ms_since(t0);
            events->push_back({ Event::BranchDone, x.cfg.id, "", "" });
        }
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
    c.lora_paths = config.lora_paths;
    c.lora_scales = config.lora_scales;
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
    long long joint_steps = 0;
    std::vector<long long> active_hist;
    std::vector<HCons> hcons;
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

    // per-row adapters: every row must name a loaded adapter, otherwise the
    // run would mix segmented and global rows with no single correct meaning
    bool per_row = false;
    for (const auto & b : request.branches) if (b.adapter_id >= 0) { per_row = true; break; }
    if (per_row) {
        for (const auto & b : request.branches) {
            if (b.adapter_id < 0 || b.adapter_id >= impl_->eng.n_adapters()) {
                im->capacity_error = "per-row adapter id out of range or missing (one --lora per index)";
                im->done_flag = true;
                return Run(std::move(im));
            }
        }

        // token_embd / output / MoE expert weights bypass build_lora_mm, so an
        // adapter that carries them cannot be applied per row
        std::string why;
        if (!impl_->eng.segments_supported(why)) {
            im->capacity_error = "per-row mode rejected: " + why;
            im->done_flag = true;
            return Run(std::move(im));
        }

        // the shared prefix is prefilled once with base weights; that is only
        // meaningful for a single adapter. anything else is an approximation.
        std::vector<int> distinct;
        for (const auto & b : request.branches)
            if (std::find(distinct.begin(), distinct.end(), b.adapter_id) == distinct.end())
                distinct.push_back(b.adapter_id);
        if (!request.shared_prefix.empty() && distinct.size() > 1) {
            if (!impl_->cfg.allow_approx_prefix) {
                im->capacity_error = "per-row run has " + std::to_string(distinct.size()) +
                    " adapters and a shared prefix: the prefix cannot carry per-row adapters "
                    "(set RuntimeConfig::allow_approx_prefix to accept the approximation)";
                im->done_flag = true;
                return Run(std::move(im));
            }
            fprintf(stderr, "warning: shared prefix prefilled with base weights for %zu adapters "
                    "(approximation enabled by allow_approx_prefix)\n", distinct.size());
        }
    }

    for (auto & b : request.branches) if (b.record_hidden) { impl_->eng.enable_embeddings(); break; }

    // capacity check: prefix + sum(suffix + max_new) must fit, no silent truncation
    auto tok = [&](const std::string & s, bool add_special) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(impl_->eng.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), add_special, true);
        if (n < 0) { t.resize(-n); n = llama_tokenize(impl_->eng.vocab(), s.c_str(), (int)s.size(), t.data(), (int)t.size(), add_special, true); }
        t.resize(n); return t;
    };
    auto prefix = tok(request.shared_prefix, impl_->cfg.prefix_add_special);
    im->shared_prefix_tokens = (int)prefix.size();
    long long needed = (long long)prefix.size();
    for (auto & b : request.branches) needed += (long long)tok(b.suffix + impl_->assistant_marker, false).size() + b.max_new_tokens + 1;
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
    impl_->eng.clear_segments();
    // arm E: the shared prefix is prefilled with base weights; adapters are
    // enabled afterwards so one prefix KV can serve adapters that would
    // otherwise force a per-adapter prefill.
    // per-row: no adapter is correct for a prefix shared by different adapters,
    // so it stays base. The suffix of each row is prefilled with its adapter.
    if (per_row || impl_->cfg.lora_after_prefill) impl_->eng.clear_adapters();
    else impl_->eng.apply_adapters_all();
    if (!impl_->eng.prefill(0, prefix, false)) { im->capacity_error = "prefill failed"; im->done_flag = true; return Run(std::move(im)); }
    impl_->eng.sync();
    im->prefill_ms = ms_since(tp0);
    if (!per_row && impl_->cfg.lora_after_prefill) impl_->eng.apply_adapters_all();

    // fork + append each branch's suffix
    // copy the shared prefix to every branch BEFORE any suffix decode
    for (int i = 1; i < (int)request.branches.size(); ++i) impl_->eng.seq_cp(0, i);

    for (int i = 0; i < (int)request.branches.size(); ++i) {
        const Branch & bcfg = request.branches[i];
        BranchState bs;
        bs.cfg = bcfg;
        bs.seq = i;
        auto st = tok(bcfg.suffix + impl_->assistant_marker, false);
        if (st.empty()) { im->capacity_error = "empty branch suffix tokenization"; im->done_flag = true; return Run(std::move(im)); }
        if (getenv("PARALLEL_DEBUG")) fprintf(stderr, "submit: branch %d seq=%d suffix=%zu\n", i, i, st.size());
        bs.suffix_tokens = (int)st.size();
        if (per_row) impl_->eng.apply_adapters({ bcfg.adapter_id }, {});
        if (!impl_->eng.decode_from(i, st, (llama_pos)prefix.size(), true)) {
            im->capacity_error = "suffix prefill failed for branch " + std::to_string(i);
            im->done_flag = true;
            return Run(std::move(im));
        }
        impl_->eng.sync();
        bs.pos = (llama_pos)(prefix.size() + st.size());
        bs.p_start = bs.pos;

        auto sp = llama_sampler_chain_default_params();
        bs.smpl = llama_sampler_chain_init(sp);
        if (!bcfg.grammar.empty()) llama_sampler_chain_add(bs.smpl, llama_sampler_init_grammar(impl_->eng.vocab(), bcfg.grammar.c_str(), "root"));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_top_k(bcfg.sampling.top_k));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_top_p(bcfg.sampling.top_p, 1));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_temp(bcfg.sampling.temperature));
        llama_sampler_chain_add(bs.smpl, llama_sampler_init_dist(bcfg.sampling.seed));
        RawStats rs;
        if (bcfg.record_probs) {
            const float * lp = llama_get_logits_ith(impl_->eng.ctx(), -1);
            bs.rawbuf.assign(lp, lp + llama_n_vocab(impl_->eng.vocab()));
            rs = raw_stats(bs.rawbuf);
        }
        bs.next.assign(1, llama_sampler_sample(bs.smpl, impl_->eng.ctx(), -1));
        if (bcfg.record_probs && rs.valid) {
            StepRec r;
            r.step     = 0;
            r.pos      = (int)bs.pos;
            r.token    = bs.next[0];
            r.entropy  = rs.entropy;
            r.p1       = rs.p1;
            r.margin   = rs.margin;
            r.logp     = (float)((double)bs.rawbuf[(size_t)r.token] - (double)rs.logz);
            r.text_off = 0;
            r.text_len = (int)tok_piece(impl_->eng.vocab(), r.token).size();
            bs.probs.push_back(r);
        }
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
    const double step_ms = im.eng->tick(64, active);
    if (step_ms < 0.0) {
        const std::string & seg = im.eng->segment_error();
        Event e;
        e.kind = Event::Error;
        e.message = seg.empty() ? "decode failed" : ("decode rejected: " + seg);
        im.events.push_back(e);
        im.events.push_back({ Event::RunDone, "", "", "" });
        im.done_flag = true;
        return next();
    }
    if ((int)im.active_hist.size() <= active) im.active_hist.resize((size_t)active + 1, 0);
    im.active_hist[(size_t)active] += 1;
    if (active >= 2) im.joint_steps += 1;
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
        im.hcons = im.proc->hcons;
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
    r.steps = im.steps;
    r.joint_steps = im.joint_steps;
    r.active_hist = im.active_hist;
    r.hcons = im.hcons;
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
        br.probs = b.probs;
        br.hidden = b.hidden;
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
