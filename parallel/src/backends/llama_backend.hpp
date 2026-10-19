// RecycleLM engine core.
//
// One model/context/batch, one decode per tick. Processes own sequences and
// declare what they want in the next step; the engine builds ONE llama_batch,
// decodes once, and routes logits back by batch index. The batch-index routing
// is the invariant (same class as the earlier logits-clobbering bug).
//
// Process hooks added up front: min_slots (allocator hint) and on_retire.

#pragma once
// Internal llama.cpp backend. Extracted from harness/engine/core.h (unchanged logic).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include "llama.h"

#include <windows.h>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace parallel { namespace detail {
struct Request {
    llama_seq_id seq;
    llama_pos    pos;
    llama_token  tok;
    bool         want_logits;
};

struct Process {
    virtual ~Process() {}
    int min_slots = 1;                     // slots the allocator should reserve
    std::function<void()> on_retire;       // called after the engine frees seqs
    virtual void  seq_ids(std::vector<llama_seq_id> & out) const = 0;
    virtual void  want(int budget, int base_idx, std::vector<Request> & out) = 0;
    virtual void  on_logits(int batch_idx, const float * logits) = 0;
    virtual bool  done() const = 0;
    virtual float value() const { return 0.0f; }  // allocator priority
};

class Engine {
public:
    struct Config {
        std::string model;
        int  n_ctx      = 8192;
        int  n_seq_max  = 32;
        int  ngl        = 99;
        int  max_batch  = 2048;
        bool kv_unified = true;
        bool fa         = true;
    };

    bool init(const Config & c) {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = c.ngl;
        mp.progress_callback = [](float, void *) { return true; };
        model_ = llama_model_load_from_file(c.model.c_str(), mp);
        if (!model_) return false;
        vocab_ = llama_model_get_vocab(model_);
        n_vocab_ = llama_vocab_n_tokens(vocab_);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = c.n_ctx; cp.n_batch = c.max_batch; cp.n_ubatch = 512;
        cp.n_seq_max = c.n_seq_max;
        cp.flash_attn_type = c.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.kv_unified = c.kv_unified;
        cp.n_threads = 8; cp.n_threads_batch = 8;
        ctx_ = llama_init_from_model(model_, cp);
        if (!ctx_) return false;
        mem_ = llama_get_memory(ctx_);
        batch_ = llama_batch_init(c.max_batch, 0, 1);
        max_batch_ = c.max_batch;
        nvml_load();
        return true;
    }

    void shutdown() {
        llama_batch_free(batch_);
        if (ctx_)   llama_free(ctx_);
        if (model_) llama_model_free(model_);
        llama_backend_free();
    }

    int  add(Process * p) { procs_.push_back(p); retired_.push_back(false); return (int)procs_.size() - 1; }

    void retire(int h) {
        if (h < 0 || retired_[h]) return;
        std::vector<llama_seq_id> ids; procs_[h]->seq_ids(ids);
        for (auto s : ids) llama_memory_seq_rm(mem_, s, -1, -1);
        if (procs_[h]->on_retire) procs_[h]->on_retire();
        retired_[h] = true;
    }

    // One decode step. slot_budget caps total tokens. Returns step ms (or -1).
    double tick(int slot_budget, int & active) {
        std::vector<Request> reqs;
        std::vector<int> owner;
        for (int pi = 0; pi < (int)procs_.size(); ++pi) {
            if (retired_[pi] || procs_[pi]->done()) continue;
            int rem = slot_budget - (int)reqs.size();
            if (rem <= 0) break;
            int before = (int)reqs.size();
            procs_[pi]->want(rem, before, reqs);
            for (int k = before; k < (int)reqs.size(); ++k) owner.push_back(pi);
        }
        if (reqs.empty()) { active = 0; return 0.0; }

        for (int i = 0; i < (int)reqs.size(); ++i) {
            batch_.token[i]     = reqs[i].tok;
            batch_.pos[i]       = reqs[i].pos;
            batch_.n_seq_id[i]  = 1;
            batch_.seq_id[i][0] = reqs[i].seq;
            batch_.logits[i]    = reqs[i].want_logits ? 1 : 0;
        }
        batch_.n_tokens = (int)reqs.size();

        const auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx_, batch_) != 0) return -1.0;
        llama_synchronize(ctx_);
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        for (int i = 0; i < (int)reqs.size(); ++i)
            if (reqs[i].want_logits) procs_[owner[i]]->on_logits(i, llama_get_logits_ith(ctx_, i));

        active = (int)reqs.size();
        return dt;
    }

    // prefill a token list on a sequence, with logits on the last token
    bool prefill(llama_seq_id seq, const std::vector<llama_token> & toks, bool logits_last = true) {
        return decode_from(seq, toks, 0, logits_last);
    }

    // decode tokens starting at an explicit position
    bool decode_from(llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos pos0, bool logits_last = true) {
        for (int off = 0; off < (int)toks.size(); off += max_batch_) {
            int n = std::min(max_batch_, (int)toks.size() - off);
            for (int i = 0; i < n; ++i) {
                batch_.token[i] = toks[off + i];
                batch_.pos[i] = pos0 + (llama_pos)(off + i);
                batch_.n_seq_id[i] = 1;
                batch_.seq_id[i][0] = seq;
                batch_.logits[i] = (logits_last && off + i + 1 == (int)toks.size()) ? 1 : 0;
            }
            batch_.n_tokens = n;
            if (llama_decode(ctx_, batch_) != 0) return false;
        }
        return true;
    }

    llama_context * ctx() { return ctx_; }
    llama_memory_t  mem() { return mem_; }
    const llama_vocab * vocab() { return vocab_; }
    int n_vocab() const { return n_vocab_; }
    void seq_cp(llama_seq_id src, llama_seq_id dst) { llama_memory_seq_cp(mem_, src, dst, -1, -1); }
    void clear() { llama_memory_clear(mem_, true); }
    void reset_processes() { procs_.clear(); retired_.clear(); }
    void retire_seq(llama_seq_id s) { llama_memory_seq_rm(mem_, s, -1, -1); }
    void sync() { llama_synchronize(ctx_); }
    bool decode_requests(const std::vector<Request> & reqs) {
        for (int i = 0; i < (int)reqs.size(); ++i) {
            batch_.token[i] = reqs[i].tok; batch_.pos[i] = reqs[i].pos;
            batch_.n_seq_id[i] = 1; batch_.seq_id[i][0] = reqs[i].seq;
            batch_.logits[i] = reqs[i].want_logits ? 1 : 0;
        }
        batch_.n_tokens = (int)reqs.size();
        return llama_decode(ctx_, batch_) == 0;
    }
    const char * model_chat_template() { return llama_model_chat_template(model_, nullptr); }
    unsigned clock_sm() const { unsigned c = 0; if (nvml_get_clock_) nvml_get_clock_(nvml_dev_, 1, &c); return c; }

private:
    void nvml_load() {
        nvml_lib_ = LoadLibraryA("nvml.dll");
        if (!nvml_lib_) return;
        auto init = (int (*)(void)) GetProcAddress(nvml_lib_, "nvmlInit_v2");
        auto getH = (int (*)(unsigned, void **)) GetProcAddress(nvml_lib_, "nvmlDeviceGetHandleByIndex_v2");
        nvml_get_clock_ = (int (*)(void *, int, unsigned *)) GetProcAddress(nvml_lib_, "nvmlDeviceGetClockInfo");
        if (init && getH) { init(); getH(0, &nvml_dev_); }
    }

    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    llama_memory_t mem_ = nullptr;
    llama_batch batch_{};
    const llama_vocab * vocab_ = nullptr;
    int n_vocab_ = 0;
    int max_batch_ = 2048;
    std::vector<Process *> procs_;
    std::vector<bool> retired_;
    HMODULE nvml_lib_ = nullptr;
    void * nvml_dev_ = nullptr;
    int (*nvml_get_clock_)(void *, int, unsigned *) = nullptr;
};

} }  // namespace parallel::detail
