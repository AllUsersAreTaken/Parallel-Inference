// devops_agent.cpp -- local multi-agent pipeline over ONE shared context.
//
// Three agent slots (orchestrator / tool_worker / event_daemon) fork from a
// single prefilled workspace prefix via seq_cp, decode concurrently in one
// llama_decode per tick, and exchange results through host-side injection
// (decode_from), not HTTP round trips.
//
//   --mode serial        3 sequential calls, workspace re-prefilled each time
//   --mode concurrent    shared prefix, concurrent decode, host tool injection
//   --mode both          both, plus byte-equality check of the outputs
//
// The serial baseline is deliberately the FAIR competitor: same process, same
// model, persistent weights, no HTTP -- only what a conventional local agent
// runtime lacks is cross-call KV sharing and concurrent decode.

#include "backends/llama_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Engine = parallel::detail::Engine;
using Request = parallel::detail::Request;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::string jesc(const std::string & s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else o += c;
    }
    o += '"';
    return o;
}

// ---------------------------------------------------------------- model helpers
static std::string format_chat(Engine & eng, const std::string & sys, const std::string & user) {
    llama_chat_message msgs[2] = { { "system", sys.c_str() }, { "user", user.c_str() } };
    std::vector<char> buf(1 << 16);
    int n = llama_chat_apply_template(eng.model_chat_template(), msgs, 2, true, buf.data(), (int)buf.size());
    if (n > (int)buf.size()) {
        buf.resize((size_t)n);
        n = llama_chat_apply_template(eng.model_chat_template(), msgs, 2, true, buf.data(), (int)buf.size());
    }
    if (n > 0) return std::string(buf.data(), (size_t)std::min(n, (int)buf.size()));
    return sys + "\n\nUser: " + user + "\nAssistant:";
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), (int)text.size(), nullptr, 0, false, true);
    if (n < 0) n = -n;
    std::vector<llama_token> toks((size_t)n);
    if (n == 0) return toks;
    const int m = llama_tokenize(vocab, text.data(), (int)text.size(), toks.data(), n, false, true);
    toks.resize(m > 0 ? (size_t)m : 0);
    return toks;
}

static std::string token_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, false);
    if (n < 0) {
        std::vector<char> big((size_t)(-n));
        const int m = llama_token_to_piece(vocab, tok, big.data(), (int32_t)big.size(), 0, false);
        return m > 0 ? std::string(big.data(), (size_t)m) : std::string();
    }
    return n > 0 ? std::string(buf, (size_t)n) : std::string();
}

// greedy argmax over the raw logits of one row
static int pick_greedy(const float * lg, int n) {
    int bi = 0;
    float bv = lg[0];
    for (int i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; bi = i; }
    return bi;
}

// ---------------------------------------------------------------- host tool
// Whitelisted shell tool. The child inherits PATH (we prepend Git's usr/bin so
// `grep` resolves) and cwd (the repo root).
static bool run_tool(const std::string & cmd_in, std::string & out, double & ms_out) {
    std::string cmd = cmd_in;
    while (!cmd.empty() && (cmd.back() == ' ' || cmd.back() == '\r' || cmd.back() == '\n'))
        cmd.pop_back();
    while (!cmd.empty() && cmd.front() == ' ') cmd.erase(cmd.begin());
    auto starts = [](const std::string & s, const char * p) { return s.rfind(p, 0) == 0; };
    if (!(starts(cmd, "grep ") || starts(cmd, "findstr ") || starts(cmd, "python "))) {
        out = "[tool-error: command not permitted: only grep, findstr, python]\n";
        return false;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { out = "[tool-error: pipe]\n"; return false; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    std::string cl = "cmd.exe /c " + cmd + " < NUL 2>&1";
    std::vector<char> buf(cl.begin(), cl.end());
    buf.push_back(0);

    const auto t0 = Clock::now();
    BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); out = "[tool-error: spawn failed]\n"; ms_out = ms_since(t0); return false; }

    // drain while the child runs, with a hard timeout and output cap
    // (read into a temp buffer -- never write past the string's size())
    std::string all;
    char tmp[1024];
    const auto t_dead = t0 + std::chrono::milliseconds(3000);
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD want = std::min<DWORD>(avail, (DWORD)sizeof(tmp));
            if (all.size() + want > 4096) want = (DWORD)(4096 - all.size());
            if (want == 0) { TerminateProcess(pi.hProcess, 1); break; }
            DWORD got = 0;
            if (ReadFile(rd, tmp, want, &got, nullptr) && got) all.append(tmp, got);
        }
        DWORD st = WaitForSingleObject(pi.hProcess, 20);
        if (st == WAIT_OBJECT_0) {
            DWORD avail2 = 0;
            while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail2, nullptr) && avail2 > 0 &&
                   all.size() < 4096) {
                DWORD want = std::min<DWORD>(avail2, (DWORD)sizeof(tmp));
                if (all.size() + want > 4096) want = (DWORD)(4096 - all.size());
                DWORD got = 0;
                if (ReadFile(rd, tmp, want, &got, nullptr) && got) all.append(tmp, got);
                else break;
            }
            break;
        }
        if (Clock::now() > t_dead) { TerminateProcess(pi.hProcess, 1); break; }
        if (all.size() >= 4096) { TerminateProcess(pi.hProcess, 1); break; }
    }
    WaitForSingleObject(pi.hProcess, 500);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(rd);
    ms_out = ms_since(t0);
    if (all.empty()) all = "(no output)\n";
    out = all;
    return true;
}

// ---------------------------------------------------------------- workspace map
static std::string repo_workspace() {
    std::error_code ec;
    fs::path root = fs::current_path();
    std::vector<std::string> files;
    fs::recursive_directory_iterator it(root, ec), end;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        if (!it->is_regular_file(ec)) continue;
        std::string rel = fs::relative(it->path(), root, ec).generic_string();
        if (rel.empty()) continue;
        auto sk = [&rel](const char * p) { return rel.rfind(p, 0) == 0; };
        if (sk("build/") || sk("vendor/") || sk("data/") || sk("results/") ||
            sk(".opencode/") || sk(".git/") || sk("harness/__pycache__/")) continue;
        std::string ext = it->path().extension().string();
        if (ext == ".cpp" || ext == ".hpp" || ext == ".h" || ext == ".py" ||
            ext == ".bat" || ext == ".md")
            files.push_back(rel);
    }
    std::sort(files.begin(), files.end());
    if (files.size() > 60) files.resize(60);
    std::string out;
    for (auto & f : files) out += f + "\n";
    return out;
}

// ---------------------------------------------------------------- slots
enum class Kind { Orch, Tool, Dmon };

struct Slot {
    Kind         kind   = Kind::Orch;
    llama_seq_id seq    = 0;
    llama_pos    pos    = 0;
    int          pending = -1;   // sampled token, not yet written to KV
    bool         paused  = false;
    bool         awaiting = false;  // paused, waiting for an injection
    bool         marker  = false;   // role marker already seen (one-shot)
    bool         done    = false;
    int          max_tok = 0;
    int          min_tok = 0;      // EOS suppressed below this (floor clamp)
    std::string  stop_seq;      // strict stop-token (one-shot)
    bool         stop_json = false;  // stop when a {...} object balances
    bool         stop_hit  = false;
    int          ntok    = 0;
    std::string  text;
    std::string  finish;
    double       decode_ms = 0;
    int          ticks     = 0;
    double       inject_ms = 0;
    Clock::time_point t_marker{};   // WAIT / LISTEN detected
    Clock::time_point t_injected{}; // injection finished
    bool         resumed  = false;  // first post-injection token committed
};

struct Phase {
    std::string name;
    double prefill_ms = 0, decode_ms = 0, wall_ms = 0;
    int    ntok = 0;
};

struct Result {
    bool        ok = false;
    std::string err;
    std::vector<Phase> phases;          // serial only
    double prefill_ms = 0, decode_ms = 0, wall_ms = 0;
    double tool_ms = 0, inject_ms = 0;
    int    ticks = 0;
    int    agg_slots = 0;              // sum of active slots per tick -> mean concurrency
    std::vector<std::string> text, finish;
    std::vector<int> ntok;
    std::string tool_cmd, tool_out;
    double wait_ms = 0, resume_ms = 0;
};

// Role schedule for one run: per-slot prompts + budgets + scheduler behaviour.
// classic  = row-52 behaviour (worker dies at </tool>, long orchestrator)
// balanced = all three slots live across the same window (mean active >= 2.6)
struct Spec {
    std::vector<std::vector<llama_token>> roles;   // [orchestrator, tool_worker, event_daemon]
    int  mo = 160, mt = 60, md = 120;              // per-slot token budgets
    int  min_tok = 0;                              // per-slot floor (0 = off)
    bool worker_continues = false;                 // worker pauses at </tool> instead of dying
    bool dmon_json = false;                        // daemon: stop on a balanced {...} object
};

// Tolerant marker detection: a 3B model will happily drop the angle brackets
// off <<WAIT>> (it wrote "WAIT." on a line). Accept the exact token OR a short
// standalone line starting with the bare word.
static bool marker_seen(const std::string & text, const char * tok) {
    if (text.find(tok) != std::string::npos) return true;
    std::string bare(tok);
    if (bare.size() >= 4 && bare[0] == '<' && bare[1] == '<') {
        bare = bare.substr(2, bare.size() - 4);
        size_t st = 0;
        while (st < text.size()) {
            size_t e = text.find('\n', st);
            if (e == std::string::npos) e = text.size();
            std::string line = text.substr(st, e - st);
            while (!line.empty() && (line.front() == ' ' || line.front() == '\r')) line.erase(line.begin());
            if (line.rfind(bare, 0) == 0 && line.size() <= bare.size() + 8) return true;
            st = e + 1;
        }
    }
    return false;
}

// ---------------------------------------------------------------- decode ticks
// One batched llama_decode for the slots that can advance (not done, not
// paused, with a pending sampled token). Commits the pending token, checks
// role markers, then samples the next token from that row's logits.
static bool pump(Engine & eng, std::vector<Slot> & slots, const std::vector<int> & who,
                 const llama_vocab * vocab, int nv, Result & r, int max_ticks = 0,
                 bool worker_continues = false) {
    // max_ticks = 0: run until no slot can advance (serial phases).
    // max_ticks = 1: one forward pass per call, so the host can inject on the
    // very next tick instead of waiting for the whole cohort to drain.
    for (int guard = 0; ; ++guard) {
        if (max_ticks > 0 && guard >= max_ticks) return true;
        if (guard >= 800) { r.err = "tick guard exceeded"; return false; }
        std::vector<int> idx;
        for (int i : who)
            if (!slots[i].done && !slots[i].paused && slots[i].pending >= 0) idx.push_back(i);
        if (idx.empty()) return true;

        std::vector<Request> reqs;
        for (int k : idx) reqs.push_back({ slots[k].seq, slots[k].pos, slots[k].pending, true });

        const auto t0 = Clock::now();
        if (!eng.decode_requests(reqs)) { r.err = "decode failed"; return false; }
        eng.sync();
        const double dt = ms_since(t0);
        r.decode_ms += dt;
        r.ticks += 1;
        r.agg_slots += (int)idx.size();

        for (size_t k = 0; k < idx.size(); ++k) {
            Slot & s = slots[idx[k]];
            const float * lg = llama_get_logits_ith(eng.ctx(), (int)k);
            if (!lg) { r.err = "missing logits for row"; return false; }

            s.text += token_piece(vocab, s.pending);   // commit
            s.ntok += 1;
            s.pos += 1;
            s.decode_ms += dt;

            // One-shot role markers: after an injection the marker is still in
            // s.text, so re-detecting it would re-pause the slot forever.
            if (!s.marker && s.kind == Kind::Tool && s.text.find("</tool>") != std::string::npos) {
                s.marker = true;
                if (worker_continues) {
                    // stay alive: wait for the result, then analyse it (keeps
                    // the worker on screen alongside the orchestrator)
                    s.finish = "tool_cmd"; s.paused = true; s.awaiting = true;
                    s.pending = -1; s.t_marker = Clock::now();
                } else {
                    s.finish = "tool"; s.done = true; s.pending = -1;
                }
                continue;
            }
            if (!s.marker && s.kind == Kind::Orch && marker_seen(s.text, "<<WAIT>>")) {
                s.marker = true;
                s.finish = "wait"; s.paused = true; s.awaiting = true;
                s.pending = -1; s.t_marker = Clock::now(); continue;
            }
            if (!s.marker && s.kind == Kind::Dmon && marker_seen(s.text, "<<LISTEN>>")) {
                s.marker = true;
                s.finish = "listen"; s.paused = true; s.awaiting = true;
                s.pending = -1; s.t_marker = Clock::now(); continue;
            }
            // strict terminators: a stop-token and/or a balanced {...} object.
            // These fire regardless of role markers and bound runaway decoding.
            if (!s.stop_hit && !s.stop_seq.empty() && s.text.find(s.stop_seq) != std::string::npos) {
                s.stop_hit = true; s.finish = "stop"; s.done = true; s.pending = -1; continue;
            }
            if (!s.stop_hit && s.stop_json && s.ntok >= s.min_tok) {
                int depth = 0; bool started = false, closed = false;
                for (char c : s.text) {
                    if (c == '{') { depth++; started = true; }
                    else if (c == '}' && started) { depth--; if (depth <= 0) { closed = true; break; } }
                }
                if (closed) { s.stop_hit = true; s.finish = "stop"; s.done = true; s.pending = -1; continue; }
            }
            int nxt = pick_greedy(lg, nv);
            // Floor clamp: a slot that EOSes right after its injection (common)
            // ends the run early and caps mean occupancy. Suppress EOG until the
            // slot meets its minimum budget -- scheduling measurement only.
            if (s.ntok < s.min_tok && llama_vocab_is_eog(vocab, nxt)) {
                const llama_token e1 = llama_vocab_eos(vocab);
                const llama_token e2 = llama_vocab_eot(vocab);
                int bi = -1; float bv = -1.0e30f;
                for (int q = 0; q < nv; ++q) {
                    if (q == e1 || q == e2) continue;
                    if (lg[q] > bv) { bv = lg[q]; bi = q; }
                }
                if (bi >= 0) nxt = bi;
            }
            if (llama_vocab_is_eog(vocab, nxt)) { s.finish = "eos"; s.done = true; s.pending = -1; continue; }
            if (s.ntok >= s.max_tok) { s.finish = "max"; s.done = true; s.pending = -1; continue; }
            s.pending = nxt;
        }
    }
}

// Append a host-produced block into a slot (tool result / event) and resume it
// from the logits of the last injected token. This is the "KV block link":
// correct by construction, costs one decode over the slot's own context.
static bool inject_block(Engine & eng, Slot & s, const std::string & block,
                         const llama_vocab * vocab, int nv, Result & r) {
    std::vector<llama_token> toks = tokenize(vocab, block);
    if (toks.empty()) { r.err = "injection tokenized empty"; return false; }
    const auto t0 = Clock::now();
    if (!eng.decode_from(s.seq, toks, s.pos, true)) { r.err = "injection decode failed"; return false; }
    eng.sync();
    const float * p = llama_get_logits_ith(eng.ctx(), -1);
    if (!p) { r.err = "no logits after injection"; return false; }
    std::vector<float> lg(p, p + nv);
    s.pos += (llama_pos)toks.size();
    const int nxt = pick_greedy(lg.data(), nv);
    s.t_injected = Clock::now();
    s.inject_ms += ms_since(t0);
    s.finish.clear();
    if (llama_vocab_is_eog(vocab, nxt)) { s.done = true; s.finish = "eos"; s.pending = -1; }
    else s.pending = nxt;
    s.paused = false;
    s.awaiting = false;
    return true;
}

static bool prefill_ws(Engine & eng, Slot & s, const std::vector<llama_token> & w,
                       double & prefill_ms, Result & r) {
    const auto t0 = Clock::now();
    if (!eng.prefill(s.seq, w, false)) { r.err = "workspace prefill failed"; return false; }
    prefill_ms += ms_since(t0);
    s.pos = (llama_pos)w.size();
    return true;
}

static bool init_role(Engine & eng, Slot & s, const std::vector<llama_token> & role,
                      double & prefill_ms, const llama_vocab * vocab, int nv, Result & r) {
    const auto t0 = Clock::now();
    if (!eng.decode_from(s.seq, role, s.pos, true)) { r.err = "role prefill failed"; return false; }
    const float * p = llama_get_logits_ith(eng.ctx(), -1);
    if (!p) { r.err = "no logits after role prefill"; return false; }
    std::vector<float> lg(p, p + nv);
    prefill_ms += ms_since(t0);
    s.pos += (llama_pos)role.size();
    const int tok = pick_greedy(lg.data(), nv);
    if (llama_vocab_is_eog(vocab, tok)) { s.done = true; s.finish = "eos"; s.pending = -1; }
    else s.pending = tok;
    return true;
}

static void make_slots(std::vector<Slot> & slots, int mo = 160, int mt = 60,
                       int md = 120, bool dmon_json = false, int min_tok = 0) {
    slots.assign(3, Slot());
    slots[0].kind = Kind::Orch; slots[0].max_tok = mo;
    slots[1].kind = Kind::Tool; slots[1].max_tok = mt;
    slots[2].kind = Kind::Dmon; slots[2].max_tok = md;
    slots[2].stop_json = dmon_json;
    for (auto & s : slots) s.min_tok = min_tok;
}

static std::string result_block(const Result & r, int slot) {
    std::string b = "\n<result>\n" + r.tool_out + "</result>\n";
    // The continuation instruction lives HERE, not in the role prompt: a short
    // role prompt is what stops the worker from echoing instructions before it
    // emits <tool>, which was delaying the tool and stalling the orchestrator.
    if (slot == 1) {
        b += "Now write a line starting with \"FINDING: \", then 3 short lines of "
             "analysis, then a line starting with \"CONFIDENCE: \" (high, medium or low).\n";
    } else {
        b += "Now write exactly 3 bullet lines starting with \"- \" answering the task, "
             "then one line starting with \"NEXT: \".\n";
    }
    return b;
}
// Simulated daemon event (fixed string; this run is internal, no live bridge).
static std::string event_block() {
    return "\n<event>\nprocess \"worker-7\" exited with code 1 at 03:14:15\n</event>\n"
           "Now write a line starting with \"ACTION: \" and then exactly one line of JSON "
           "with keys id, event, severity, source, affected and text, all short strings.\n";
}

static bool take_tool(Slot & s, Result & r) {
    const size_t a = s.text.find("<tool>");
    const size_t b = s.text.find("</tool>");
    if (a != std::string::npos && b != std::string::npos && b > a + 6)
        r.tool_cmd = s.text.substr(a + 6, b - a - 6);
    double ms = 0;
    if (r.tool_cmd.empty()) r.tool_out = "[tool-error: no tool command emitted]\n";
    else run_tool(r.tool_cmd, r.tool_out, ms);
    r.tool_ms = ms;
    return true;
}

// ---------------------------------------------------------------- live sidecar
// The playground polls this file to stream slot text while the run is in
// flight. Never set it for benchmark runs -- a write per few ticks perturbs
// the timing we are measuring.
static std::string g_live_path;
static const char * SLOT_NAMES[3] = { "orchestrator", "tool_worker", "event_daemon" };

static void write_live(const char * phase, const std::vector<Slot> & slots) {
    if (g_live_path.empty()) return;
    std::string js = "{\"phase\":\"" + std::string(phase) + "\",\"slots\":[";
    for (size_t i = 0; i < slots.size() && i < 3; ++i) {
        if (i) js += ",";
        js += "{\"name\":\"";
        js += SLOT_NAMES[i];
        js += "\",\"ntok\":" + std::to_string(slots[i].ntok) +
              ",\"finish\":\"" + slots[i].finish + "\",\"text\":" + jesc(slots[i].text) + "}";
    }
    js += "]}";
    FILE * f = std::fopen(g_live_path.c_str(), "wb");
    if (!f) return;
    std::fwrite(js.data(), 1, js.size(), f);
    std::fclose(f);
}

// ---------------------------------------------------------------- mode: serial
static bool do_serial(Engine & eng, const std::vector<llama_token> & w,
                      const Spec & spec,
                      const llama_vocab * vocab, int nv, Result & r) {
    const auto t_all = Clock::now();
    std::vector<Slot> slots;
    make_slots(slots, spec.mo, spec.mt, spec.md, spec.dmon_json, spec.min_tok);
    const char * names[3] = { "B_orchestrator", "A_tool_worker", "C_event_daemon" };
    const int order[3] = { 1, 0, 2 };   // tool first, then orchestrator, then daemon
    bool tool_ran = false;

    for (int oi = 0; oi < 3; ++oi) {
        const int i = order[oi];
        const auto t0 = Clock::now();
        Phase ph;
        ph.name = names[i];
        eng.clear();
        Slot & s = slots[i];
        s.seq = 0; s.pos = 0;
        if (!prefill_ws(eng, s, w, ph.prefill_ms, r)) return false;
        if (!init_role(eng, s, spec.roles[i], ph.prefill_ms, vocab, nv, r)) return false;
        if (!pump(eng, slots, { i }, vocab, nv, r, 0, spec.worker_continues)) return false;
        if (i == 1 && spec.worker_continues) {
            if (s.awaiting) {
                take_tool(slots[1], r);          // worker produced the command
                tool_ran = true;
                if (!inject_block(eng, s, result_block(r, 1), vocab, nv, r)) return false;
                if (!pump(eng, slots, { i }, vocab, nv, r, 0, spec.worker_continues)) return false;
            } else if (!tool_ran) {
                take_tool(slots[1], r);          // never emitted </tool>: record it, keep flowing
                tool_ran = true;
            }
        } else if (s.awaiting) {
            const std::string blk = (i == 0) ? result_block(r, 0) : event_block();
            if (!inject_block(eng, s, blk, vocab, nv, r)) return false;
            if (!pump(eng, slots, { i }, vocab, nv, r, 0, spec.worker_continues)) return false;
        }
        if (i == 1 && !spec.worker_continues && s.done) take_tool(s, r);
        ph.wall_ms = ms_since(t0);
        ph.ntok = s.ntok;
        r.phases.push_back(ph);
        if (!g_live_path.empty()) write_live("serial", slots);
    }

    for (auto & ph : r.phases) r.prefill_ms += ph.prefill_ms;   // serial phases do not share
    r.wall_ms = ms_since(t_all);
    for (auto & s : slots) {
        r.text.push_back(s.text);
        r.finish.push_back(s.finish);
        r.ntok.push_back(s.ntok);
        r.inject_ms += s.inject_ms;
        if (s.t_injected.time_since_epoch().count() && s.t_marker.time_since_epoch().count())
            r.wait_ms = std::max(r.wait_ms,
                std::chrono::duration<double, std::milli>(s.t_injected - s.t_marker).count());
    }
    return true;
}

// ---------------------------------------------------------------- mode: concurrent
static bool do_concurrent(Engine & eng, const std::vector<llama_token> & w,
                          const Spec & spec,
                          const llama_vocab * vocab, int nv, Result & r) {
    const auto t_all = Clock::now();
    std::vector<Slot> slots;
    make_slots(slots, spec.mo, spec.mt, spec.md, spec.dmon_json, spec.min_tok);

    // 1. prefill the workspace ONCE, fork to the three agent slots
    {
        const auto t0 = Clock::now();
        if (!eng.prefill(0, w, false)) { r.err = "workspace prefill failed"; return false; }
        r.prefill_ms += ms_since(t0);
    }
    eng.seq_cp(0, 1);
    eng.seq_cp(0, 2);

    // 2. each slot decodes only its own role suffix (forked at the same pos)
    for (int i = 0; i < 3; ++i) {
        slots[i].seq = (llama_seq_id)i;
        slots[i].pos = (llama_pos)w.size();
        if (!init_role(eng, slots[i], spec.roles[i], r.prefill_ms, vocab, nv, r)) return false;
    }

    // 3. interleaved decode; the host reacts AFTER every forward pass
    bool tool_done = false, inj0 = false, inj1 = false, inj2 = false;
    int idle = 0;
    for (int tick = 0; tick < 800; ++tick) {
        std::vector<int> idx;
        for (int i = 0; i < 3; ++i)
            if (!slots[i].done && !slots[i].paused && slots[i].pending >= 0) idx.push_back(i);

        if (!idx.empty()) {
            if (!pump(eng, slots, idx, vocab, nv, r, 1, spec.worker_continues)) return false;
            idle = 0;
        } else {
            if (++idle > 4) { r.err = "no slot can advance"; return false; }
        }

        if ((tick & 3) == 0) write_live("decoding", slots);

        // host tool: the worker reached </tool> -> run it now (36 ms here)
        if (!tool_done && slots[1].marker) {
            take_tool(slots[1], r);
            tool_done = true;
        }
        if (!tool_done && slots[1].done) {   // never emitted </tool>: unblock anyway
            take_tool(slots[1], r);
            tool_done = true;
        }
        // background event: inject as soon as the daemon parks -- it must not
        // wait on the tool, that is the whole point of a background slot
        if (slots[2].awaiting && !inj2) {
            if (!inject_block(eng, slots[2], event_block(), vocab, nv, r)) return false;
            inj2 = true;
        }
        if (tool_done && slots[1].awaiting && !inj1) {
            if (!inject_block(eng, slots[1], result_block(r, 1), vocab, nv, r)) return false;
            inj1 = true;
        }
        if (tool_done && slots[0].awaiting && !inj0) {
            if (!inject_block(eng, slots[0], result_block(r, 0), vocab, nv, r)) return false;
            inj0 = true;
        }
        if (slots[0].done && slots[1].done && slots[2].done) break;
    }

    r.wall_ms = ms_since(t_all);
    Phase ph;
    ph.name = "concurrent";
    ph.prefill_ms = r.prefill_ms;
    ph.wall_ms = r.wall_ms;
    for (auto & s : slots) { ph.ntok += s.ntok; r.text.push_back(s.text); r.finish.push_back(s.finish); r.ntok.push_back(s.ntok); r.inject_ms += s.inject_ms; }
    r.phases.push_back(ph);
    if (!g_live_path.empty()) write_live("done", slots);
    if (slots[0].t_injected.time_since_epoch().count() && slots[0].t_marker.time_since_epoch().count())
        r.wait_ms = std::chrono::duration<double, std::milli>(slots[0].t_injected - slots[0].t_marker).count();
    return true;
}

// ---------------------------------------------------------------- main
int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // diagnostics must survive a crash
    std::string model = "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
    std::string mode  = "both";
    std::string roles_mode = "classic";
    std::string out_path = "results/devops_agent.json";

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (k == "--model") model = nx();
        else if (k == "--mode") mode = nx();
        else if (k == "--roles") roles_mode = nx();
        else if (k == "--live") g_live_path = nx();
        else if (k == "--out") out_path = nx();
        else { std::fprintf(stderr, "devops_agent: unknown argument: %s\n", k.c_str()); return 2; }
    }
    if (mode != "serial" && mode != "concurrent" && mode != "both") {
        std::fprintf(stderr, "devops_agent: --mode serial|concurrent|both\n");
        return 2;
    }
    if (roles_mode != "classic" && roles_mode != "balanced") {
        std::fprintf(stderr, "devops_agent: --roles classic|balanced\n");
        return 2;
    }

    // Git for Windows ships grep; make it resolvable for the host tool child
    char pbuf[32768];
    DWORD pn = GetEnvironmentVariableA("PATH", pbuf, (DWORD)sizeof(pbuf));
    if (pn > 0 && pn < (DWORD)sizeof(pbuf)) {
        std::string np = "C:\\Program Files\\Git\\usr\\bin;" + std::string(pbuf, pn);
        SetEnvironmentVariableA("PATH", np.c_str());
    }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }, nullptr);

    Engine eng;
    Engine::Config c;
    c.model = model;
    c.n_ctx = 4096;
    c.n_seq_max = 4;
    c.max_batch = 1024;
    c.ngl = 99;
    c.fa = true;
    if (!eng.init(c)) { std::fprintf(stderr, "devops_agent: init failed\n"); return 1; }
    std::printf("[stage] model loaded\n");

    const llama_vocab * vocab = eng.vocab();
    const int nv = eng.n_vocab();

    const std::string sys =
        "You are running inside a local multi-agent runtime with three slots that share one "
        "context window: orchestrator, tool_worker, event_daemon.\n"
        "Workspace file map (relative paths):\n" + repo_workspace();
    const std::string task =
        "Task: locate every place where req.shared_prefix is assigned an empty string in the "
        "CLI source, and report the file and line numbers.";

    const std::string w_str = format_chat(eng, sys, task);
    const std::vector<llama_token> w = tokenize(vocab, w_str);

    const std::vector<std::string> role_classic = {
        "<slot role=\"orchestrator\"> First output one line beginning with PLAN: . Then "
        "output a line containing exactly <<WAIT>> and stop. When a result block is injected "
        "later, output exactly 3 bullet lines, each starting with '- ', answering the task.\n",
        "<slot role=\"tool_worker\"> Output exactly one shell command wrapped as "
        "<tool>...</tool> and nothing else. Example: <tool>grep -rn \"needle\" app</tool>\n",
        "<slot role=\"event_daemon\"> First output a line containing exactly <<LISTEN>> and "
        "stop. When an event block is injected later, output exactly one line of JSON with "
        "keys event, severity and text.\n",
    };
    // balanced: comparable work in every slot so all three are live across the
    // same window -- equal token budgets, the worker survives its own tool call,
    // and the daemon's output is bounded by a balanced-{...} terminator.
    // NOTE: plain ROLE:/PHASE: prose, no <slot> style tags -- the 3B echoes
    // tags it sees, which burned ~20 tokens of the daemon on header gibberish.
    const std::vector<std::string> role_balanced = {
        "ROLE: orchestrator. FIRST write exactly three short sentences starting with "
        "\"PLAN: \". Then write one line containing exactly <<WAIT>>.\n",
        "ROLE: tool_worker. FIRST output one shell command of at most 60 characters "
        "wrapped as <tool>...</tool> and nothing before it. Keep it short and close the "
        "tag. Example: <tool>grep -rn X app</tool>\n",
        "ROLE: event_daemon. FIRST write one line containing exactly <<LISTEN>>.\n",
    };

    Spec spec;
    spec.roles.resize(3);
    const std::vector<std::string> & role_sel = (roles_mode == "balanced") ? role_balanced : role_classic;
    for (int i = 0; i < 3; ++i) spec.roles[i] = tokenize(vocab, role_sel[i]);
    if (roles_mode == "balanced") {
        spec.mo = 110; spec.mt = 110; spec.md = 110;
        spec.min_tok = 88;          // floor: above early-EOS, below natural stop
        spec.worker_continues = true;
        spec.dmon_json = true;
    }
    std::printf("roles mode: %s (budgets %d/%d/%d, worker_continues=%d, dmon_json=%d)\n",
                roles_mode.c_str(), spec.mo, spec.mt, spec.md,
                spec.worker_continues ? 1 : 0, spec.dmon_json ? 1 : 0);

    int prefill_serial = 0, prefill_conc = (int)w.size();
    for (int i = 0; i < 3; ++i) { prefill_serial += (int)w.size() + (int)spec.roles[i].size(); prefill_conc += (int)spec.roles[i].size(); }
    std::printf("workspace prefix %d tok; roles %d/%d/%d tok\n", (int)w.size(),
                (int)spec.roles[0].size(), (int)spec.roles[1].size(), (int)spec.roles[2].size());
    std::printf("prefill tokens: serial %d (3x workspace) vs concurrent %d (1x shared)\n",
                prefill_serial, prefill_conc);

    // ---- warmup (graph capture + memory alloc), untimed
    {
        Result wr;
        std::vector<Slot> slots;
        make_slots(slots);
        slots[0].max_tok = 8;
        if (!prefill_ws(eng, slots[0], w, wr.prefill_ms, wr) ||
            !init_role(eng, slots[0], spec.roles[0], wr.prefill_ms, vocab, nv, wr)) {
            std::fprintf(stderr, "devops_agent: warmup failed: %s\n", wr.err.c_str());
            return 1;
        }
        pump(eng, slots, { 0 }, vocab, nv, wr);
        eng.clear();
        std::printf("[stage] warmup ok\n");
    }

    Result rs, rc;
    if (mode == "serial" || mode == "both") {
        std::printf("[stage] serial start\n");
        if (!do_serial(eng, w, spec, vocab, nv, rs)) {
            std::fprintf(stderr, "devops_agent: serial failed: %s\n", rs.err.c_str());
            return 1;
        }
        eng.clear();
    }
    if (mode == "concurrent" || mode == "both") {
        std::printf("[stage] concurrent start\n");
        if (!do_concurrent(eng, w, spec, vocab, nv, rc)) {
            std::fprintf(stderr, "devops_agent: concurrent failed: %s\n", rc.err.c_str());
            return 1;
        }
        eng.clear();
    }

    const char * slot_names[3] = { "orchestrator", "tool_worker", "event_daemon" };
    auto res_json = [&](const Result & r, int ptoks) {
        std::string s = "{";
        s += "\"prefill_tokens\":" + std::to_string(ptoks);
        s += ",\"prefill_ms\":" + std::to_string(r.prefill_ms);
        s += ",\"decode_ms\":" + std::to_string(r.decode_ms);
        s += ",\"wall_ms\":" + std::to_string(r.wall_ms);
        s += ",\"ticks\":" + std::to_string(r.ticks);
        s += ",\"avg_active_slots\":" + std::to_string(r.ticks ? (double)r.agg_slots / r.ticks : 0.0);
        s += ",\"tool_ms\":" + std::to_string(r.tool_ms);
        s += ",\"inject_ms\":" + std::to_string(r.inject_ms);
        s += ",\"marker_to_inject_ms\":" + std::to_string(r.wait_ms);
        s += ",\"slots\":[";
        for (size_t i = 0; i < r.text.size(); ++i) {
            if (i) s += ",";
            s += "{\"name\":" + jesc(slot_names[i]) + ",\"ntok\":" + std::to_string(r.ntok[i]) +
                 ",\"finish\":" + jesc(r.finish[i]) + ",\"text\":" + jesc(r.text[i]) + "}";
        }
        s += "],\"phases\":[";
        for (size_t i = 0; i < r.phases.size(); ++i) {
            if (i) s += ",";
            s += "{\"name\":" + jesc(r.phases[i].name) + ",\"prefill_ms\":" +
                 std::to_string(r.phases[i].prefill_ms) + ",\"wall_ms\":" +
                 std::to_string(r.phases[i].wall_ms) + ",\"ntok\":" +
                 std::to_string(r.phases[i].ntok) + "}";
        }
        s += "]}";
        return s;
    };

    std::string js = "{\n  \"mode\": " + jesc(mode) + ",\n  \"model\": " + jesc(model) +
                     ",\n  \"prefix_tokens\": " + std::to_string(w.size());
    int identical = -1;
    if (mode == "serial" || mode == "both") js += ",\n  \"serial\": " + res_json(rs, prefill_serial);
    if (mode == "concurrent" || mode == "both") js += ",\n  \"concurrent\": " + res_json(rc, prefill_conc);
    js += ",\n  \"tool\": {\"cmd\": " + jesc(rs.tool_cmd.empty() ? rc.tool_cmd : rs.tool_cmd) +
           ", \"ms\": " + std::to_string(rc.tool_ms > 0 ? rc.tool_ms : rs.tool_ms);
    const std::string & tout = rs.tool_out.empty() ? rc.tool_out : rs.tool_out;
    js += ", \"out\": " + jesc(tout) + "}";
    if (mode == "both") {
        identical = 0;
        for (int i = 0; i < 3; ++i) if (i < (int)rs.text.size() && i < (int)rc.text.size() && rs.text[i] == rc.text[i]) identical++;
        js += ",\n  \"identical_slots\": " + std::to_string(identical);
        js += ",\n  \"wall_ratio_serial_over_concurrent\": " +
              std::to_string(rs.wall_ms / (rc.wall_ms > 1.0 ? rc.wall_ms : 1.0));
    }
    js += "\n}\n";

    FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "devops_agent: cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(js.data(), 1, js.size(), f);
    std::fclose(f);

    // ---- summary
    std::printf("\n%-12s %10s %10s %10s %8s %8s\n", "mode", "prefill", "decode", "wall", "ticks", "active");
    if (mode == "serial" || mode == "both")
        std::printf("%-12s %10.1f %10.1f %10.1f %8d %8.2f\n", "serial", rs.prefill_ms, rs.decode_ms,
                    rs.wall_ms, rs.ticks, rs.ticks ? (double)rs.agg_slots / rs.ticks : 0.0);
    if (mode == "concurrent" || mode == "both")
        std::printf("%-12s %10.1f %10.1f %10.1f %8d %8.2f\n", "concurrent", rc.prefill_ms, rc.decode_ms,
                    rc.wall_ms, rc.ticks, rc.ticks ? (double)rc.agg_slots / rc.ticks : 0.0);
    if (mode == "both") {
        std::printf("\nserial phases:\n");
        for (auto & p : rs.phases)
            std::printf("  %-16s prefill %7.1f  wall %7.1f  tokens %d\n",
                        p.name.c_str(), p.prefill_ms, p.wall_ms, p.ntok);
        std::printf("wall ratio serial/concurrent: %.2fx\n", rs.wall_ms / (rc.wall_ms > 1.0 ? rc.wall_ms : 1.0));
        std::printf("outputs identical: %d/3 slots\n", identical);
        std::printf("host tool: %.2f ms, %d bytes\n", rc.tool_ms > 0 ? rc.tool_ms : rs.tool_ms, (int)tout.size());
        std::printf("marker -> injection complete: %.1f ms (inject decode %.1f ms)\n",
                    rc.wait_ms, rc.inject_ms);
        std::printf("prefill tokens: %d serial vs %d concurrent\n", prefill_serial, prefill_conc);
    }
    std::printf("\nwrote %s\n", out_path.c_str());

    for (int i = 0; i < 3; ++i) {
        const Result & r = (mode == "concurrent") ? rc : rs;
        if (i >= (int)r.text.size()) continue;
        std::printf("\n--- %s [%s, %d tok]\n%s\n", slot_names[i], r.finish[i].c_str(),
                    r.ntok[i], r.text[i].c_str());
    }
    return 0;
}

