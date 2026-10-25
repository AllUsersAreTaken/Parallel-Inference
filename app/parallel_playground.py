#!/usr/bin/env python3
"""RecycleLM Playground - a small GUI for the parallel-stream runtime.

One model, 1-10 concurrent branches. Each branch gets its own prompt; the
backend batches the decodes on the GPU. Run compares each stream against its
own N=1 solo run; the footer shows the median latency overhead and the true
speedup (solo sum / batched wall). Scaling sweeps one prompt at N=1..10.

Run:  python app\\parallel_playground.py
"""
import json
import math
import os
import statistics as st
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "app", "parallel_cli.exe")
DEFAULT_MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
INPUT = os.path.join(ROOT, "app", "_gui_input.tsv")
OUTPUT = os.path.join(ROOT, "app", "_gui_out.json")
LOG = os.path.join(ROOT, "app", "_gui_run.log")
MEM_STATE = os.path.join(ROOT, "app", "memory_state.json")
# agents tab (app/devops_agent.exe): 3 slots over one shared KV prefix
DEVOPS = os.path.join(ROOT, "app", "devops_agent.exe")
AG_OUT = os.path.join(ROOT, "app", "_agents_out.json")
AG_LIVE = os.path.join(ROOT, "app", "_agents_live.json")
AG_LOG = os.path.join(ROOT, "app", "_agents_run.log")
AG_MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
AG_TASK = ("Task: locate every place where req.shared_prefix is assigned an empty string "
           "in the CLI source, and report the file and line numbers.")
AG_SLOT_NAMES = ("orchestrator", "tool_worker", "event_daemon")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_engine

BG = "#12141a"
CARD = "#1b1e27"
CARD2 = "#20242f"
FG = "#c0caf5"
MUTED = "#7a84a6"
ACCENT = "#7aa2f7"
GREEN = "#9ece6a"
RED = "#f7768e"
BORDER = "#2b3040"
MONO = ("Consolas", 9)
UI = ("Segoe UI", 10)
UI_SM = ("Segoe UI", 9)
UI_BOLD = ("Segoe UI", 10, "bold")

EXAMPLES = [
    "What is the capital of France?",
    "Explain photosynthesis in three sentences.",
    "Write a haiku about the ocean.",
    "Summarize the plot of Hamlet in two sentences.",
    "Give one practical tip for better sleep.",
    "What is 17 * 23?",
    "Name three prime numbers greater than 50.",
    "Explain why the sky is blue, briefly.",
    "Write a one-line slogan for a coffee shop.",
    "List two differences between TCP and UDP.",
]


class App:
    def __init__(self, root):
        self.root = root
        self.proc = None
        self.started = 0.0
        self.rows = []
        self.running = False
        self.ag_proc = None
        self.ag_started = 0.0
        self.ag_running = False
        self.ag_logf = None
        self.ag_cards = []

        root.title("RecycleLM Playground - parallel streams")
        root.geometry("1240x900")
        root.minsize(980, 700)
        root.configure(bg=BG)
        self._style()
        self._build()
        self._rebuild_rows()

    # ---------- theme ----------
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, font=UI)
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD, relief="flat")
        st.configure("TLabel", background=BG, foreground=FG, font=UI)
        st.configure("Card.TLabel", background=CARD, foreground=FG, font=UI)
        st.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 16, "bold"))
        st.configure("Muted.TLabel", background=BG, foreground=MUTED, font=UI_SM)
        st.configure("Stream.TLabel", background=CARD, foreground=ACCENT, font=UI_BOLD)
        st.configure("Tok.TLabel", background=CARD, foreground=GREEN, font=UI_SM)
        st.configure("Meta.TLabel", background=CARD, foreground=MUTED, font=UI_SM)
        st.configure("TButton", background=CARD2, foreground=FG, borderwidth=0,
                     focusthickness=0, padding=(12, 6), font=UI)
        st.map("TButton", background=[("active", "#2c3243"), ("disabled", CARD)],
               foreground=[("disabled", MUTED)])
        st.configure("Accent.TButton", background=ACCENT, foreground="#0b1020",
                     font=("Segoe UI", 10, "bold"), padding=(16, 7))
        st.map("Accent.TButton", background=[("active", "#9ab8ff"), ("disabled", CARD2)],
               foreground=[("disabled", MUTED)])
        st.configure("TEntry", fieldbackground=CARD2, foreground=FG, insertcolor=FG,
                     borderwidth=0, padding=6)
        st.configure("TSpinbox", fieldbackground=CARD2, foreground=FG, arrowcolor=FG,
                     borderwidth=0, padding=4)
        st.configure("TCombobox", fieldbackground=CARD2, foreground=FG)
        st.configure("Vertical.TScrollbar", background=CARD2, troughcolor=BG, borderwidth=0,
                     arrowcolor=MUTED)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=CARD2, foreground=MUTED,
                     padding=(14, 6), font=UI)
        st.map("TNotebook.Tab", background=[("selected", CARD)],
               foreground=[("selected", ACCENT)])
        st.configure("Memory.Treeview", background=CARD2, foreground=FG,
                     fieldbackground=CARD2, borderwidth=0, rowheight=22, font=UI_SM)
        st.configure("Memory.Treeview.Heading", background=CARD, foreground=MUTED,
                     font=UI_SM)

    # ---------- layout ----------
    def _build(self):
        top = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        top.pack(fill="x")
        ttk.Label(top, text="RecycleLM Playground", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="  one model  ·  N concurrent streams  ·  batched decode",
                  style="Muted.TLabel").pack(side="left", padx=(8, 0))

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=18, pady=(0, 4))
        streams_tab = ttk.Frame(self.nb)
        mem_tab = ttk.Frame(self.nb)
        self.nb.add(streams_tab, text="  Streams  ")
        self.nb.add(mem_tab, text="  Memory  ")
        agents_tab = ttk.Frame(self.nb)
        self.nb.add(agents_tab, text="  Agents  ")

        controls = ttk.Frame(streams_tab, padding=(0, 12, 0, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Streams", style="Muted.TLabel").pack(side="left")
        self.n_var = tk.IntVar(value=3)
        self.n_spin = ttk.Spinbox(controls, from_=1, to=10, width=4, textvariable=self.n_var,
                                  command=self._rebuild_rows)
        self.n_spin.pack(side="left", padx=(6, 14))
        ttk.Label(controls, text="Max tokens", style="Muted.TLabel").pack(side="left")
        self.max_var = tk.IntVar(value=256)
        ttk.Spinbox(controls, from_=16, to=1024, increment=16, width=6,
                    textvariable=self.max_var).pack(side="left", padx=(6, 14))
        ttk.Label(controls, text="Temp", style="Muted.TLabel").pack(side="left")
        self.temp_var = tk.StringVar(value="0.7")
        ttk.Spinbox(controls, from_=0.0, to=1.5, increment=0.05, width=5,
                    textvariable=self.temp_var).pack(side="left", padx=(6, 14))
        ttk.Label(controls, text="Seed", style="Muted.TLabel").pack(side="left")
        self.seed_var = tk.IntVar(value=1234)
        ttk.Spinbox(controls, from_=0, to=999999, width=7,
                    textvariable=self.seed_var).pack(side="left", padx=(6, 14))
        self.run_btn = ttk.Button(controls, text="Run", style="Accent.TButton",
                                  command=self._run)
        self.run_btn.pack(side="left", padx=(4, 6))
        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Copy #1 to all", command=self._copy).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Clear", command=self._clear).pack(side="left")
        self.scale_btn = ttk.Button(controls, text="Scaling N=1..10", command=self._scaling)
        self.scale_btn.pack(side="left", padx=(6, 0))

        sysrow = ttk.Frame(streams_tab, padding=(0, 0, 0, 8))
        sysrow.pack(fill="x")
        ttk.Label(sysrow, text="System", style="Muted.TLabel").pack(side="left")
        self.sys_var = tk.StringVar(value="You are a helpful assistant.")
        ttk.Entry(sysrow, textvariable=self.sys_var).pack(side="left", fill="x", expand=True,
                                                         padx=(8, 0))

        modelrow = ttk.Frame(streams_tab, padding=(0, 0, 0, 10))
        modelrow.pack(fill="x")
        ttk.Label(modelrow, text="Model", style="Muted.TLabel").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(modelrow, textvariable=self.model_var).pack(side="left", fill="x", expand=True,
                                                             padx=(8, 6))
        ttk.Button(modelrow, text="Browse", command=self._browse).pack(side="left")

        body = ttk.Frame(streams_tab, padding=(0, 0, 0, 0))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=BG, highlightthickness=0, bd=0)
        self.scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self.inner_id, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._wheel)

        self._build_memory(mem_tab)
        self._build_agents(agents_tab)

        footer = ttk.Frame(self.root, padding=(18, 8, 18, 12))
        footer.pack(fill="x")
        self.overall_var = tk.StringVar(value="idle - run to see timings")
        ttk.Label(footer, textvariable=self.overall_var, style="Muted.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="ready")
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").pack(side="right")

    def _wheel(self, event):
        try:
            if self.nb.index(self.nb.select()) != 0:
                return
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
        except tk.TclError:
            pass

    # ---------- rows ----------
    def _rebuild_rows(self):
        if self.running:
            return
        n = max(1, min(10, int(self.n_var.get() or 1)))
        old = [self._read_prompt(r) for r in self.rows]
        for r in self.rows:
            r["frame"].destroy()
        self.rows = []
        for i in range(n):
            prompt = old[i] if i < len(old) else (EXAMPLES[i] if i < len(EXAMPLES) else "")
            self._make_row(i, prompt)
        self.overall_var.set("idle - run to see timings")

    def _make_row(self, i, prompt):
        card = ttk.Frame(self.inner, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=(0, 10), padx=2)
        head = ttk.Frame(card, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 6))
        ttk.Label(head, text=f"Stream {i + 1}", style="Stream.TLabel").pack(side="left")
        ttk.Label(head, text=f"seed {self.seed_var.get() + i}", style="Meta.TLabel").pack(
            side="left", padx=(10, 0))
        tok = ttk.Label(head, text="", style="Tok.TLabel")
        tok.pack(side="right")

        prompt_box = tk.Text(card, height=2, wrap="word", bg=CARD2, fg=FG,
                             insertbackground=FG, relief="flat", font=UI,
                             padx=8, pady=6, highlightthickness=0)
        prompt_box.insert("1.0", prompt)
        prompt_box.pack(fill="x", pady=(0, 6))

        out = tk.Text(card, height=7, wrap="word", bg=BG, fg=FG,
                      insertbackground=FG, relief="flat", font=MONO,
                      padx=8, pady=6, highlightthickness=0)
        out.configure(state="disabled")
        out.pack(fill="x")
        meta = ttk.Label(card, text="", style="Meta.TLabel")
        meta.pack(fill="x", pady=(4, 0))

        self.rows.append({"frame": card, "prompt": prompt_box, "out": out,
                          "tok": tok, "meta": meta})

    def _read_prompt(self, row):
        return row["prompt"].get("1.0", "end-1c").strip()

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Choose GGUF model", initialdir=os.path.join(ROOT, "models"),
            filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")])
        if p:
            self.model_var.set(p)

    def _copy(self):
        if not self.rows:
            return
        first = self._read_prompt(self.rows[0])
        for r in self.rows[1:]:
            r["prompt"].delete("1.0", "end")
            r["prompt"].insert("1.0", first)

    def _clear(self):
        for r in self.rows:
            r["out"].configure(state="normal")
            r["out"].delete("1.0", "end")
            r["out"].configure(state="disabled")
            r["tok"].configure(text="")
            r["meta"].configure(text="")
        self.overall_var.set("idle - run to see timings")
        self.status_var.set("ready")

    def _all_run_state(self, state):
        for b in (self.run_btn, self.scale_btn, *self.mem_buttons):
            b.configure(state=state)

    # ---------- run ----------
    def _run(self):
        if self.running:
            return
        if not os.path.exists(CLI):
            messagebox.showerror("Playground", f"Backend not found:\n{CLI}")
            return
        prompts = [self._read_prompt(r) for r in self.rows]
        if not any(prompts):
            messagebox.showwarning("Playground", "Enter at least one prompt.")
            return
        model = self.model_var.get().strip()
        if not os.path.exists(model):
            messagebox.showerror("Playground", f"Model not found:\n{model}")
            return
        n = len(self.rows)
        with open(INPUT, "w", encoding="utf-8", newline="\n") as f:
            f.write("id\tprompt\n")
            for i, p in enumerate(prompts):
                f.write(f"{i + 1}\t{p.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
        if os.path.exists(OUTPUT):
            os.remove(OUTPUT)

        env = dict(os.environ)
        env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
        args = [CLI, "--model", model, "--data", INPUT, "--out", OUTPUT,
                "--mode", "both",
                "--max-tokens", str(self.max_var.get()),
                "--temperature", str(self.temp_var.get()),
                "--top-p", "0.9",
                "--seed", str(self.seed_var.get()),
                "--system", self.sys_var.get()]
        for r in self.rows:
            r["tok"].configure(text="")
            r["meta"].configure(text="")
            r["out"].configure(state="normal")
            r["out"].delete("1.0", "end")
            r["out"].configure(state="disabled")

        self.logf = open(LOG, "w", encoding="utf-8")
        self.proc = subprocess.Popen(args, cwd=ROOT, env=env,
                                     stdout=subprocess.DEVNULL,
                                     stderr=self.logf,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.started = time.perf_counter()
        self.running = True
        self._all_run_state("disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set(f"running {n} streams...")
        self.overall_var.set("running...")
        self.root.after(120, self._poll)

    def _poll(self):
        if not self.running:
            return
        if self.proc.poll() is None:
            self.status_var.set(
                f"running {len(self.rows)} streams... {time.perf_counter() - self.started:.1f}s")
            self.root.after(120, self._poll)
            return

        wall = time.perf_counter() - self.started
        rc = self.proc.returncode
        self._close_log()
        self.running = False
        self._all_run_state("normal")
        self.stop_btn.configure(state="disabled")

        if rc != 0 or not os.path.exists(OUTPUT):
            self.status_var.set(f"backend failed (exit {rc}) - see {os.path.basename(LOG)}")
            self.overall_var.set("error - check the backend log")
            return
        try:
            data = json.load(open(OUTPUT, encoding="utf-8"))
        except Exception as exc:
            self.status_var.set(f"bad backend JSON: {exc}")
            return
        self._show(data, wall)

    def _show(self, data, wall):
        streams = data.get("streams", [])
        solo = data.get("serial", [])
        gen = 0
        serial_ms = 0.0
        toks = []
        solo_sum = 0.0
        over = []
        ttft_over = []
        for i, s in enumerate(streams[:len(self.rows)]):
            r = self.rows[i]
            text = s.get("text", "")
            r["out"].configure(state="normal")
            r["out"].delete("1.0", "end")
            r["out"].insert("1.0", text)
            r["out"].configure(state="disabled")
            g = int(s.get("generated", 0))
            tps = float(s.get("tok_s", 0))
            comp = float(s.get("completion_ms", 0)) / 1000.0
            ttft = float(s.get("ttft_ms", 0))
            gen += g
            serial_ms += float(s.get("completion_ms", 0))
            toks.append(tps)
            if i < len(solo):
                sc = float(solo[i].get("completion_ms", 0)) / 1000.0
                stt = float(solo[i].get("ttft_ms", 0))
                solo_sum += float(solo[i].get("completion_ms", 0))
                oh = (comp / sc - 1.0) * 100.0 if sc > 0 else 0.0
                th = (ttft / stt - 1.0) * 100.0 if stt > 0 else 0.0
                over.append(oh)
                ttft_over.append(th)
                extra = f"   ·   solo {sc:.2f} s ({oh:+.0f}%)"
            else:
                extra = ""
            r["tok"].configure(text=f"{tps:.1f} tok/s")
            r["meta"].configure(
                text=f"{comp:.2f} s batched{extra}   ·   {g} tokens   ·   "
                     f"ttft {ttft:.0f} ms   ·   {s.get('finish', '')}")
        wall_ms = float(data.get("wall_ms", wall * 1000.0))
        wall_s = wall_ms / 1000.0
        agg = (gen / wall_s) if wall_s else 0.0
        overlap = (serial_ms / wall_ms) if wall_ms else 0.0
        speed = (solo_sum / wall_ms) if (wall_ms and solo_sum) else 0.0
        joint = int(data.get("joint_steps", 0))
        steps = int(data.get("steps", 0))
        jr = (100.0 * joint / steps) if steps else 0.0
        avg = (sum(toks) / len(toks)) if toks else 0.0
        med_oh = st.median(over) if over else 0.0
        med_tt = st.median(ttft_over) if ttft_over else 0.0
        self.overall_var.set(
            f"wall {wall_s:.2f} s   ·   solo sum {solo_sum / 1000.0:.2f} s   ·   "
            f"speedup {speed:.2f}x   ·   median latency {med_oh:+.0f}% "
            f"(ttft {med_tt:+.0f}%)   ·   overlap {overlap:.2f}x   ·   "
            f"aggregate {agg:.1f} tok/s   ·   joint {joint}/{steps} ({jr:.0f}%)   ·   "
            f"{len(streams)} streams   ·   {gen} tokens")
        self.status_var.set(f"done in {wall_s:.2f} s")

    def _scaling(self):
        if self.running or not self.rows:
            return
        prompt = self._read_prompt(self.rows[0])
        if not prompt:
            messagebox.showwarning("Playground", "Stream 1 needs a prompt for the sweep.")
            return
        model = self.model_var.get().strip()
        if not os.path.exists(model):
            messagebox.showerror("Playground", f"Model not found:\n{model}")
            return
        path = os.path.join(ROOT, "app", "_gui_sweep_input.tsv")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("id\tprompt\n1\t" + prompt.replace("\t", " ").replace("\n", " ") + "\n")
        out = os.path.join(ROOT, "app", "_gui_sweep.json")
        if os.path.exists(out):
            os.remove(out)

        win = tk.Toplevel(self.root)
        win.title("Latency vs N - same prompt")
        win.configure(bg=BG)
        win.geometry("900x430")
        cols = ("n", "wall", "lat", "ttft", "tps", "agg", "latx", "thrx")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=10)
        heads = {"n": "N", "wall": "wall s", "lat": "avg latency s", "ttft": "TTFT ms",
                 "tps": "stream tok/s", "agg": "aggregate tok/s",
                 "latx": "latency vs N=1", "thrx": "throughput vs N=1"}
        widths = {"n": 40, "wall": 80, "lat": 110, "ttft": 80, "tps": 110,
                  "agg": 120, "latx": 120, "thrx": 140}
        for c in cols:
            tree.heading(c, text=heads[c])
            tree.column(c, width=widths[c], anchor="center")
        tree.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        ttk.Label(win, text="One prompt replicated at each N. Latency = per-stream "
                            "completion; N=1 includes warmup.",
                  style="Muted.TLabel").pack(anchor="w", padx=12, pady=(0, 10))

        self.running = True
        self._all_run_state("disabled")
        self.status_var.set("scaling 1..10...")
        env = dict(os.environ)
        env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
        args = [CLI, "--model", model, "--data", path, "--out", out,
                "--mode", "sweep", "--sweep-max", "10",
                "--max-tokens", str(self.max_var.get()),
                "--temperature", str(self.temp_var.get()),
                "--top-p", "0.9", "--seed", str(self.seed_var.get()),
                "--system", self.sys_var.get()]

        def work():
            try:
                subprocess.run(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                               timeout=900)
                data = json.load(open(out, encoding="utf-8"))
            except Exception as exc:
                self.root.after(0, lambda: self._scale_error(win, str(exc)))
                return
            self.root.after(0, lambda: self._show_sweep(win, tree, data))

        threading.Thread(target=work, daemon=True).start()

    def _scale_error(self, win, msg):
        self.running = False
        self._all_run_state("normal")
        self.status_var.set("scaling failed")
        messagebox.showerror("Playground", msg, parent=win)

    def _show_sweep(self, win, tree, data):
        rows = data.get("sweep", [])
        base = rows[0] if rows else None
        for s in rows:
            comp = float(s.get("avg_completion_ms", 0))
            ttft = float(s.get("avg_ttft_ms", 0))
            agg = float(s.get("aggregate_tok_s", 0))
            latx = comp / base["avg_completion_ms"] if base and base["avg_completion_ms"] else 1.0
            thrx = agg / base["aggregate_tok_s"] if base and base["aggregate_tok_s"] else 1.0
            tree.insert("", "end", values=(
                s["n"], f"{s['wall_ms'] / 1000:.3f}", f"{comp / 1000:.3f}",
                f"{ttft:.0f}", f"{s['avg_tok_s']:.1f}", f"{agg:.1f}",
                f"{latx:.2f}x", f"{thrx:.2f}x"))
        self.running = False
        self._all_run_state("normal")
        self.status_var.set("scaling done")

    # ---------- memory tab ----------
    def _build_memory(self, tab):
        frozen = ttk.Frame(tab, padding=(0, 10, 0, 0))
        frozen.pack(fill="x")
        ttk.Label(frozen,
                  text=f"frozen k=3 recipe  ·  {os.path.basename(memory_engine.MODEL)}  ·  "
                       f"temp 0.2  ·  seed 1234  ·  max 256  ·  "
                       f"state shared with the CLI (app/memory_state.json)",
                  style="Muted.TLabel").pack(side="left")

        mrow = ttk.Frame(tab, padding=(0, 8, 0, 8))
        mrow.pack(fill="x")
        self.mem_input = tk.Text(mrow, height=2, wrap="word", bg=CARD2, fg=FG,
                                 insertbackground=FG, relief="flat", font=UI,
                                 padx=8, pady=6, highlightthickness=0)
        self.mem_input.pack(side="left", fill="x", expand=True, padx=(0, 8))
        btns = ttk.Frame(mrow)
        btns.pack(side="right")
        self.mem_send_btn = ttk.Button(btns, text="Send", style="Accent.TButton",
                                       command=self._mem_send)
        self.mem_send_btn.pack(side="left", padx=(0, 6))
        self.mem_demo_btn = ttk.Button(btns, text="Demo turn 1/2", command=self._mem_demo)
        self.mem_demo_btn.pack(side="left", padx=(0, 6))
        self.mem_reset_btn = ttk.Button(btns, text="Reset memory", command=self._mem_reset)
        self.mem_reset_btn.pack(side="left")
        self.mem_buttons = [self.mem_send_btn, self.mem_demo_btn, self.mem_reset_btn]

        paned = ttk.Panedwindow(tab, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(0, 4))
        left = ttk.Frame(paned)
        mid = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(mid, weight=1)
        paned.add(right, weight=1)

        ttk.Label(left, text="Assistant reply", style="Stream.TLabel").pack(anchor="w")
        self.mem_reply = tk.Text(left, height=12, wrap="word", bg=CARD, fg=FG,
                                 relief="flat", font=UI, padx=10, pady=8,
                                 highlightthickness=0, state="disabled")
        self.mem_reply.pack(fill="both", expand=True, pady=(4, 4))
        self.mem_meta = ttk.Label(left, text="", style="Meta.TLabel")
        self.mem_meta.pack(anchor="w")

        ttk.Label(mid, text="Memory graph", style="Stream.TLabel").pack(anchor="w")
        self.mem_canvas = tk.Canvas(mid, bg=CARD, highlightthickness=0, bd=0)
        self.mem_canvas.pack(fill="both", expand=True, pady=(4, 0))
        self.mem_canvas.bind("<Configure>", lambda e: self._draw_graph())

        ttk.Label(right, text="Graph edges", style="Stream.TLabel").pack(anchor="w")
        cols = ("key", "object", "valid_from", "status")
        self.mem_tree = ttk.Treeview(right, columns=cols, show="headings", height=5,
                                     style="Memory.Treeview")
        for c, h, w in (("key", "relation", 150), ("object", "object", 170),
                        ("valid_from", "valid from", 140), ("status", "status", 90)):
            self.mem_tree.heading(c, text=h)
            self.mem_tree.column(c, width=w, anchor="w")
        self.mem_tree.tag_configure("old", foreground=MUTED)
        self.mem_tree.pack(fill="x", pady=(4, 8))

        ttk.Label(right, text="Facts (last turn)", style="Stream.TLabel").pack(anchor="w")
        self.mem_facts = tk.Text(right, height=4, wrap="word", bg=CARD, fg=GREEN,
                                 relief="flat", font=MONO, padx=8, pady=6,
                                 highlightthickness=0, state="disabled")
        self.mem_facts.pack(fill="x", pady=(4, 8))

        ttk.Label(right, text="Indexed queries", style="Stream.TLabel").pack(anchor="w")
        self.mem_queries = tk.Listbox(right, height=5, bg=CARD, fg=ACCENT,
                                      relief="flat", font=MONO, highlightthickness=0,
                                      selectbackground=CARD2, exportselection=False)
        self.mem_queries.pack(fill="x", pady=(4, 8))

        logrow = ttk.Frame(right)
        logrow.pack(fill="both", expand=True)
        ttk.Label(logrow, text="Event log", style="Stream.TLabel").pack(anchor="w")
        self.mem_logw = tk.Text(logrow, height=7, wrap="word", bg=BG, fg=MUTED,
                                relief="flat", font=MONO, padx=8, pady=6,
                                highlightthickness=0, state="disabled")
        self.mem_logw.tag_configure("add", foreground=GREEN)
        self.mem_logw.tag_configure("close", foreground=RED)
        self.mem_logw.tag_configure("conflict", foreground="#e0af68")
        self.mem_logw.tag_configure("query", foreground=ACCENT)
        self.mem_logw.tag_configure("err", foreground=RED)
        self.mem_logw.tag_configure("head", foreground=FG)
        self.mem_logw.pack(fill="both", expand=True, pady=(4, 0))

        self.demo_i = 0
        self._mem_load_state()

    def _text_set(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _mem_log(self, msg, tag=None):
        self.mem_logw.configure(state="normal")
        if tag:
            self.mem_logw.insert("end", msg + "\n", tag)
        else:
            self.mem_logw.insert("end", msg + "\n")
        self.mem_logw.see("end")
        self.mem_logw.configure(state="disabled")

    def _refresh_graph(self):
        self.mem_tree.delete(*self.mem_tree.get_children())
        for k, versions in self.mem_state.get("edges", {}).items():
            for e in versions:
                old = e.get("invalid_at") is not None
                self.mem_tree.insert("", "end", tags=("old",) if old else (),
                                     values=(k, e.get("object", ""),
                                             e.get("valid_from", ""),
                                             "superseded" if old else "open"))
        self._draw_graph()

    @staticmethod
    def _short(s, n=14):
        return s if len(s) <= n else s[: n - 1] + "…"

    def _draw_graph(self):
        """Entity graph: User at center, entities on a ring, edges node-to-node.

        Handles scoped subjects (Rule 5 amendment): [pi4, RUNS, Plex] draws a
        line from the pi4 node to the Plex node, not a User-centered spoke.
        Open edges solid green; superseded versions dim dashed ghost lines.
        """
        c = self.mem_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            return
        cx, cy = w / 2.0, h / 2.0
        edges = self.mem_state.get("edges", {})

        order = []
        nodes = {}
        draws = []  # (subj_l, obj_l, pred, is_open)

        def node(lk, display):
            if lk not in nodes:
                nodes[lk] = {"name": display, "open": False}
                order.append(lk)

        for k, versions in edges.items():
            parts = k.split("|")
            subj_l = parts[0]
            pred = parts[1] if len(parts) > 1 else k
            node(subj_l, "User" if subj_l == "user" else subj_l)
            if any(e.get("invalid_at") is None for e in versions):
                nodes[subj_l]["open"] = True
            for e in versions:
                obj = e.get("object", "?")
                lk = obj.lower()
                node(lk, obj)
                is_open = e.get("invalid_at") is None
                if is_open:
                    nodes[lk]["open"] = True
                draws.append((subj_l, lk, pred, is_open))

        # 'user' (or first subject) anchors the center; everyone else on a ring
        center = "user" if "user" in nodes else (order[0] if order else None)
        ring_order = [lk for lk in order if lk != center]
        n = len(ring_order)
        r_center, r_node = 26, 22
        ring = max(min(w, h) / 2.0 - 52, 74)
        pos = {}
        if center is not None:
            pos[center] = (cx, cy)
        for i, lk in enumerate(ring_order):
            ang = -math.pi / 2 + 2 * math.pi * i / max(n, 1)
            pos[lk] = (cx + ring * math.cos(ang), cy + ring * math.sin(ang))

        # edges first (under nodes); label sits at the line midpoint,
        # offset per duplicate (s, o) pair so stacked predicates stay legible
        seen_pairs = {}
        for subj_l, obj_l, pred, is_open in draws:
            if subj_l == obj_l or subj_l not in pos or obj_l not in pos:
                continue
            x1, y1 = pos[subj_l]
            x2, y2 = pos[obj_l]
            dx, dy = x2 - x1, y2 - y1
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dist, dy / dist
            r1 = r_center if subj_l == center else r_node
            r2 = r_center if obj_l == center else r_node
            ax, ay = x1 + ux * (r1 + 2), y1 + uy * (r1 + 2)
            bx, by = x2 - ux * (r2 + 2), y2 - uy * (r2 + 2)
            tag = "open" if is_open else "old"
            c.create_line(ax, ay, bx, by, fill=GREEN if is_open else MUTED,
                          dash=() if is_open else (3, 3),
                          width=2 if is_open else 1, tags=(tag,))
            pair = (subj_l, obj_l)
            idx = seen_pairs.get(pair, 0)
            seen_pairs[pair] = idx + 1
            t = 0.42 + 0.14 * idx
            lx, ly = x1 + dx * t, y1 + dy * t
            c.create_text(lx + 12, ly - 10, text=self._short(pred.replace("_", " ")),
                          fill=ACCENT if is_open else MUTED,
                          font=("Segoe UI", 8), tags=(tag,))

        # nodes
        for lk in order:
            x, y = pos[lk]
            is_center = lk == center
            r = r_center if is_center else r_node
            is_open = nodes[lk]["open"] or is_center
            c.create_oval(x - r, y - r, x + r, y + r,
                          fill=CARD2, outline=ACCENT if is_open else MUTED,
                          width=2 if is_open else 1,
                          dash=() if is_open else (2, 2))
            c.create_text(x, y, text=self._short(nodes[lk]["name"], 12),
                          fill=FG if is_open else MUTED,
                          font=("Segoe UI", 9 if is_center else 8,
                                "bold" if is_center else "normal"))

        if not order:
            c.create_text(cx, cy + 48, text="no edges yet - send a turn",
                          fill=MUTED, font=("Segoe UI", 8))

    def _mem_load_state(self):
        state = {"history": [], "edges": {}, "queries": []}
        if os.path.exists(MEM_STATE):
            try:
                state = json.load(open(MEM_STATE, encoding="utf-8"))
            except Exception:
                pass
        # older state files stored one dict per key; normalize to version lists
        for k, v in list(state.get("edges", {}).items()):
            if isinstance(v, dict):
                state["edges"][k] = [v]
        self.mem_state = state
        self._text_set(self.mem_reply, state.get("assistant") or "(no turns yet - say something)")
        self._refresh_graph()
        self.mem_queries.delete(0, "end")
        for q in [q["q"] for q in state.get("queries", [])][-30:]:
            self.mem_queries.insert("end", q)

    def _mem_send(self):
        text = self.mem_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Playground", "Type a message first.")
            return
        self._mem_start(text)

    def _mem_demo(self):
        turns = memory_engine.DEMO_TURNS
        if self.demo_i >= len(turns):
            self.status_var.set("demo finished - Reset memory to run again")
            return
        text = turns[self.demo_i]
        self.demo_i += 1
        nxt = self.demo_i + 1
        self.mem_demo_btn.configure(
            text=f"Demo turn {nxt}/{len(turns)}" if nxt <= len(turns) else "Demo done")
        self._mem_start(text)

    def _mem_reset(self):
        if self.running:
            return
        if os.path.exists(MEM_STATE):
            os.remove(MEM_STATE)
        self.mem_state = {"history": [], "edges": {}, "queries": []}
        self.demo_i = 0
        self.mem_demo_btn.configure(text=f"Demo turn 1/{len(memory_engine.DEMO_TURNS)}")
        self._text_set(self.mem_reply, "(memory cleared)")
        self._text_set(self.mem_facts, "")
        self.mem_queries.delete(0, "end")
        self.mem_tree.delete(*self.mem_tree.get_children())
        self._draw_graph()
        self.mem_meta.configure(text="")
        self.mem_logw.configure(state="normal")
        self.mem_logw.delete("1.0", "end")
        self.mem_logw.configure(state="disabled")
        self.status_var.set("memory cleared")
        self.overall_var.set("memory cleared")

    def _mem_start(self, text):
        if self.running:
            return
        self._all_run_state("disabled")
        self.running = True
        self.status_var.set("memory turn running...")
        self.overall_var.set("running k=3 (3 batched streams)...")

        def work():
            err = None
            r = None
            try:
                state = {"history": [], "edges": {}, "queries": []}
                if os.path.exists(MEM_STATE):
                    state = json.load(open(MEM_STATE, encoding="utf-8"))
                r = memory_engine.run_turn(state, text)
                json.dump(state, open(MEM_STATE, "w", encoding="utf-8"), indent=2)
                self.mem_state = state
            except Exception as exc:
                err = str(exc)
            self.root.after(0, lambda: self._mem_done(r, err, text))

        threading.Thread(target=work, daemon=True).start()

    def _mem_done(self, r, err, text):
        self.running = False
        self._all_run_state("normal")
        if err:
            self.status_var.set("memory turn failed")
            self.overall_var.set("error - see event log")
            self._mem_log(f"turn: {text[:60]}{'...' if len(text) > 60 else ''}", "head")
            self._mem_log(f"ERROR: {err}", "err")
            return
        self._text_set(self.mem_reply, r["reply"])
        self.mem_input.delete("1.0", "end")
        slots = "/".join(f"{ms:.0f}" for ms in r["slot_ms"]) if r["slot_ms"] else "?"
        self.mem_meta.configure(
            text=f"k=3 batched · wall {r['wall_ms']:.0f} ms · slot completions {slots} ms")
        self._text_set(self.mem_facts, "\n".join(r["facts"]) if r["facts"] else "(none)")
        self.mem_queries.delete(0, "end")
        for q in [q["q"] for q in self.mem_state.get("queries", [])][-30:]:
            self.mem_queries.insert("end", q)
        self._refresh_graph()
        self._mem_log(f"turn: {text[:60]}{'...' if len(text) > 60 else ''}", "head")
        for kind, obj in r["events"]:
            self._mem_log(f"  graph {kind}: {obj}", "add" if kind == "ADD" else "close")
        for c in r["conflicts"]:
            self._mem_log(f"  CONFLICT: {c}", "conflict")
        if r["repaired"]:
            self._mem_log("  robust_parse repaired a missing brace", "query")
        if r.get("s1_recovered"):
            self._mem_log(f"  Slot 1 recovered by first-JSON fallback: {r['s1_recovered']!r}",
                          "conflict")
        if r.get("s2_recovered"):
            self._mem_log(f"  Slot 2 recovered by first-JSON fallback: {r['s2_recovered']!r}",
                          "conflict")
        for q in r["queries"]:
            self._mem_log(f"  indexed: {q}", "query")
        if r["s1_error"]:
            self._mem_log(f"  Slot 1 unparseable: {r['s1_error']} (state not updated)", "err")
        if r["s2_error"]:
            self._mem_log(f"  Slot 2 unparseable: {r['s2_error']}", "err")
        self._mem_log(f"  wall {r['wall_ms']:.0f} ms · 3 streams, 1 decode", "head")
        self.status_var.set(f"memory turn done in {r['wall_ms'] / 1000:.2f} s")
        self.overall_var.set(
            f"memory · wall {r['wall_ms'] / 1000:.2f} s · slots {slots} ms · "
            f"3 streams in 1 batched decode")

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self._close_log()
        self.running = False
        self._all_run_state("normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("stopped")
        self.overall_var.set("stopped")

    def _close_log(self):
        try:
            if getattr(self, "logf", None):
                self.logf.close()
                self.logf = None
        except Exception:
            pass

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self._close_log()
        if self.ag_proc and self.ag_proc.poll() is None:
            self.ag_proc.kill()
        self._agents_close_log()
        self.root.destroy()


    # ---------- agents tab: 3 slots decoded over one shared KV prefix ----------
    def _build_agents(self, tab):
        controls = ttk.Frame(tab, padding=(0, 12, 0, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Roles", style="Muted.TLabel").pack(side="left")
        self.ag_roles_var = tk.StringVar(value="balanced")
        ttk.Combobox(controls, textvariable=self.ag_roles_var, state="readonly", width=10,
                     values=("balanced", "classic")).pack(side="left", padx=(6, 16))
        ttk.Label(controls, text="Mode", style="Muted.TLabel").pack(side="left")
        self.ag_mode_var = tk.StringVar(value="both")
        ttk.Combobox(controls, textvariable=self.ag_mode_var, state="readonly", width=11,
                     values=("both", "concurrent")).pack(side="left", padx=(6, 16))
        ttk.Label(controls, text="Model", style="Muted.TLabel").pack(side="left")
        self.ag_model_var = tk.StringVar(value=AG_MODEL)
        ttk.Entry(controls, textvariable=self.ag_model_var, width=52).pack(side="left", padx=6)
        ttk.Button(controls, text="Browse", command=self._ag_browse).pack(side="left")
        self.ag_run_btn = ttk.Button(controls, text="Run", style="Accent.TButton",
                                     command=self._agents_run)
        self.ag_run_btn.pack(side="left", padx=(16, 6))
        self.ag_stop_btn = ttk.Button(controls, text="Stop", command=self._agents_stop,
                                      state="disabled")
        self.ag_stop_btn.pack(side="left")

        ttk.Label(tab, text=AG_TASK, style="Muted.TLabel", justify="left",
                  wraplength=1170).pack(fill="x", padx=2, pady=(0, 8))

        self.ag_cards = []
        subs = ("plan + synthesis", "shell + finding", "alert json")
        for name, sub in zip(AG_SLOT_NAMES, subs):
            card = ttk.Frame(tab, style="Card.TFrame", padding=12)
            card.pack(fill="x", pady=(0, 8), padx=2)
            head = ttk.Frame(card, style="Card.TFrame")
            head.pack(fill="x", pady=(0, 6))
            ttk.Label(head, text=name, style="Stream.TLabel").pack(side="left")
            ttk.Label(head, text=sub, style="Meta.TLabel").pack(side="left", padx=(10, 0))
            tok = ttk.Label(head, text="", style="Tok.TLabel")
            tok.pack(side="right")
            out = tk.Text(card, height=5, wrap="word", bg=BG, fg=FG, insertbackground=FG,
                          relief="flat", font=MONO, padx=8, pady=6, highlightthickness=0)
            out.configure(state="disabled")
            out.pack(fill="x")
            meta = ttk.Label(card, text="", style="Meta.TLabel")
            meta.pack(fill="x", pady=(4, 0))
            self.ag_cards.append({"out": out, "tok": tok, "meta": meta})

        mcard = ttk.Frame(tab, style="Card.TFrame", padding=12)
        mcard.pack(fill="both", expand=True, padx=2)
        ttk.Label(mcard, text="metrics", style="Stream.TLabel").pack(anchor="w", pady=(0, 6))
        self.ag_metrics = tk.Text(mcard, height=8, wrap="none", bg=BG, fg=FG,
                                  insertbackground=FG, relief="flat", font=MONO,
                                  padx=8, pady=6, highlightthickness=0)
        self.ag_metrics.configure(state="disabled")
        self.ag_metrics.pack(fill="both", expand=True)

    def _ag_browse(self):
        p = filedialog.askopenfilename(
            title="Choose GGUF model", initialdir=os.path.join(ROOT, "models"),
            filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")])
        if p:
            self.ag_model_var.set(p)

    def _ag_metrics_set(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        self.ag_metrics.configure(state="normal")
        self.ag_metrics.delete("1.0", "end")
        self.ag_metrics.insert("1.0", "\n".join(lines))
        self.ag_metrics.configure(state="disabled")

    def _ag_render(self, slots):
        for i, s in enumerate(slots[:len(self.ag_cards)]):
            c = self.ag_cards[i]
            c["out"].configure(state="normal")
            c["out"].delete("1.0", "end")
            c["out"].insert("1.0", s.get("text", ""))
            c["out"].configure(state="disabled")
            c["tok"].configure(text="%d tok" % int(s.get("ntok", 0)))
            c["meta"].configure(text=s.get("finish", ""))

    def _agents_run(self):
        if self.ag_running:
            return
        if not os.path.exists(DEVOPS):
            messagebox.showerror("Playground", f"Backend not found:\n{DEVOPS}")
            return
        model = self.ag_model_var.get().strip()
        if not os.path.exists(model):
            messagebox.showerror("Playground", f"Model not found:\n{model}")
            return
        for p in (AG_OUT, AG_LIVE):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        for c in self.ag_cards:
            c["out"].configure(state="normal")
            c["out"].delete("1.0", "end")
            c["out"].configure(state="disabled")
            c["tok"].configure(text="")
            c["meta"].configure(text="")
        self._ag_metrics_set("running...")
        env = dict(os.environ)
        env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
        args = [DEVOPS, "--roles", self.ag_roles_var.get(),
                "--mode", self.ag_mode_var.get(),
                "--model", model, "--out", AG_OUT, "--live", AG_LIVE]
        self.ag_logf = open(AG_LOG, "w", encoding="utf-8")
        self.ag_proc = subprocess.Popen(args, cwd=ROOT, env=env,
                                        stdout=subprocess.DEVNULL, stderr=self.ag_logf,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.ag_started = time.perf_counter()
        self.ag_running = True
        self.ag_run_btn.configure(state="disabled")
        self.ag_stop_btn.configure(state="normal")
        self.status_var.set("agents: running...")
        self.overall_var.set("running agents...")
        self.root.after(120, self._agents_poll)

    def _agents_poll(self):
        if not self.ag_running:
            return
        if self.ag_proc.poll() is None:
            el = time.perf_counter() - self.ag_started
            try:
                d = json.load(open(AG_LIVE, encoding="utf-8"))
                self._ag_render(d.get("slots", []))
                self.status_var.set("agents: %s  %.1fs" % (d.get("phase", "run"), el))
            except Exception:
                self.status_var.set("agents: running %.1fs" % el)
            self.root.after(150, self._agents_poll)
            return
        rc = self.ag_proc.returncode
        self._agents_close_log()
        self.ag_running = False
        self.ag_run_btn.configure(state="normal")
        self.ag_stop_btn.configure(state="disabled")
        if rc != 0 or not os.path.exists(AG_OUT):
            self.status_var.set("agents failed (exit %d) - see %s" % (rc, os.path.basename(AG_LOG)))
            self.overall_var.set("error - check the agents log")
            return
        try:
            data = json.load(open(AG_OUT, encoding="utf-8"))
        except Exception as exc:
            self.status_var.set("bad agents JSON: %s" % exc)
            return
        self._agents_show(data, time.perf_counter() - self.ag_started)

    def _agents_show(self, data, wall):
        conc = data.get("concurrent")
        ser = data.get("serial")
        src = conc or ser
        if not src:
            self._ag_metrics_set("no result")
            return
        self._ag_render(src.get("slots", []))
        ntok = sum(int(s.get("ntok", 0)) for s in src.get("slots", []))
        ticks = max(1, int(src.get("ticks", 1)))
        ms_tok = src.get("decode_ms", 0.0) / ntok if ntok else 0.0
        lines = []
        if conc and ser:
            lines.append("prefill   %5d tok / %5.0f ms   vs serial %5d tok / %5.0f ms   %.2fx"
                         % (conc.get("prefill_tokens", 0), conc.get("prefill_ms", 0.0),
                            ser.get("prefill_tokens", 0), ser.get("prefill_ms", 0.0),
                            ser.get("prefill_ms", 1.0) / max(1e-9, conc.get("prefill_ms", 1.0))))
            lines.append("wall      %6.0f ms vs serial %6.0f ms   %.2fx"
                         % (conc.get("wall_ms", 0.0), ser.get("wall_ms", 0.0),
                            ser.get("wall_ms", 1.0) / max(1e-9, conc.get("wall_ms", 1.0))))
            lines.append("parity    %d/3 slots byte-identical across modes"
                         % data.get("identical_slots", -1))
        else:
            lines.append("prefill   %5d tok / %5.0f ms"
                         % (src.get("prefill_tokens", 0), src.get("prefill_ms", 0.0)))
            lines.append("wall      %6.0f ms" % src.get("wall_ms", 0.0))
        lines.append("decode    %6.0f ms / %d tok = %.2f ms/token   (tick %.1f ms)"
                     % (src.get("decode_ms", 0.0), ntok, ms_tok,
                        src.get("decode_ms", 0.0) / ticks))
        lines.append("occupancy mean active %.2f / 3   ticks %d"
                     % (src.get("avg_active_slots", 0.0), ticks))
        lines.append("handoff   marker->injected %.1f ms   tool %.1f ms"
                     % (src.get("marker_to_inject_ms", 0.0), src.get("tool_ms", 0.0)))
        tool = data.get("tool", {})
        lines.append("tool      %s  -> %d bytes"
                     % ((tool.get("cmd") or "(none)")[:78], len(tool.get("out", ""))))
        self._ag_metrics_set(lines)
        ratio = ""
        if conc and ser and conc.get("wall_ms", 0) > 0:
            ratio = "%.2fx" % (ser.get("wall_ms", 0.0) / conc["wall_ms"])
        self.overall_var.set("occupancy %.2f  |  %.2f ms/token  |  wall %s"
                             % (src.get("avg_active_slots", 0.0), ms_tok, ratio))
        self.status_var.set("agents done in %.1f s" % wall)

    def _agents_stop(self):
        if self.ag_proc and self.ag_proc.poll() is None:
            self.ag_proc.kill()
        self._agents_close_log()
        self.ag_running = False
        self.ag_run_btn.configure(state="normal")
        self.ag_stop_btn.configure(state="disabled")
        self.status_var.set("stopped")
        self.overall_var.set("stopped")

    def _agents_close_log(self):
        try:
            if self.ag_logf:
                self.ag_logf.close()
                self.ag_logf = None
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
