# What should the inference-time state represent?

A mathematical memo on why attention is not the efficient representation, and what
the cheap one has to be. Written 2026-11-16.

**Scope.** Parts I and II (§1–§17) contain no model measurements: cost laws, three
lower bounds, one diagnosis, a proposed state decomposition, and the
trainability/expressiveness half, which turns the decomposition's central claim from a
prediction into a theorem about objectives and **corrects one claim in §3** (§11.4).
§18–§19 are the measured part: §19 runs one 3B model at three decode horizons and
compares two closed-form codes at matched bits, with no training and no gradients.
Algebra is executed in `harness/theory_check.py` (checks 1–6 in §10, checks 7–11 in
§16); every other number is labelled for where it comes from.

**Revised 2026-11-17**, after §19's three-length measurement: the abstract's exclusion
sentence, §5.3's fallback trigger, §8 and §9's status lines, §15's third prediction
and §17's F5 row were narrowed to what the three lengths support, §19.6 now states
what the result does and does not establish, and §19.8 names the next falsifiable
claim. Each changed passage carries *[rev. 2026-11-17]*. A second revision the same day closed
§19.6's floor hedge with the oracle bound (`harness/f5_oracle.py`): §19.6, §17's row,
the paragraph above and §19.8 were rewritten to the attainment result and carry
*[rev. 2026-11-17b]*.

**The claim, now a theorem.** Exact attention has no cheap implementation — the KV
cache is already the *minimal* state for the function attention computes (§2.1), so
"make attention cheaper without changing what it computes" is a dead end. The
efficiency error is upstream: attention computes a **universal** sufficient statistic
(it can answer any question about the transcript) where decoding needs only a
**predictive** one. Those are different objects: they tie exactly on next-token loss
(§11.3), and their sizes differ by $\Theta(t)$ vs $O(\log|Z|)$. Under a capacity
$R$ measured in *bits*, the achievable next-token loss is exactly flat for $R$ above
the belief's dimension and strictly worse below it (§12.2), so the transcript state is
excluded from the feasible set — not out-competed, excluded (§12.3) — and any state
that achieves the bound must be a minimal sufficient statistic. That is why the
representation is a *factorisation of the state by statistical role*: a fixed-size
carrier for language, a sparse index for memory, a bounded latent workspace for
reasoning, with the cost law of each matched to tokens, items or difficulty. Exclusion is a statement about *code length*: it is proved,
and what it does and does not force for a model at a measured context is §19.6.

**What the measurement changed about this.** §12.3's exclusion is a theorem about a
state whose code length grows as $\Theta(t)$ — the KV transcript. §19 measured a
different comparison: a rank-$r$ code of the *fixed-size* hidden state (3 072 floats,
49 152 bits) against a target-fitted code of the same state, at three contexts
(mean 156, 188, 236 tokens). There the copy is **not excluded** — it is out-competed
by 0.6–2.5 % of the achievable loss reduction at $R^\*$ (+0.026 / +0.111 / +0.096 nats),
the advantage does not grow with context over that range, and the state itself
compresses 6× for the next-token target (8 192 of 49 152 bits, §19.5). So the claim
at the strength the artifacts support is: **attention's state is not the minimal
state for prediction — it is about six times larger than the next-token target needs
— while the argument that a transcript-shaped code must grow with $t$ remains a
theorem about code length that no measurement at large $t$ has yet touched.**
**What settling the floor did to the claim.** The estimate §19.6 used to carry is now
definite. At $R^\* = 8\,192$ bits the best code of the family — the oracle subspace that
is allowed to see the targets themselves — scores 1.2304 / 1.0799 / 1.1026 nats,
**3.82–4.05× $H(Y \mid X)$**, so no member of the family attains the entropy target at
that rate: the binding constraint is rate, and the target's own log-prob field needs
about 1 000 of its 1 024 dimensions. At 16 368 bits the oracle *does* attain $H$
(0.3224 / 0.2838 / 0.2827) while the three codes sit 1.2680 / 1.1524 / 1.1235 nats above
it, so the floor's 1.1235–1.2680-nat distance above $H$ is entirely estimator error,
and that residual is not sample-count-limited (an exponent of 0.11 over a 3.06× range of $n$). The claim that
survives is therefore narrower and sharper than a pure capacity claim: **the state
compresses to 8 192 bits for prediction, the rate needed to reach the target's entropy is
about twice that, and what stops the registered linear readout from reaching it at any
rate is the readout.** §19.8 tests the readout. *[rev. 2026-11-17b]*

---

## 1. The cost functional

A causal model is a pair $(F, g)$ with state $s_t \in \mathcal{S}$:

```
s_t = F(s_{t-1}, x_t)              p(x_{t+1} | x_{<=t}) = g(s_t)
```

Two numbers describe it:

- **state size** $|s|$ — bytes touched per token (decode is bandwidth-bound, so this
  is the real currency; the 4060's E0 curve is a measurement of exactly this term),
- **operator cost** $c(F)$ — FLOPs per token.

Total decode cost over $L$ tokens is $\sum_t \big[|s_t| + c(F_t)\big]$. Everything
below is a statement about this sum.

Attention: $s_t = (K_t, V_t) \in \mathbb{R}^{t \times 2d}$, so $|s_t| = 2dt$ and the
sum is $\Theta(L^2 d)$. Linear attention: $s_t = \sum_j \phi(k_j) v_j^\top$, $|s| =
d^2$, sum is $\Theta(L d^2)$. SSM: $|s| = d$, $\Theta(L d)$.

So the standard story — "attention is quadratic, use a smaller state" — is right
about the algebra and wrong about the question. The algebra says *smaller state is
cheaper*. It does not say *smaller state can do the same job*, and the next two
sections show it cannot.

---

## 2. Three lower bounds

### 2.1 Exactness: attention's state cannot be compressed

Fix $t$ items with distinct keys $k_1..k_t$ and a query $q$. Attention computes

```
Attn(q) = sum_j w_j(q) v_j ,     w(q) = softmax(q·k_1, ..., q·k_t)
```

The object that must be stored is the map $q \mapsto w(q) \in \Delta^{t-1}$. This is
a $t$-dimensional function family: for $t$ distinct queries the matrix
$W_{ij} = w_i(j)$ is generically full rank (check 6: $t=40$, $d=16$ → numerical
rank **40**, condition number $4.7\times10^4$). A state of dimension $d$ supplies at
most $d$ degrees of freedom. Hence:

> **Proposition 1.** No fixed-size state implements exact softmax attention for
> unbounded $L$. Exact attention requires $|s| = \Omega(t)$.

Attention's KV cache is therefore not a wasteful implementation — it is the minimal
state for its own function. This is the important reframe: **you cannot optimise
attention; you can only replace it.** Every "efficient attention" that keeps the
function is either approximate (sparse, low-rank) or is a different function wearing
the same name.

Proposition 1 also explains the empirical shape of the field: log-linear attention
grows the state logarithmically (Guo et al., ICLR 2026), Sparse Delta Memory and
Sparse State Expansion grow it by orders of magnitude with sparse addressing
(arXiv:2607.07386; ICLR 2026), and hybrids keep a few full layers. All of them are
buying back capacity along the only axis Proposition 1 leaves open: making the state
grow, just more slowly.

### 2.2 Capacity: a fixed state holds Θ(d) associations

The alternative is to stop being exact and hold a *compressed* memory. That has a
hard wall. For a Hebbian/associative memory of dimension $d$ storing $N$ random
patterns, retrieval is clean up to load $\alpha = N/d \approx 0.138$ and collapses
above it (check 5, $d=64$):

| $\alpha = N/d$ | 0.05 | 0.10 | 0.138 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| mean overlap | +1.000 | +1.000 | +0.995 | +0.979 | +0.931 |
| fraction exact fixed points | 1.00 | 1.00 | 0.88 | 0.55 | 0.12 |

> **Proposition 2.** A state of dimension $d$ holds $\Theta(d)$ associations with
> bounded crosstalk; beyond that, retrieval error grows as superposition
> interference, not gracefully but as a phase transition.

This is why pure linear attention fails associative recall and why every fix in the
literature is a capacity fix (bigger state, sparse state, partitioned state) rather
than a better update rule. It is also the design equation for any replacement: the
carrier dimension is set by the number of things that must be recalled *simultaneously
and exactly*, and nothing else.

### 2.3 Composition: depth cannot be bought with layers

A 2026 communication-complexity result (arXiv:2602.1763) proves an unconditional
separation: no number of linear-attention layers substitutes for one full-attention
layer on deep sequential composition (multi-hop retrieval where each step's output
defines the next step's query). This is a real constraint on any "replace attention
with a cheap mixer" programme, and it is the strongest argument for the user's
suspicion that one mechanism cannot do everything.

But note *what* the bound is about: composition performed **inside one forward
pass**, where the only way to get another step is another layer. It says nothing
about composition performed by **iteration at test time** on a bounded latent.
That is a different compute axis, and it is the one the reasoning module below uses.

---

## 3. The diagnosis: data-sufficient vs belief-sufficient

Let $z_t$ be the latent situation — everything about the world and the discourse
that the future depends on. Then

```
p(x_{t+1} | x_{<=t}) = ∫ p(x_{t+1} | z) d b_t(z),      b_t = P(z_t | x_{<=t})
```

$b_t$ is *a* sufficient statistic of the prefix for the next token, and — when the
situation is identifiable from the predictive distribution — the minimal one. $K_t,V_t$
is *also* sufficient: the prefix itself, embedded. Here is the point this section got
wrong in its first draft, and §11 states it properly: **both carry exactly the same
information about the next token**, $I(\cdot;Y)=I(X;Y)$, because that is what
sufficiency means (Lemma 11.3). There is no next-token mutual-information excess for
the KV to be guilty of. The excess is about everything *else*: the KV is sufficient for
**any** functional of the transcript — it can answer a question about any earlier token,
which is what a prompt can ask — while the belief is sufficient only for the next
token. The excess is $\Delta = H(S\mid Y) - H(b\mid Y)$: information that survives
conditioning on the target.

> **Diagnosis.** Attention computes a *universal* sufficient statistic where decoding
> needs a *predictive* one. Its state size is $O(\text{tokens})$; the quantity the
> computation needs is $O(\text{bits of belief}) = O(\text{entities} \times
> \text{attributes})$. Text length grows without bound; situation complexity does not.
> Universal sufficiency costs $\Theta(t)$; predictive sufficiency costs
> $O(\log|Z|)$ (§12.2), and the ratio is unbounded in $L$.

This is the mathematical content of "attention is a lookup table that avoids doing
inference". It also predicts the observed failure profile rather than merely
describing it:

- long context degrades gracefully in *fluency* but sharply in *retrieval* — the
  interference term of Proposition 2, not a modelling flaw;
- "lost in the middle" — the softmax weighting map of 2.1 is rank-deficient in its
  tail (check 6: 73% of the energy in 8 of 40 directions), so low-energy directions
  are exactly the ones whose queries get outvoted;
- linear/SSM models are strong on language and weak on copying — they replaced the
  belief state with a *linear filter*, which is a different approximation to the
  same posterior, and filters cannot represent "retrieve this one item and exclude
  the rest".

Every efficient architecture in the field is an approximation to $b_t$ with a
different bias. None of them is *doing the inference*.

---

## 4. The representation

Three states, because there are three jobs with three different scaling quantities.
"One model" is not the right unit; one *token generator* over three state objects is.

### 4.1 Carrier — what language needs. Cost $\propto$ tokens, state $\propto$ 0.

```
M_t = A_t M_{t-1} + beta_t (v_t - M_{t-1} k_t) k_t^T ,    A_t = I - 2 u_t u_t^T
```

$M \in \mathbb{R}^{d \times m}$ per head. $A_t$ a reflection (or a contraction
$I - \beta k k^\top$ with $\beta < 2$), the bracket is the delta rule.

- **Cost:** $O(dm)$ per token, independent of $L$.
- **Job:** local syntax, morphology, style, the "finite monoid" part of language —
  regular-language quotient structure, position-sensitive recurrence, agreement,
  and the smooth part of the next-token distribution.
- **Why it is enough for that job:** the syntactic monoid of natural-language local
  structure is finite and small; a rank-1-write matrix state of dimension $dm$
  realises monoids far larger than needed. What the carrier cannot do is verbatim
  recall of an arbitrary item — and it should not try (Proposition 2).

### 4.2 Index — what memory needs. Cost $\propto$ items, growth $\propto$ distinct facts.

```
I_t = { (kappa_i, rho_i) }        kappa = quantised semantic key, rho = compressed payload + provenance
```

- **Write policy (the only novel part):** write when the carrier's residual
  $\lVert v_t - M_{t-1} k_t \rVert$ exceeds a threshold $\tau$. The delta rule's
  write error is a *free, calibrated surprise signal* — it is literally "the
  compiled state could not predict this token". §13 separates what this buys from
  what it does not: the *write* is the derived min-norm projection (no free
  parameters, and exactly zero when the token is already predicted), while the
  *decision to write* ($\tau$) and the metric are not derived.
- **Read:** top-1/top-$k$ over $\kappa$; hashed buckets give $O(1)$ average,
  tree/ANN gives $O(\log N)$.
- **Cost:** $O(\log N)$ per token, $O(1)$ amortised with hashing.
- **Why it must be separate:** Proposition 2. Exact recall of $N$ facts needs
  $\Omega(N)$ capacity; no fixed carrier can hold it, and forcing it to try is what
  makes both the model and the memory bad. The index also *is* the thing the
  hardware can afford: 512-byte items, not 32 KB/token of KV.

Crucially the index grows with **distinct retrievable facts**, not with tokens.
A 10k-token document that states 40 facts costs 40 items, not 10k.

### 4.3 Workspace — what reasoning needs. Cost $\propto$ difficulty.

```
z in R^{k x p}      E(z ; M_t, I_t) = sum_i U_i(z_i) + sum_{(i,j) in G} P_{type(i,j)}(z_i, z_j)
z* = argmin E     for S sweeps, S chosen by an energy threshold
```

- $k$ = number of currently bound slots (adaptive: grows when a new entity is
  introduced, shrinks when one is resolved), $G$ a *typed sparse* factor graph,
  $P_r$ a low-rank potential per relation type.
- **Cost:** $O(S \cdot k \cdot \gamma)$ per query, $\gamma$ = mean factor degree,
  **independent of context length and of hop count**. Composition is bought with
  iterations at fixed parameters, which is the axis the §2.3 lower bound does not
  close.
- **Why energy descent and not more CoT:** CoT pays $O(\text{tokens} \times
  \text{model})$ for sequential composition. Descent pays $O(S \cdot k)$ local
  updates on a bounded object. CRAM (arXiv:2611.17309) is the existence proof in the
  constraint domain: 282× fewer inference FLOPs than looped models at matched
  accuracy, with built-in stopping. Its weakness is not the descent — it is the
  *energy family*, which is the next point.

### 4.4 The four objects a token touches, and the full cost law

```
read:   carrier (O(dm))  +  index probe (O(log N))  +  workspace (only when reasoning)
write:  carrier residual -> maybe index append
emit:   readout(carrier, retrieved payload, z*)
```

$$\text{cost/token} = \Theta(dm + \log N) \quad\text{vs attention's}\quad \Theta(Ld)$$

with a per-query term $\Theta(S k \gamma)$ that is spent only when the task needs
composition.

---

## 5. Trainability (this is where the proposal can die)

Expressiveness is easy — the class contains attention as the special case where the
index holds every token and the readout is softmax over it. Trainability is the hard
part, and it is where the naive version of this idea fails.

### 5.1 The carrier has a benign Jacobian

With reflections, the scan's Jacobian is a product of orthogonal matrices:

> **T1.** $\lVert \partial s_t / \partial s_0 \rVert_2 = 1$ exactly, for any length.
> Check 1: max deviation from 1 over 200 reflections is $1.3\times10^{-15}$.

No vanishing, no explosion, no length-dependent gradient scaling. With
$\beta \in (0,2)$ the transition is non-expansive, so the state is bounded by the
input norm (input-to-state stability). This is exactly what the additive form
lacks: check 2 shows the additive state growing as $\sqrt{t}$ (10.0 → 19.7 → 40.1 at
$t$ = 100/400/1600, matching $\sqrt{t}$), while the delta state stays near
$\sqrt{d}$. The consequence is not cosmetic: a state whose norm drifts with context
length makes every downstream readout's scale context-dependent, which is why plain
linear attention needs normalisation and gating to train at all.

### 5.2 The write rule is derived, not learned

> **T2.** The delta write is the exact minimiser of $\lVert Mk - v \rVert^2$:
> $\beta = 1$ gives $P = I - kk^\top$ idempotent and post-write error
> $1.1\times10^{-15}$ (check 4).

So the carrier has only two learned maps (token $\to (k,v)$, state $\to$ readout) and
one derived update. Fewer free parameters in the recurrence means a better
conditioned optimisation problem, and the projection property means a write never
increases the error on the key it just wrote.

### 5.3 The bottleneck is not identified by next-token loss

This is the failure mode that kills "just add a small state and train it".

> **T3.** Let $\mathcal{B}$ be the hypothesis class of states. The NLL
> $\mathbb{E}[-\log p(x_{t+1}\mid b)]$ is invariant to adding redundant coordinates
> to $b$: every $b$ in the sufficient set is a minimiser. If $\mathcal{B}$ contains
> a copy map from the history, then a copy is a minimiser, and it is the *easiest*
> one to reach, because it is linear in the input and needs no inference.

So minimality cannot come from the loss. It must come from one of:

1. **Hard capacity** — make the copy unrepresentable (fixed-size carrier, discrete
   slots, quantised keys). Cost: you must get the capacity right a priori.
2. **Auxiliary objective that is not redundancy-invariant** — predict the *effect of
   an intervention* (supersede a fact and ask what changes) rather than the next
   token. Redundant surface information cannot help predict a counterfactual, so
   the gradient has nowhere to hide.
3. **Supervision on the write decision** — label what deserves an item, from
   provenance/line-diffs rather than from a prompt.

§12 decides between these: **(1) is the only admissible one, and only in its
bit-counting form** — a constraint on *bits*, not on dimension, norm or slot count.
The reason is not that penalties are weak (Prop. 12.5(a) shows a true rate penalty
excludes the copy within ~3 tokens) but that the penalties actually used do not measure
rate at all (Prop. 12.5(b)). Options (2) and (3) remain the fallbacks if no linear code at $R^*$ is good.
§19 measured that the target-fitted linear code reaches 98.8 % of the achievable
reduction at $R^*$ (§19.5), so that trigger has not fired in the range measured, and
they stay untested repairs for a learnability problem rather than for the objective.
*[rev. 2026-11-17]*

This is not a hypothetical. The repo's angles 27–37 are a controlled demonstration:
a 3B model asked to maintain a compacted profile kept every surface detail and
failed at exactly the two operations that require a belief state — invalidation
(angle 30: recall 0/5) and attrition under multi-turn rewriting (32c: 10/10 turns
with loss; 36's delta schema made it *worse*, 2/10 correct). The model was fluent
throughout. That is the signature of T3: the objective never asked for the statistic,
so the copy survived and the inference did not happen.

### 5.4 The energy must be trained where the inference happens

> **T4.** Pseudolikelihood is informative only where $\partial E / \partial \theta
> \neq 0$. On the data manifold the conditional excludes the variable being scored,
> so for a count-type energy the gradient is identically zero.

The repo measured this: cycle 35 trained on *solved* boards and got a flat
likelihood; training on **corrupted** boards was what moved the parameters. The
general rule: the training distribution must cover the inference trajectory, not the
answer. This is also why CRAM trains 1000× cheaper — no differentiating through the
solver, one local objective, and the stop certificate is a scalar threshold calibrated
on the energy, not a learned head.

### 5.5 Three timescales, trained separately

| module | objective | why separate |
|---|---|---|
| carrier | next-token NLL through the scan | parallelisable (chunked delta scan); orthogonal transitions make depth free |
| index | write-policy classification + contrastive read | discrete; gradient through retrieval is neither necessary nor stable |
| workspace | pseudolikelihood on corrupted states | no BPTT; gradient must see the solver's distribution (T4) |

Joint end-to-end training would put a discrete write policy and a non-differentiable
argmin on the same graph as the language loss, and T3 says the language loss will
win by degrading the other two into copies.

---

## 6. Cost accounting at matched expressivity (arithmetic, not measurement)

Llama-3.2-1B config: 16 layers, $d_{model}=2048$, 32 heads, 8 KV heads, head dim 64.
Per token: $2 \cdot 16 \cdot 8 \cdot 64 = 16384$ values = **32 KB** (fp16).

| $L$ | KV state | carrier state ($d{=}64$, 8 heads, 16 layers) | KV bytes/token read |
|---|---:|---:|---:|
| 1k | 32 MB | 1 MB | 32 KB |
| 8k | 256 MB | 1 MB | 32 KB |
| 32k | 1.0 GB | 1 MB | 32 KB |
| 128k | 4.1 GB | 1 MB | 32 KB |

The carrier is **1024× smaller than the 32k KV and constant**. At the 4060's
~272 GB/s, reading 1 GB of KV per decoded token is a ~3.7 ms/token floor before any
math; at 32k the model is spending most of its decode budget re-reading a transcript
it has already processed. This is the same term the repo's E0 curve measures as the
tail penalty — the theory says that term is not intrinsic to intelligence, it is
intrinsic to storing the transcript.

Compute per token: attention $\approx 4Ld_{model}$ (QK + AV) $= 268$ MFLOP at
$L{=}32k$; carrier $\approx 2 \cdot 64 \cdot 64 \cdot 8 \cdot 16 = 1.0$ MFLOP. Two
orders of magnitude at 32k, roughly 8× even at $L{=}1k$.

Workspace per query: $k{=}64$ slots × 9 candidates × 40 sweeps ≈ 23k cell
evaluations, i.e. the same order as CRAM's Sudoku budget, independent of context.

---

## 7. Why this is not just another hybrid

| | what grows | what it still stores | what it cannot do |
|---|---|---|---|
| attention | $L$ · KV | the transcript | be cheap |
| linear attention / SSM | nothing | a linear filter of the past | exact recall (§2.2) |
| log-linear attention | $\log L$ states | a hierarchy of filters | sharp single-item retrieval at scale |
| SDM / SSE | big sparse state | addressed dense memory | decouple capacity from FLOPs fully; still no explicit composition |
| hybrids | $n_{full}$ layers | partial transcript | escape §2.3 per-layer bound |
| **this proposal** | items (facts) | belief + index + workspace | nothing claimed; the risk is optimisation (T3) |

The difference is not "use a cheaper mixer". It is that the *growing* object is
chosen to be the one that actually needs to grow — the set of distinct facts — while
the per-token state is fixed and the composition cost tracks difficulty. Attention's
error is not its kernel; it is that it made the transcript, the belief, and the
working memory the same object.

---

## 8. What would falsify this

Ordered cheapest-first. F5 has now been run (§19); F1–F4 have not.

**F1 — is the state redundant? (zero training, existing dumps).**
If the KV is a non-minimal statistic, its predictive content must saturate well
below $t$. Cheap version with what the repo already has: per-position next-token
loss / pre-sampler entropy on a long passage; check whether the top-surprise
positions are the ones a later needle probe depends on. If surprise is uncorrelated
with later retrieval need, the write-on-surprise policy is dead and §4.2 needs a
different trigger. **Kill condition:** AUC of surprise → "token later needed" below
~0.6.

**F2 — how many things are bound at once? (probing, small).**
Probe the effective number of distinct entities/attributes active in a forward pass
(entity count per sentence, pronoun resolution set size). The whole decomposition
rests on this number being small (tens, not hundreds). **Kill condition:** if the
effective bound-variable count in ordinary text is $\gtrsim 500$, the workspace and
carrier together need KV-sized state and nothing is gained.

**F3 — is attention's low-rank tail the recall circuitry? (capture, small).**
Check 6 shows the query→weights map is full rank with 73% of its energy in 8 of 40
directions. Predict: in trained models, ablating the *residual* directions costs
retrieval tasks specifically and leaves perplexity nearly unchanged. **Kill
condition:** if the residual directions carry no task-specific function, the
"capacity wall is real but irrelevant" reading wins and a compressed carrier is
enough for everything — which would actually be the *better* outcome for efficiency,
and would remove the index.

**F4 — does descent beat CoT on composition? (small model, stage 2).**
Multi-hop over a small KB with a known factor graph. Predict: at ≥3 hops, energy
descent on $k$ slots beats CoT at matched FLOPs, because descent cost is independent
of hop count. **Kill condition:** if required sweeps grow linearly with hops, the
win collapses to a constant factor.

**F5 — the trainability claim itself (§15).** The one experiment that separates
"the bit constraint makes the belief the optimum" from "the copy wins even under the
constraint", using two closed-form codes on frozen features. No gradient training. Its
third kill condition refutes the central claim outright. **Run 2026-11-17 at three
lengths (§19): no kill condition fires.** The predictive code enters the 2 %-of-gap
band at 8 192 bits at all three lengths, the copy never beats it there, and the copy's
own requirement (12 288 bits) does not grow across the measured range. What the run
cannot say — long context, a decoder below the floor, the entropy target — is §19.6;
what it leaves open is §19.8. *[rev. 2026-11-17]*

**The internal warning that already exists.** The repo's CRAM replication
(`cram_mech` → `cram_bound` → `cram_latent`) found a repair boundary at ~30 blanked
cells where the paper reports 99.996%, and learned energies lost three times. The
most likely cause is the energy *family*, not the dynamics: the harness used a
type-shared count penalty $p[\text{type},d]$, while CRAM's own energy is a per-pair
$K_{ij}$ (263k parameters for Sudoku). If that diagnosis is right, it is evidence
*for* §4.3 — the factor graph's expressivity is the binding constraint, and it must
be typed and sparse rather than shared. It is also a warning: an under-expressive
energy does not degrade gracefully; it hits a boundary and stops.

---

## 9. Limits of this memo

- **One LLM was run, once, at short context.** Parts I–II are predictions. §19 is the
  only measured result: one 3B model, three decode horizons (mean context 156, 188,
  236 tokens), one 32-problem corpus, closed-form linear codes, no training and no
  gradients. Everything §19 asserts is scoped to that. The algebra checks in §10 are
  algebra, not evidence about language.
- **Expressivity is settled for the class (§14.1, exact containment) but still not
  for SGD.** §14.2 quantifies what sublinear capacity costs: verbatim recall error
  falls only as $R/L$, while smooth error falls exponentially in bits per dimension.
  What remains open is not whether a good state exists at rate $R^*$ but whether a
  *learnable* one does — T3's ε > 0. F5 (§15) has now run at three lengths (§19): a
  target-fitted linear code at $R^*$ closes 98.1–98.8 % of the achievable reduction
  while the variance-preserving code closes 96.3–97.5 %, so the premise is not refuted
  over that range. It is not established either: nothing was trained, and the comparison
  is between two closed-form codes, not between two learned models. The ε question
  is therefore untouched in the direction that matters — whether SGD finds a code
  that exists in closed form.
- **§12's theorem is about $R$ in bits.** Every statement assumes the capacity is a
  code length. For a model whose state is a float vector of unconstrained precision,
  the theorem does not apply to its dimension — which is exactly the gap Prop. 12.5(b)
  identifies, and not a limitation of the result.
- **A1 (situation screening) is the weakest link in Part II.** If the future depends on
  the transcript in ways no latent situation screens off — indexical facts, exact
  surface form, a user's phrasing that matters for the next token — then the belief is
  not sufficient, $R^*$ is not small, and Theorem 12.2 is vacuous for that component.
  The theorem is conditional on A1; nothing in it can repair A1.
- **§2.3's hierarchy result is accepted, not evaded.** Composition is moved to test
  time, which is a different compute axis, not a proof that the bound does not
  apply. If test-time iteration turns out to need $\Omega(\text{hops})$ sweeps *and*
  each sweep needs full attention over a growing workspace, the advantage shrinks to
  the constant CRAM measures.
- **Hardware-specific.** The bandwidth argument in §6 is for one 4060 at 272 GB/s
  with WDDM. On a part with much higher compute-to-bandwidth ratio the crossover
  moves.
- **Prior art is dense here.** Sparse state expansion, sparse delta memory,
  log-linear attention, structured-sparse entity tracking (EMNLP 2026) and CRAM each
  cover part of this. What is not covered, as far as I can tell, is the specific
  claim that the state should be **factored by statistical role with a cost law per
  role**, and that the carrier's own residual is the write policy for the index.

---

## 10. The algebra checks (executed)

Part I below; Part II's checks (rate–distortion flatness, penalty vs constraint,
non-identifiability, delta crosstalk, Gaussian rate–distortion) are in §16.

`python harness/theory_check.py` — numpy only, < 10 s, no GPU, no model:

```
1. reflection product orthogonal:      min sv 0.9999999999999987, max 1.0000000000000007
                                       max |sv-1| = 1.3e-15
2. state norm vs t:                    additive 10.01 / 19.67 / 40.07   (sqrt(t) = 10/20/40)
                                       delta     7.08 /  7.98 /  7.94   (bounded)
3. diagonal transitions:               commutator 0.0, filter rebuild error 4.4e-16
4. delta rule beta=1:                  idempotent 3.3e-16, Pk 6.9e-17, post-write err 1.1e-15
5. Hebbian capacity (d=64):            alpha 0.05/0.10/0.138/0.20/0.30 ->
                                       overlap 1.000/1.000/0.995/0.979/0.931
                                       fixed points 1.00/1.00/0.88/0.55/0.12
6. softmax weight rank:                t=40, d=16 -> rank 40, cond 4.7e4,
                                       73% of energy in 8/40 directions
```

Check 5 reproduces the classical $\alpha_c \approx 0.138$ collapse, which is the
capacity wall in §2.2. Check 6 is the one that matters for the argument: the
query→weights map is **full rank but ill-conditioned**, so the *typical* case is
compressible and the *tail* is not. That tail is where exact retrieval lives, and it
is the reason a fixed carrier cannot replace attention outright — only the part of
attention that language actually needs.

---

# Part II — identification, capacity, and what the write rule buys

Part I argued that the state should be factored and that minimality cannot come from
the loss. That was a prediction. This part proves the objective-level statement,
quantifies the expressiveness loss, and states the experiment that would falsify the
learnability premise.

---

## 11. Identification: sufficiency, minimality, and where the excess actually is

**Setting.** $X = x_{\le t}$ (the transcript), $Y = x_{t+1}$ (the target), $Z = z_t$
(the latent situation). Write $b_t(\cdot) = P(Z \in \cdot \mid X)$ for the belief.

**Assumption A1 (situation screening).** $Y \perp X \mid Z$.

A1 is an assumption, not a theorem, and everything below inherits it. It says the
future depends on the transcript only through the situation — that $Z$ really is the
situation. It is the minimal assumption under which "the belief is enough" is even
meaningful, and it is the one thing in this memo that a reader should attack first.

> **Lemma 11.1 (the belief is sufficient).** Under A1, $p(Y \mid b_t) = p(Y \mid X)$.

*Proof.* $b_t = f(X)$, so
$p(Y\mid X) = E[p(Y\mid Z,X)\mid X] = E[p(Y\mid Z)\mid X] = \int p(Y\mid z)\,db_t(z)$
by A1, and the last expression is a function of $b_t$ alone. $\square$

> **Lemma 11.2 (minimality).** Define predictive equivalence $x \sim x'$ iff
> $p(Y\mid\cdot\mid x) = p(Y\mid\cdot\mid x')$, and let $[x]$ be the quotient. Then
> $[x]$ is sufficient and is a function of every sufficient statistic; hence it is the
> minimal sufficient statistic (the standard Blackwell–Rao / Lehmann–Scheffé result,
> in its measurable form). $b_t$ is minimal whenever the map $b \mapsto p(Y\mid b)$ is
> injective — i.e. whenever the situation is identifiable from the predictive
> distribution. Otherwise $b_t$ is sufficient but not minimal, and the minimal
> statistic is coarser.

*Proof sketch.* Sufficiency of $[x]$ is by definition. If $S$ is sufficient then
$p(Y\mid X) = p(Y\mid S)$, so $S$'s level sets refine the predictive classes and
$[x] = \varphi(S)$. Minimality is then immediate; injectivity makes $b_t$ a bijection
onto the quotient. $\square$

### 11.3 The correction: all sufficient statistics tie on the target

> **Lemma 11.3.** If $S = S(X)$ is sufficient then $I(S;Y) = I(X;Y)$.

*Proof.* $S = f(X)$ gives $I(S;Y) \le I(X;Y)$ by data processing. Sufficiency is
$Y \perp X \mid S$, so $I(X;Y) = I(S;Y) + I(X;Y\mid S) = I(S;Y)$. $\square$

Every sufficient statistic — the KV cache, the belief, the minimal quotient — carries
exactly $I(X;Y)$ nats about the next token. There is no such thing as "the KV carries
more information about $x_{t+1}$".

> **Corollary 11.4 (what the excess is).** $H(S) - H(b) = H(S\mid Y) - H(b\mid Y) \ge 0$,
> and for the KV this is $\Theta(t)$ against $O(\log|Z|)$.

*Proof.* By Lemma 11.3 the $I(\cdot;Y)$ terms are equal, so
$H(S) - H(b) = [H(S) - I(S;Y)] - [H(b) - I(b;Y)] = H(S\mid Y) - H(b\mid Y)$. $\square$

The excess is precisely the information that **survives conditioning on the target** —
the part of the transcript that the next token does not determine. That is the honest
form of "the KV is redundant": it is not redundant about $Y$, it is redundant about
everything $Y$ leaves unresolved. Check 9 is this lemma in numbers: $H(b) = 3.0$ bits,
$H(\text{transcript at } t{=}1000) = 7000$ bits, and the loss gap is exactly $0.0$ nats.

### 11.4 What this overturns in §3

§3's first draft said the KV "carries far more information about $x_{t+1}$". That is
false by Lemma 11.3, and it is corrected in place: about $x_{t+1}$ the two are exactly
tied; about the rest of the transcript the KV has more, and that is what costs
$\Theta(t)$. The diagnosis survives in a sharper form — attention buys *universal*
sufficiency (any question about the past) when decoding needs *predictive* sufficiency
(the next token) — and it is that distinction, not a quantity of next-token
information, that §12 turns into a theorem.

---

## 12. The constrained-objective theorem

A state is any $S = f(X)$; a decoder is any $q(y\mid S)$. The expected log-loss is
$L(S) = E[-\ln q(Y\mid S)] = H(Y\mid S) + \mathrm{KL} \ge H(Y\mid S)$, with equality
at $q = p(\cdot\mid S)$. So the best achievable loss at capacity $R$ is

$$
L^*(R) \;=\; \min_{f:\; H(f(X)) \le R} H\big(Y \mid f(X)\big),
$$

and **capacity means bits**, not dimension, not norm, not slot count. That choice is
the content of Proposition 12.5.

> **Lemma 12.1.** $L^*$ is non-increasing, and $L^*(R) \ge H(Y\mid X)$ with equality iff
> some sufficient statistic is feasible at rate $\le R$.

> **Theorem 12.2 (flatness at the belief's dimension).** Let
> $R^* = \min\{H(S) : S \text{ sufficient}\}$. Then
>
> $$L^*(R) = H(Y\mid X) \ \text{ for all } R \ge R^*, \qquad L^*(R) > H(Y\mid X) \ \text{ for all } R < R^*,$$
>
> and $I(X;Y) \le R^* \le H(b_t) \le \log|Z| = O(k\,a\,\log v)$ in the notation of §4.

*Proof.* $R^* \le H(b_t)$ because $b_t$ is sufficient (Lemma 11.1); $H(b_t) \le \log|Z|$
because $b_t$ is a distribution on $Z$; $R^* \ge I(X;Y)$ because every sufficient $S$
satisfies $H(S) \ge I(S;Y) = I(X;Y)$ by Lemma 11.3. Let $S^*$ attain $R^*$; it is
sufficient, so $L^*(R) \le H(Y\mid S^*) = H(Y\mid X)$ for $R \ge R^*$, and
$\ge$ holds always. For $R < R^*$: any feasible $S$ has $H(S) \le R < R^*$ and is
therefore not sufficient; and
$H(Y\mid S) - H(Y\mid X) = I(Y;X\mid S)$ (chain rule, using $S = f(X)$), which is
$0$ only if $S$ is sufficient. So the inequality is strict. $\square$

**Check 7 evaluates $L^*$ exactly** — by enumerating every state function (every
partition of an 8-state situation space), not by bounding it. $R^* = 2.99$ bits:

| $R$ (bits) | 0 | 1 | 2 | **3** | 4 | 5 |
|---|---|---|---|---|---|---|
| best loss (nats) | 2.0713 | 2.0290 | 1.9863 | **1.9469** | 1.9469 | 1.9469 |
| excess over Bayes | +0.1244 | +0.0821 | +0.0395 | **0.0000** | 0.0000 | 0.0000 |

The excess falls about 0.041 nats per bit and then is **exactly zero, forever**. The
transcript rate for that same loss is 6988 bits at $t{=}1000$. So capacity above
$R^*$ buys nothing for the next token, and the transcript is 2300× past the point
where it stopped buying anything (check 9).

> **Corollary 12.3 (the transcript is infeasible, not merely worse).** Since
> $H(K_t,V_t) = \Theta(t\,b_v)$, for any capacity $R \in [R^*, \Theta(t\,b_v))$ the
> transcript state is outside the feasible set. No objective value, gradient, or
> initialisation can select it: **exclusion is by feasibility**. The constrained
> minimiser is attained by a minimal sufficient statistic, and every state that attains
> it is sufficient (at $R = R^*$, minimal).

> **Corollary 12.4 (the gap is unbounded).** $\partial L^*/\partial R = 0$ on
> $[R^*,\infty)$ while the transcript's rate grows as $\Theta(t\,b_v)$, so capacity
> wasted above $R^*$ is $\Theta(t\,b_v) - R^* = \Theta(t)$, and the ratio
> $|KV|/R^*$ diverges in $L$.

### 12.5 Which constraint is admissible

Let $J_\lambda(S) = L(S) + \lambda\rho(S)$ be a penalised objective, $\varepsilon$ the
excess loss of the best statistic the model class can actually represent at rate
$\le R^*$, and $b_v$ the bits per token of the transcript.

> **(a) If $\rho$ is a code length** ($\rho = H$ up to constants), the copy is
> penalised-optimal only while
> $\lambda b_v t < \varepsilon + \lambda R^*$, i.e. $t < t^* = (\varepsilon + \lambda R^*)/(\lambda b_v)$.
> Check 8: $\varepsilon = 0.05$ nats, $\lambda = 10^{-3}$, $b_v = 16$ gives
> $t^* = 3.31$ tokens. A true rate penalty excludes the copy almost immediately —
> **penalties are not inherently weaker than constraints.**
>
> **(b) If $\rho$ is not a code length** — dimension, Frobenius norm, number of slots —
> then $\rho$ is invariant to precision: a $d$-dimensional slot at $p$ bits carries
> $dp$ bits, so the copy's $\rho$ is *constant in $t$* while its information grows as
> $\Theta(t)$. No $\lambda$ excludes it at any $t$. Check 8: one `float64` slot is 53
> bits; a 1-dimensional state is not a 1-bit constraint.
>
> **(c) Under the hard constraint** the copy is infeasible for every $t > R/b_v$,
> independent of $\lambda$ and — this is the part that matters — independent of
> $\varepsilon$.

So the criterion is not soft versus hard. It is **bits versus shape**. A hard bound is
the simplest bit constraint and the only one whose verdict does not depend on $\varepsilon$,
the approximation error of whatever statistic the model class happens to contain. That is
the precise sense in which §5.3's "minimality has to come from hard capacity" is right,
and the precise sense in which it was underspecified: capacity means a **code length**,
and a state with unconstrained precision has none.

**This re-reads the repo's angles 29–32c.** Their constraint was ≤75 tokens of state
text (median 32–48). At ~1–2 bits of entropy per token of generated text that is
roughly 10² bits — the right *order* for a belief about one vehicle, one year, one
city, one pet and one residence. So capacity was probably *not* the binding problem;
$\varepsilon > 0$ was. The model could not represent the belief at that rate (0/5
invalidation in angle 30, 10/10 attrition in 32c, and 2/10 correct under the delta
schema in angle 36), so the copy-like summary won. That is case (a) with a large
$\varepsilon$ and no rate term at all — a learnability failure, not a capacity
shortfall. It is also why the fix has to change what the model must infer (line-diff
supervision, a quantised slot schema), not restate the instruction.

---

## 13. What the delta write buys, exactly

> **Definition 13.1 (the write).** With $M_{t-1}$ the carrier, $k_t$ a unit key and
> $v_t$ a value,
>
> $$M_t = \arg\min_{M} \lVert M - M_{t-1}\rVert_F \ \ \text{s.t. } M k_t = v_t
> \;=\; M_{t-1} + (v_t - M_{t-1}k_t)\,k_t^\top .$$

*Proof.* The constraint set is affine; its min-norm element is the orthogonal
projection of $M_{t-1}$ onto it, which is the rank-1 update. $\square$ Verified: check 4
($P^2 = P$ to $3.3\times10^{-16}$, $Pk = 0$ to $6.9\times10^{-17}$, post-write error
$1.1\times10^{-15}$).

**Derived — no free parameters.** (i) The update is rank-1 along $k_t$; (ii) its step
size is the residual, so it is the identity when the state already predicts the token
($v_t = M_{t-1}k_t \Rightarrow M_t = M_{t-1}$, exactly); (iii) the interpolation
constraint; (iv) the transition's non-expansiveness (§5.1). "Write only what the
current state fails to predict" is therefore a *theorem about this rule*, not a
heuristic — which is the whole reason the residual can be trusted as a surprise
signal for the index.

**Learned.** The maps $x_t \mapsto (k_t, v_t)$, the readout, and the reflection/decay
schedule $A_t$.

**Not derived — the honest gap.**

1. *Whether to write at all.* The rule always writes; only $\tau$ decides. $\tau$ is a
   free parameter that no part of the objective fixes.
2. *The metric.* Min-norm is Euclidean, but the loss is a function of the readout
   distribution, so the loss-optimal correction is Fisher-preconditioned:
   $M_t = \arg\min \lVert M - M_{t-1}\rVert_G^2$ s.t. $Mk_t = v_t$, with $G$ the
   readout's Fisher information. For $G \neq I$ the greedy step is not the
   loss-optimal step.
3. *The schedule.* $A_t$ is chosen for conditioning (§5.1), not derived.

**The suboptimality, quantified (check 10).** With $d = 64$ random unit keys, the item
written last is exact to $\sim10^{-32}$, while the item written first degrades:

| $t$ | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| load $\alpha = t/d$ | 0.39 | 0.78 | 1.56 | 3.12 | 6.25 |
| first item MSE | 0.277 | 0.848 | 1.873 | 2.141 | 2.004 |
| mean MSE over stored items | 0.202 | 0.465 | 0.894 | 1.369 | 1.711 |

The write is optimal item-wise and suboptimal sequence-wise, and the suboptimality is
superposition interference — **the same term that sets the $\Theta(d)$ capacity of
Proposition 2**. So the gap cannot be closed by a better update rule; it is the
capacity wall seen from the write side. That is the formal reason the carrier cannot
replace the index, and the reason a carrier of order $10^3$ bits plus a
$\Theta(\log N)$ index is the right pair rather than a bigger carrier.

---

## 14. Expressiveness preserved, and exactly where it is lost

> **Theorem 14.1 (exact containment).** For any attention layer with context $L$ and
> head dimension $d$, there is a factored configuration with
> $R_{\text{index}} = 2Ldb$ bits, a trivial carrier and an empty workspace whose
> next-token map equals the layer's exactly.

*Construction.* $\kappa_j = j$, $\rho_j = (k_j, v_j)$, readout
$\mathrm{softmax}(q^\top \kappa_j/\sqrt{d})\cdot \rho_j$. $\square$

So $\mathcal{F}_{\text{factored}} \supseteq \mathcal{F}_{\text{attention}}$ at
capacity $\Theta(L)$, and the class inherits transformer universality for continuous
sequence-to-sequence maps on compact domains (Yun et al. 2020), positional encodings
included. **No expressiveness is lost in the class.** The configuration that proves it
is exactly as expensive as attention, and that is the point: efficiency is a *capacity
choice inside one class*, not a different class. A proposal that changed the class
would have to defend an approximation; this one only has to defend an optimisation.

> **Proposition 14.2 (what sublinear capacity costs, quantitatively).** Let
> $R < 2Ldb$.
>
> **(a) Verbatim regime.** Holding $k$ key-value pairs exactly needs $k\,db$ bits of
> content (each pair is $db$ bits and the pairs must be distinguishable), so
> $k \le \lfloor R/(db)\rfloor$, and a uniformly random query is answered wrong with
> probability $\ge 1 - R/(Ldb) \to 1$ as $L$ grows at fixed $R$. Check 9: at $d{=}64$,
> $b{=}16$ an item costs 1024 bits, so a 4096-bit state holds 4 items exactly out of
> however many the transcript contains.
>
> **(b) Smooth regime.** The output error is Lipschitz in the KV distortion, and for a
> $d$-dimensional Gaussian source the rate–distortion function is
> $D(R) = \sum_i \min(\sigma_i^2, \theta)$ with $\theta$ solving
> $\sum_i \max(0, \tfrac12\log_2(\sigma_i^2/\theta)) = R/d$. Check 11 ($d{=}64$, total
> variance 39.74): $D = 26.07$ at $R{=}16$, $18.10$ at 32, $9.00$ at 64, $2.25$ at 128,
> $0.14$ at 256, with the allocation satisfying its budget exactly and exchange-optimal
> (moving 0.1 bit between any pair never lowers $D$).

**The two regimes have different exponents.** Smooth error decays *exponentially* in
bits per dimension; verbatim error decays only *linearly* in $R/L$. That is the
quantitative form of §2.2: the capacity wall bites only on the non-smooth, exact-copy
part of the map, and it bites hard — the error rate goes to 1, not to a small
constant. Everything else is cheap.

> **Corollary 14.3 (the index relocates the cost — this is the whole trick).** Exact
> storage of $L$ pairs costs $\Theta(Ldb)$ bits of **storage** but only $O(\log L)$ bits
> of **bandwidth** per query; attention pays $\Theta(L)$ **bandwidth**. Expressiveness
> is preserved not by a bigger carrier but by moving capacity from bandwidth to
> storage.

This is the mathematical reason the index is the right object, and the reason the
4060's E0 tail penalty is a bandwidth artifact. A transcript that must be re-read per
token is a bandwidth problem; the same transcript stored in a bucket-addressed table is
a storage problem, and storage is the resource this hardware has spare.

---

## 15. F5: the falsifier for the trainability claim

§12 says: at capacity $R^*$ the belief is the optimum and the copy is infeasible.
T3 (§5.3) says: a model class with approximation error $\varepsilon > 0$ may still not
reach it. These two claims predict different things about a *matched-capacity*
comparison, and that is what F5 measures. Zero training, closed form, frozen features:

1. Teacher-force a long passage through a frozen model; at each position capture the
   state (KV or last hidden, $h_t \in \mathbb{R}^{m}$) and the next-token distribution
   $p_t$.
2. Build two codes at the same bit budget $R$, both by least squares:
   - **copy code** — rank-$r$ SVD of the stacked states, i.e. the projection that
     minimises $\lVert H - \hat H\rVert_F$ (preserve the transcript);
   - **predictive code** — reduced-rank regression from $h_t$ to $\log p_t$, i.e. the
     projection that preserves only what predicts the next token.
3. Plot next-token loss against $R$ for $L \in \{1k, 8k, 32k\}$.

**Prediction.** The predictive code reaches the full-context loss at $R \approx R^*$
and is flat beyond it; the copy code needs $R \propto L$; the gap widens with $L$.

*Measured (§19), clause by clause — the registration is left as written.* The first
clause holds in the sense K2 tests: a plateau at 8 192 bits at all three lengths. The
third does not: the gap is +0.026 / +0.111 / +0.096 nats at L = 64 / 128 / 256, i.e.
it appears between 64 and 128 and then flattens. The second is **not testable as
written here**: F5's copy code is a rank-$r$ code of a fixed 3 072-float hidden state,
so its stored requirement is bounded by 16 368 bits — the rank ceiling — and cannot
grow as $\Theta(t)$ however long the context. $R \propto L$ is Cor. 12.3's statement
about the *KV transcript*, a different object; §19.8 restates it on objects whose
sizes are allowed to differ. *[rev. 2026-11-17]*

**Kill conditions.**

- **K1 (the index becomes unnecessary).** If the copy code matches the predictive code
  at $R \approx R^*$ for all three lengths, capacity is not binding in real models and
  the whole capacity argument — hence §4.2 — is wrong.
- **K2 ($R^*$ is not small).** If the predictive code's loss keeps falling beyond
  $R^*$ with no plateau, then $R^* \approx \Theta(\log|Z|)$ is false for real text, so
  the premise that a low-dimensional belief exists is false.
- **K3 (the central claim dies).** If *both* codes need $R \ge 0.5\,(2Ldb)$ — no
  compression at all — then the state is irreducible for the next-token target and
  attention is already minimal for it. That would refute the thesis, not just the
  design.

**Why it discriminates.** Both codes are feasible at the same $R$, so the comparison
*is* the constrained-objective question: which statistic wins at a bit budget. If the
predictive code wins at $R^*$, capacity is doing the work (Theorem 12.2 is operative).
If the copy wins, then the class cannot reach the belief at $R^*$ — T3's $\varepsilon$
— and the fix must be the auxiliary objective of §5.3(2), not a tighter constraint.
**Caveat:** both codes are linear, so a predictive-code failure does not refute the
nonlinear version; it refutes the version that needs only linear compression, which is
the version the carrier and index assume.

---

## 16. Part II checks (executed)

`python harness/theory_check.py`, sections 7–11 (numpy only, no GPU, no model):

```
7.  exact L*(R), M=8 situations, F=16 filler symbols, all state functions enumerated
    Bayes 1.9469 nats; belief dimension 2.99 bits
    R=0  2.0713 (+0.1244)   R=1  2.0290 (+0.0821)   R=2  1.9863 (+0.0395)
    R=3  1.9469 (+0.0000)   R=4  1.9469 (+0.0000)   R=5  1.9469 (+0.0000)
    transcript rate at t=1000: 6988 bits for the same loss
8.  penalty vs constraint, J = loss + 1e-3 * rate
    crossover t* = 3.31 tokens; copy wins at t=1, loses at t>=10
    dimension is not capacity: 1 slot = 53 bits (f64), 64 slots = 1024 bits (f16)
9.  non-identifiability at matched loss
    H(belief) = 3.0 bits; H(transcript, t=1000) = 7000 bits; loss gap = 0.0 nats
    copy feasible at R = H(belief): False; belief feasible: True
    exact recall: 1024 bits per item (d=64, b=16)
10. delta-rule crosstalk, d=64 random unit keys
    t=25 a=0.39 MSE 0.202 | t=100 a=1.56 MSE 0.894 | t=400 a=6.25 MSE 1.711
    newest item exact (~1e-32) at every t
11. Gaussian rate-distortion, d=64, total variance 39.74
    R=0  sum_bits 0.00  D=39.7400     R=16  16.00  D=26.0687
    R=32  32.00  D=18.1016           R=64  64.00  D=8.9985
    R=128 128.00 D=2.2496            R=256 256.00 D=0.1406
    exchange gain from moving 0.1 bit between any pair: 0.00e+00 (optimal)
```

Two of these were repaired during writing and are worth recording: check 11's first
implementation used the high-rate closed form $\theta = 2^{-2R/d}(\prod\sigma_i^2)^{1/d}$,
which violates its own budget below the water-filling floor (at $R{=}0$ it reported
`sum_bits = 8.82`) — it now solves for $\theta$ by bisection, and `sum_bits = R` exactly
at every rate. And check 7 was written to enumerate *every* state function rather than
bound $L^*$, because a bound would not have shown the flatness.

---

## 17. Status of every claim

| claim | status |
|---|---|
| Prop. 1: exact attention needs $\Omega(t)$ state | proved (§2.1) + checked (check 6) |
| Prop. 2: $\Theta(d)$ associations per fixed state | proved (classical) + checked (check 5) |
| §2.3 composition bound | quoted from the literature, accepted not evaded |
| Lemma 11.1–11.3, Cor. 11.4 | proved (A1 is used by 11.1) |
| Thm 12.2 flatness, Cor. 12.3–12.4 | proved + evaluated exactly (check 7) |
| Prop. 12.5 admissibility | proved + arithmetic (check 8) |
| Def. 13.1, the derived write | proved + checked (check 4) |
| §13 suboptimality | measured in the toy (check 10), not proved for trained models |
| Thm 14.1 containment | proved by construction |
| Prop. 14.2(a) verbatim loss | proved by counting |
| Prop. 14.2(b) smooth loss | standard rate–distortion + checked (check 11) |
| A1 (situation screening) | **assumption** — the weakest link |
| §13 learned/not-derived items | stated, not measured |
| F5 prediction | **measured at three lengths** (§19): K1 **fail** (the copy code does not match at $R^*$ at two of three lengths and is never better: gap +0.026 / +0.111 / +0.096 nats at L = 64 / 128 / 256, with $R^*$ = 8 192 bits at all three); K2 **fail** (plateau at every length); K3 **fail** (8 192 bits = 16.7 % of the single-position state, ≤ 0.054 % of the KV at the measured context). Recall (§19.9): at the same 8 192 bits the predictive code loses 5–8 points of lag-1 recall to the copy code and both sit below the recall-fitted oracle, so neither single-objective code serves both axes; a code fitted to both recovers
   the recall for 1–3 % CE (§19.9b). **Attainment** (§19.6): at 16 368 bits the oracle subspace attains $H$, so the whole floor above it is estimator error; at 8 192 bits no code of the family can ($3.82–4.05\times H$, the oracle bound). K3 is a compression statement about the fixed-size hidden state, not a defeat of attention's KV; the residual floor (4.8–5.4× $H(Y\mid X)$) and the context range (156–236 tokens) bound what it can claim — §19.6, §19.8 |

Nothing in Part II required a model run. What it does not settle is whether a
*learnable* statistic reaches $R^*$; that is $\varepsilon$, it is what angles 27–37
measured to be large, and F5 (§19) now bounds one side of it: a closed-form
target-fitted code does reach the band at three short contexts, which says the
statistic exists and says nothing yet about whether SGD finds it.

---

## 18. F5 was attempted against the existing dumps: it is blocked on capture

*(Superseded by §19: the two fields this section specifies were implemented and
captured, and F5 was run. §18 is kept as the record of what the old dumps could and
could not support.)*

The attempt is recorded here because the negative is itself information: the repo has
per-step dumps for 55,567 emitted tokens, and **no dump pairs a usable state with a
usable target** — the probe dumps have the right layer and the right target but reduce
both to scalars, and the only vector dump is layer-0 with no target at all.

### 18.1 What the dumps actually contain

| dump | rows | fields | what it is |
|---|---|---|---|
| `data/_hid_pNN.jsonl` | one per (stream, step) | `k, m, s, step, norm, cmean` | last-layer embedding **norm** and cross-stream cosine |
| `data/_probs_pNN.jsonl` | 55,567 total | `H, p1, mg, lp, tok, pos, step, s, off, len` | four **scalars** of the next-token distribution + the emitted id |
| `data/_probs_pNN.json` | one per problem | stream texts, phase split | no vectors |
| `results/ebr/*.npz` | 350 / 800 rows | `e1, maxp, ent, C, dEdT, instab, kladj, swap` | probe features and one fitted head, not states |
| `results/circuit/act_ffn_norm0_full.bin` | **2047 × 2048, float32** | tensor `ffn_norm-0` | the only real vector dump in the repo |

That last file is the one honest candidate and it is worth being exact about why it
cannot carry F5 (`RESULTS_modify.md` §2): it is a single mixed-corpus context, one
2047-token stretch, **layer 0**. It fails on three counts, and none is fixable by
re-reading the file:

1. **No target was dumped for it** — no token ids, no distribution, at any position.
2. **It is layer 0.** `ffn_norm-0` is the embedding after one RMSNorm; the next token is
   barely linearly predictable from it. Any readout fitted on it would have a loss floor
   that measures layer 0's poverty, not the cost of compression, so a K1/K2 verdict from
   it would be about the wrong quantity.
3. **One length.** $L = 2047$ only, so there is no $L$-ladder and no out-of-length
   transfer to test.

The probe side is the mirror image — right layer, right target, scalars only: the state
is $\mathbb{R}^1$ (a norm), the target $\mathbb{R}^4$ (entropy, top-1 prob, margin,
emitted logprob). A rank-$r$ SVD of a 1-dimensional state is the identity, and a
reduced-rank regression to four scalars is not a regression to the next-token
distribution. And the two dumps cannot be combined: different model (1B vs 3B),
different corpus, different run. Running anything on these files would be a proxy
substitution, not F5.

### 18.2 Why: both quantities are computed, then thrown away

Serialisation is not the obstacle — the circuit dump proves a 2047 × 2048 state costs
16.8 MB on disk. The obstacle is that on the probe path two `fprintf` calls drop what is
already in memory:

- **State.** `parallel/src/runtime.cpp:89` keeps `std::vector<float> hvec; // last
  embedding vector`, and `record_hidden` fills it per step. `app/prefix_cli.cpp:559-561`
  then writes `{"k":"h",...,"norm":%.6f,"cmean":%.6f}` and drops the vector.
- **Target.** `app/parallel_cli.cpp:651` writes `"tok":%d,"H":%.5f,"p1":%.6f,"mg":%.6f`.
  The full-vocab pre-sampler statistics are already computed in one pass
  (`Branch::raw_stats()`), so the distribution is in memory at the write site and only
  four summaries are serialised.

So this is not a "capture would be expensive" problem. It is a serialisation choice at
two `fprintf` calls.

### 18.3 The capture F5 needs (and nothing more)

Two fields, at the same alignment the dumps already use (`s`, `step`):

1. **`hvec`, dtype float16, length `n_embd`** (3072 for Llama-3.2-3B).
   55,567 rows × 3072 × 2 B ≈ **341 MB**.
2. **top-$k$ next-token distribution**, `k = 64`: token ids (uint32) + logprobs (float16).
   55,567 × 64 × 6 B ≈ **21 MB**. Full logits would be 55,567 × 128,256 × 2 B ≈ 14 GB and
   are unnecessary — the rank ladder only needs the head of the distribution plus the
   emitted token's logprob.

Total ≈ 360 MB, no extra forward pass (both are already in memory), no model change, no
training. With those two fields, F5 is the §15 procedure unchanged: rank-$r$ SVD of $H$
versus reduced-rank regression of $h_t$ onto $\log p_t$, both at $R = r \cdot 2\,b_v$ bits,
scored by held-out cross-entropy against the emitted token. The kill conditions K1–K3
are unchanged, and **none of the three can be issued from the current dumps**: K1 needs
both codes, K2 needs the predictive code's curve, K3 needs a measured bit requirement.

### 18.4 The one F5-adjacent quantity the dumps do support

Not a test of F5, and labelled as such: the predictive information is bounded above by
the target's entropy, and the dumps contain the model's entropy at every step.

Over the 55,567 rows: mean $H = 0.264$ nats ($0.380$ bits/token), median $0.013$,
90th percentile $0.878$. Since $I(X;Y) \le H(Y)$,

$$I(X;Y) \;\le\; 0.38 \ \text{bits per emitted token (mean)}, \quad \le 1.27 \ \text{bits at } p_{90}.$$

One step reads 112 KB of KV (Llama-3.2-3B: $2 \cdot 28 \cdot 8 \cdot 128$ values, fp16)
= 917,504 bits, to resolve a target whose model entropy is 0.38 bits — an instantaneous
over-provisioning factor of ~$2.4 \times 10^6$.

**What this does not do.** It does not test whether a rank-$r$ code at $R^*$ preserves
that information, which is the whole question, and it does not bound $R^*$ from below. It
also inherits a bias: these are model-generated CoT traces at temperature 0.6 where the
model is near-deterministic on 92% of steps, so $H$ is a model-entropy estimate and an
underestimate of the conditional entropy of text generally. It is reported because it is
the only thing the existing files can honestly say about the theorem, not because it
substitutes for the falsifier.

---

## 19. F5, measured: copy vs predictive at matched bits, at three lengths

§18 recorded that F5 could not run because no dump paired a state with a target. That
blocker was two serialisation choices, and it is removed (§19.1). This section is the
measurement, now at the three lengths §15 asks for: loss-vs-$R$ curves for both codes at
identical stored bits, the resolution of the estimator that produces them, and K1–K3
issued on the result.

Every number below is recomputed in this pass from files generated in this pass:
`results/f5L{64,128,256}_c1024.json` (train p01–p20, test p21–p32) and
`results/f5L{64,128,256}_cv2.json` (two folds by problem, every position scored
out-of-fold once). `harness/f5_doc_tables.py` writes the tables from those artifacts,
`harness/f5_verify.py` re-derives the counts, the reference points and curve rungs from
the raw sidecars by separate code, and the table cells are checked against the artifacts
mechanically (§19.7). Nothing from the earlier version of this section is carried
forward unverified.

### 19.1 The capture, at three lengths

Unchanged capture path — the two fields §18.3 specified, at the two write sites that
already compute them, with no extra forward pass:

1. `hvec`, float16, `n_embd` = 3072, from the `Branch` that `raw_stats()` holds
   (`HRec.vec`/`StepRec` in `parallel/include/parallel/types.hpp`,
   `parallel/src/runtime.cpp`), written by `app/parallel_cli.cpp` as a flat `.f16`
   file plus a `row` index.
2. top-64 token ids and raw log-softmax values per emitted token (`tid`, `tlp`) in the
   existing `--probs` sidecar.

Both are written in the same `accept_token` tick, so a state and a distribution with the
same `step` come from the same forward pass and the emitted token is not part of the
state. The join is therefore on `(stream, step)`. Per length, the number of rows written
minus the number paired is exactly 256 = 8 streams × 32 problems, the prompt-side rows
that have no state yet:

| L (max new tokens) | rows written | paired on (s, step) | unpaired |
|---|---|---|---|
| 64 | 16 384 | 16 128 | 256 |
| 128 | 32 470 | 32 214 | 256 |
| 256 | 51 845 | 51 589 | 256 |

Same corpus (`data/_probs_pNN.tsv`, 32 problems × 8 streams), same model
(Llama-3.2-3B-Instruct-Q4_K_M), seed 1234, temperature 0.6, top-p 0.95, so the three
captures differ only in decode horizon. Context length is measured, not assumed: the
sidecar's `pos` field gives the absolute token position, and the mean context is 156,
188 and 236 tokens at L = 64, 128, 256 — the prompts are ~100 tokens, so the horizon
dominates the range.

**Determinism check.** The L = 256 capture was re-run from scratch in this pass and
reproduced the earlier capture row for row: the same 51 589 paired positions and the same
curve (floor 1.4061 nats, $R^\*$ = 8 192 bits, gap at $R^\*$ +0.0957). The pipeline is
reproducible, not merely re-used.

### 19.2 The codes, the bits, and the two estimators

**Bit convention, identical across codes and rungs.** Each code stores $r$ coefficients
per position at 16 bits (fp16), so $R = 16r$ bits per position. The encoder ($m \times r$)
and decoder ($r \times C$) are amortised over positions, have the same shape for every
code, and are excluded from the per-position budget for all codes alike. Both codes are
scored on the same held-out rows against the same target, so $R$ is matched exactly at
every rung.

- **copy code** — rank-$r$ truncated SVD of the standardised states: the $r$ directions
  the state *varies* in, no target information (§15's $\min\lVert H - \hat H\rVert_F$
  projection).
- **predictive code** — rank-$r$ truncation of the ridge-fitted readout $h \mapsto \log p$:
  the $r$ directions the readout *uses* (§15's reduced-rank regression, expressed as a
  truncated map). $\lambda$ is one amortised scalar, chosen on held-out *problems*
  (p01–p12 fit, p13–p20 validate) over a 5-point path.
- **rrr** — the literal whitened reduced-rank regression $\Sigma^{-1/2}X^\top Y$,
  reported as a third curve.

Target: the model's top-64 log-softmax restricted to the 1 024 most frequent ids and
renormalised; classes outside a row's top-64 are censored at their 64th value and imputed
at a global floor (deepest observed log-prob − 1), so their total mass is ≈ 0. Positions
whose top-64 mass is covered by less than half the candidate set are dropped (1 704 /
3 605 / 6 017 at L = 64 / 128 / 256): they are the flat, rare-token positions, and every
statement below is conditioned on the rest.

**Two estimators per length.** *full* trains on p01–p20 and tests on p21–p32 (the
original protocol). *cv2* splits by problem into two folds and scores every position
out-of-fold exactly once — the same corpus, 2.5× the paired test rows at L = 256, and a
probe of how much the answers depend on sample size. $\lambda$ is not re-tuned per fold
there (fixed at 1 % of the mean Gram eigenvalue), so the CV curves are not selected.

**Intervals.** Rows inside a problem share a prompt and a stream, so the problem is the
independent unit: the gap's interval is computed from the 12 (full) or 32 (cv2) per-problem
means with a $t$ correction, not from the row count. Row-level standard errors would be
2–4× narrower and are reported in the artifacts only for reference.

### 19.3 The curves

Reference points and the plateau onset per length:

| L | mean ctx tok | paired rows | kept | dropped | train | test | target H | no-state CE | no-state top-1 | floor CE | R\* (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 156 | 16,128 | 14,424 | 1,704 | 9,027 | 5,397 | 0.3224 | 5.9391 | 0.077 | 1.5588 | 8,192 |
| 128 | 188 | 32,214 | 28,609 | 3,605 | 17,895 | 10,714 | 0.2838 | 5.9451 | 0.090 | 1.4362 | 8,192 |
| 256 | 236 | 51,589 | 45,572 | 6,017 | 27,650 | 17,922 | 0.2719 | 5.8735 | 0.096 | 1.4061 | 8,192 |

| L | at R\*: copy CE | pred CE | gap (nats) | 95 % CI | copy % of gap | pred % of gap | copy top-1 | pred top-1 |
|---|---|---|---|---|---|---|---|---|
| 64 | 1.6687 | 1.6406 | +0.0258 | [-0.0198, +0.0714] | 97.5 | 98.1 | 0.761 | 0.758 |
| 128 | 1.6016 | 1.4884 | +0.1112 | [+0.0525, +0.1699] | 96.3 | 98.8 | 0.785 | 0.785 |
| 256 | 1.5534 | 1.4588 | +0.0957 | [+0.0642, +0.1272] | 96.7 | 98.8 | 0.791 | 0.798 |

**L = 64** (mean context 156 tokens, test 5,397 positions)

| r | R (bits) | copy CE | pred CE | rrr CE | copy % | pred % | gap nats | 95 % CI | copy top-1 | pred top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 7.6342 | 7.1705 | 6.9683 | -38.7 | -28.1 | +0.4630 | [+0.2603, +0.6657] | 0.114 | 0.094 |
| 2 | 32 | 6.7998 | 7.3166 | 6.6495 | -19.6 | -31.4 | -0.5215 | [-0.8732, -0.1698] | 0.149 | 0.136 |
| 4 | 64 | 6.5479 | 6.8984 | 7.4878 | -13.9 | -21.9 | -0.3480 | [-0.6377, -0.0584] | 0.208 | 0.153 |
| 8 | 128 | 5.9891 | 6.5951 | 6.7821 | -1.1 | -15.0 | -0.6008 | [-1.0127, -0.1890] | 0.253 | 0.203 |
| 16 | 256 | 5.1862 | 5.9655 | 5.5497 | 17.2 | -0.6 | -0.7843 | [-1.0181, -0.5506] | 0.375 | 0.243 |
| 32 | 512 | 4.2598 | 4.7970 | 4.1321 | 38.3 | 26.1 | -0.5427 | [-0.8375, -0.2479] | 0.464 | 0.352 |
| 64 | 1,024 | 3.2990 | 3.6014 | 3.1682 | 60.3 | 53.4 | -0.3129 | [-0.5688, -0.0571] | 0.574 | 0.489 |
| 128 | 2,048 | 2.5284 | 2.6379 | 2.3743 | 77.9 | 75.4 | -0.1157 | [-0.2566, +0.0253] | 0.676 | 0.609 |
| 192 | 3,072 | 2.1563 | 2.1919 | 2.0789 | 86.4 | 85.5 | -0.0395 | [-0.1489, +0.0699] | 0.716 | 0.690 |
| 256 | 4,096 | 1.9183 | 1.9635 | 1.9026 | 91.8 | 90.8 | -0.0479 | [-0.1496, +0.0537] | 0.737 | 0.724 |
| 384 | 6,144 | 1.7472 | 1.7312 | 1.7253 | 95.7 | 96.1 | +0.0144 | [-0.0348, +0.0635] | 0.753 | 0.745 |
| 512 | 8,192 | 1.6687 | 1.6406 | 1.6591 | 97.5 | 98.1 | +0.0258 | [-0.0198, +0.0714] | 0.761 | 0.758 |
| 768 | 12,288 | 1.5588 | 1.5896 | 1.5980 | 100.0 | 99.3 | -0.0316 | [-0.0565, -0.0067] | 0.770 | 0.758 |
| 1023 | 16,368 | 1.5904 | 1.5904 | 1.5904 | 99.3 | 99.3 | +0.0000 | [-0.0000, +0.0000] | 0.758 | 0.758 |

**L = 128** (mean context 188 tokens, test 10,714 positions)

| r | R (bits) | copy CE | pred CE | rrr CE | copy % | pred % | gap nats | 95 % CI | copy top-1 | pred top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 7.5948 | 7.8139 | 7.3867 | -36.6 | -41.4 | -0.2158 | [-0.4225, -0.0091] | 0.130 | 0.103 |
| 2 | 32 | 7.1513 | 7.2581 | 7.3102 | -26.7 | -29.1 | -0.1009 | [-0.3162, +0.1145] | 0.169 | 0.123 |
| 4 | 64 | 6.6690 | 7.3672 | 7.6413 | -16.1 | -31.5 | -0.6869 | [-1.0859, -0.2879] | 0.172 | 0.157 |
| 8 | 128 | 6.0524 | 6.9865 | 6.9638 | -2.4 | -23.1 | -0.9237 | [-1.3047, -0.5427] | 0.232 | 0.196 |
| 16 | 256 | 5.2940 | 6.1299 | 5.5859 | 14.4 | -4.1 | -0.8362 | [-1.0840, -0.5885] | 0.334 | 0.259 |
| 32 | 512 | 4.4809 | 4.7875 | 4.4914 | 32.5 | 25.7 | -0.3122 | [-0.5510, -0.0734] | 0.444 | 0.363 |
| 64 | 1,024 | 3.5361 | 3.5988 | 3.3842 | 53.4 | 52.0 | -0.0756 | [-0.3056, +0.1544] | 0.556 | 0.479 |
| 128 | 2,048 | 2.6308 | 2.5078 | 2.3088 | 73.5 | 76.2 | +0.1144 | [-0.0550, +0.2839] | 0.673 | 0.633 |
| 192 | 3,072 | 2.2346 | 2.0399 | 1.9339 | 82.3 | 86.6 | +0.1902 | [+0.0778, +0.3026] | 0.717 | 0.715 |
| 256 | 4,096 | 1.9307 | 1.8046 | 1.7591 | 89.0 | 91.8 | +0.1231 | [+0.0423, +0.2039] | 0.744 | 0.749 |
| 384 | 6,144 | 1.7022 | 1.5882 | 1.5851 | 94.1 | 96.6 | +0.1118 | [+0.0577, +0.1658] | 0.776 | 0.772 |
| 512 | 8,192 | 1.6016 | 1.4884 | 1.5009 | 96.3 | 98.8 | +0.1112 | [+0.0525, +0.1699] | 0.785 | 0.785 |
| 768 | 12,288 | 1.4703 | 1.4371 | 1.4436 | 99.2 | 100.0 | +0.0319 | [-0.0034, +0.0673] | 0.790 | 0.787 |
| 1023 | 16,368 | 1.4362 | 1.4362 | 1.4362 | 100.0 | 100.0 | +0.0000 | [-0.0000, +0.0000] | 0.787 | 0.787 |

**L = 256** (mean context 236 tokens, test 17,922 positions)

| r | R (bits) | copy CE | pred CE | rrr CE | copy % | pred % | gap nats | 95 % CI | copy top-1 | pred top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 7.7584 | 8.1500 | 7.6529 | -42.2 | -51.0 | -0.3850 | [-0.5651, -0.2048] | 0.133 | 0.088 |
| 2 | 32 | 7.3309 | 7.4772 | 7.5637 | -32.6 | -35.9 | -0.1394 | [-0.3268, +0.0481] | 0.169 | 0.121 |
| 4 | 64 | 6.6708 | 7.6666 | 7.5341 | -17.8 | -40.1 | -0.9766 | [-1.4171, -0.5362] | 0.170 | 0.161 |
| 8 | 128 | 6.1247 | 7.0361 | 7.0211 | -5.6 | -26.0 | -0.8834 | [-1.2391, -0.5278] | 0.232 | 0.207 |
| 16 | 256 | 5.2299 | 5.8908 | 5.6827 | 14.4 | -0.4 | -0.6360 | [-0.9272, -0.3449] | 0.342 | 0.279 |
| 32 | 512 | 4.4749 | 4.7949 | 4.5772 | 31.3 | 24.1 | -0.2879 | [-0.5824, +0.0067] | 0.445 | 0.375 |
| 64 | 1,024 | 3.4831 | 3.5974 | 3.2916 | 53.5 | 51.0 | -0.0918 | [-0.3431, +0.1594] | 0.577 | 0.493 |
| 128 | 2,048 | 2.5931 | 2.4579 | 2.2623 | 73.4 | 76.5 | +0.1560 | [-0.0247, +0.3366] | 0.688 | 0.658 |
| 192 | 3,072 | 2.1929 | 1.9941 | 1.8955 | 82.4 | 86.8 | +0.2068 | [+0.1122, +0.3015] | 0.725 | 0.735 |
| 256 | 4,096 | 1.9307 | 1.7676 | 1.7379 | 88.3 | 91.9 | +0.1669 | [+0.0959, +0.2379] | 0.753 | 0.764 |
| 384 | 6,144 | 1.6770 | 1.5573 | 1.5588 | 93.9 | 96.6 | +0.1205 | [+0.0863, +0.1547] | 0.779 | 0.787 |
| 512 | 8,192 | 1.5534 | 1.4588 | 1.4684 | 96.7 | 98.8 | +0.0957 | [+0.0642, +0.1272] | 0.791 | 0.798 |
| 768 | 12,288 | 1.4609 | 1.4084 | 1.4137 | 98.8 | 99.9 | +0.0531 | [+0.0222, +0.0839] | 0.798 | 0.799 |
| 1023 | 16,368 | 1.4061 | 1.4061 | 1.4061 | 100.0 | 100.0 | +0.0000 | [-0.0000, +0.0000] | 0.800 | 0.800 |

The three ladders are the registered curves. Reading them together:

1. **$R^\*$ does not move over this range.** The predictive code enters the 2 %-of-gap
   band at **8 192 bits at every length**, and in both estimators. Across 156–236
   tokens the bit requirement behaves like a property of the map from state to
   next-token distribution, not of how much context the state has absorbed. Whether
   that holds over a decade of context is C1 (§19.8) and is not settled here.
2. **Below ~1 kbit the copy code wins, significantly.** At L = 256 it is better by
   $0.64$ nats at 256 bits, $0.88$ at 128 and $0.39$ at 16, with intervals excluding zero,
   and by $0.29$ at 512 with an interval that includes it; at L = 128 by $0.84$ at 256,
   $0.92$ at 128 and $0.31$ at 512; at L = 64 by $0.78$ at 256, $0.60$ at 128, $0.54$ at
   512 and $0.31$ at 1 024. The copy code's advantage is not universal at the bottom: at
   16 bits at L = 64 the *predictive* code is better, by $0.46$. A low-rank code that
   keeps the state's variance wins until the budget is large enough to pay for the
   readout's directions.
3. **The two codes converge at the top of the ladder** (identical at $r = 1\,023$,
   where 1 023 features fit a 1 023-dimensional centred target exactly), so the floor is
   the least-squares interpolation error of a saturated fit, not a property of either
   code.
4. **Top-1 tracks the curves but not their ordering**: at L = 64 the copy code is ahead
   (0.761 vs 0.758), at L = 128 they tie (0.785 / 0.785), at L = 256 the predictive code
   is ahead (0.798 vs 0.791). Decision accuracy does not separate the codes the way the
   loss does.

### 19.4 The estimation floor: size, direction, and what it does to the gap

The curves do not reach the model's own loss. This is the estimator, and it must be
sized before any verdict is read off the curves.

| L | target H | floor CE | floor / H | pred train CE at top | pred test CE at top | generalization gap | out-of-fold floor | out-of-fold floor / H |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.3224 | 1.5588 | 4.83 | 0.4982 | 1.5904 | +1.0923 | 1.6480 | 5.15 |
| 128 | 0.2838 | 1.4362 | 5.06 | 0.4723 | 1.4362 | +0.9639 | 1.4651 | 5.39 |
| 256 | 0.2719 | 1.4061 | 5.17 | 0.4467 | 1.4061 | +0.9594 | 1.2765 | 5.09 |

| r | R (bits) | L=64 gap [95 % CI] | L=128 gap [95 % CI] | L=256 gap [95 % CI] | L=64 out-of-fold | L=128 out-of-fold | L=256 out-of-fold |
|---|---|---|---|---|---|---|---|
| 1 | 16 | +0.463 [+0.260, +0.666] | -0.216 [-0.422, -0.009] | -0.385 [-0.565, -0.205] | +0.533 | -0.012 | -0.090 |
| 8 | 128 | -0.601 [-1.013, -0.189] | -0.924 [-1.305, -0.543] | -0.883 [-1.239, -0.528] | -0.537 | -0.573 | -0.473 |
| 16 | 256 | -0.784 [-1.018, -0.551] | -0.836 [-1.084, -0.588] | -0.636 [-0.927, -0.345] | -0.621 | -0.705 | -0.497 |
| 32 | 512 | -0.543 [-0.837, -0.248] | -0.312 [-0.551, -0.073] | -0.288 [-0.582, +0.007] | -0.564 | -0.285 | -0.242 |
| 64 | 1,024 | -0.313 [-0.569, -0.057] | -0.076 [-0.306, +0.154] | -0.092 [-0.343, +0.159] | -0.461 | -0.327 | -0.235 |
| 128 | 2,048 | -0.116 [-0.257, +0.025] | +0.114 [-0.055, +0.284] | +0.156 [-0.025, +0.337] | -0.218 | -0.075 | +0.038 |
| 192 | 3,072 | -0.040 [-0.149, +0.070] | +0.190 [+0.078, +0.303] | +0.207 [+0.112, +0.302] | -0.135 | +0.056 | +0.097 |
| 256 | 4,096 | -0.048 [-0.150, +0.054] | +0.123 [+0.042, +0.204] | +0.167 [+0.096, +0.238] | -0.081 | +0.082 | +0.135 |
| 384 | 6,144 | +0.014 [-0.035, +0.064] | +0.112 [+0.058, +0.166] | +0.120 [+0.086, +0.155] | -0.037 | +0.094 | +0.139 |
| 512 | 8,192 | +0.026 [-0.020, +0.071] | +0.111 [+0.052, +0.170] | +0.096 [+0.064, +0.127] | -0.008 | +0.078 | +0.098 |
| 768 | 12,288 | -0.032 [-0.056, -0.007] | +0.032 [-0.003, +0.067] | +0.053 [+0.022, +0.084] | +0.004 | +0.040 | +0.052 |
| 1023 | 16,368 | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] | +0.000 | -0.000 | -0.000 |

| L | R\* (bits) | R\* / state (49 152 bits) | mean context | KV bits at that context | R\* / KV | plateau drop beyond R\* | fraction of gap |
|---|---|---|---|---|---|---|---|
| 64 | 8,192 | 16.7 % | 156 | 15,307,154 | 0.0535 % | -0.0501 | -1.14 % |
| 128 | 8,192 | 16.7 % | 188 | 18,465,700 | 0.0444 % | -0.0522 | -1.16 % |
| 256 | 8,192 | 16.7 % | 236 | 23,174,320 | 0.0353 % | -0.0526 | -1.18 % |

**Size.** The plateau sits **4.83× / 5.06× / 5.17×** the target's entropy at
L = 64 / 128 / 256, and by construction no point on either curve goes below it. At the
top rung the train CE is 0.4982 / 0.4723 / 0.4467 against a test CE of 1.5904 / 1.4362 /
1.4061, so **+1.0923 / +0.9639 / +0.9594 nats of the floor is train-to-test generalization
error** of a linear readout with 3 072 inputs at 9 027 / 17 895 / 27 650 sample
positions. The out-of-fold estimator, which scores every position from a fit that never
saw its problem, puts the floor at 1.6480 / 1.4651 / 1.2765 — the same height within a
tenth of a nat at L = 64 and 128, and 0.13 nats lower at L = 256.

**Direction and effect on the between-code gap.** The floor inflates both codes' CE at
every rung, since both are scored against the same target on the same rows; the question
is whether it also manufactures the *difference*. It does not, and the two estimators
bracket the answer:

| L | floor (full) | gap at $R^\*$ (full) | floor (out-of-fold) | gap at $R^\*$ (out-of-fold) |
|---|---|---|---|---|
| 64 | 1.5588 | +0.0258 [−0.0198, +0.0714] | 1.6480 | −0.0080 [−0.0591, +0.0432] |
| 128 | 1.4362 | +0.1112 [+0.0525, +0.1699] | 1.4651 | +0.0780 [+0.0454, +0.1105] |
| 256 | 1.4061 | +0.0957 [+0.0642, +0.1272] | 1.2765 | +0.0976 [+0.0698, +0.1255] |

The floor moves by at most 0.13 nats between the two estimators and the gap at $R^\*$
by at most 0.034; they are independent because the gap is a within-row difference, in
which any shift common to both codes cancels. At L = 128 the estimator with the
*higher* floor gives the *smaller* gap, so the residual floor attenuates the
between-code difference rather than inflating it; at L = 256 the floor drops by 0.13
nats and the gap does not move. The measured bias is therefore, in direction, against
the predictive code.

**What the floor does not explain.** $R^\*$ is 8 192 bits in all six analyses, at two
sample sizes, two split schemes and two test sets. A data-limited $R^\*$ would move with
$n$; this one does not. The floor limits the *level* every curve reaches and the
precision of each interval, and it leaves the location of the plateau and the ordering of
the two codes intact.

**What remains confounded, stated plainly.** No code reaches $H(Y \mid X)$
(0.3224 / 0.2838 / 0.2719 nats), so we never observe "the code reaches the full-context
loss", which is what §15's prediction is phrased as. We observe where each curve stops
falling, which is a different statement. With a better-conditioned readout — more
positions than this corpus can supply, or a nonlinear decoder that §15's linear codes do
not include — every curve would move down, and $R^\*$ would move with them; the comparison
at matched $R$ is what survives that.

### 19.5 The verdicts

Convention: a kill condition **fires** (verdict *pass*) if the measurement meets it, which
refutes the claim it was built to kill. *Fail* means the condition is not met and the
claim survives. *Undecidable-at-this-scale* is used only with a named mechanical reason.

**K1 — "the index becomes unnecessary": fail (the kill does not fire).**

K1 fires only if the copy code matches the predictive code at $R \approx R^\*$ *for all
three lengths*. Per length, at $R^\* = 8\,192$ bits, with 95 % problem-clustered intervals:

| L | copy CE | pred CE | gap (nats) | 95 % CI | gap as % of the total gap | copy top-1 | pred top-1 |
|---|---|---|---|---|---|---|---|
| 64 | 1.6687 | 1.6406 | +0.0258 | [−0.0198, +0.0714] | 0.6 % | 0.761 | 0.758 |
| 128 | 1.6016 | 1.4884 | +0.1112 | [+0.0525, +0.1699] | 2.5 % | 0.785 | 0.785 |
| 256 | 1.5534 | 1.4588 | +0.0957 | [+0.0642, +0.1272] | 2.1 % | 0.791 | 0.798 |

The copy code does **not** match at L = 128 and L = 256 (intervals exclude zero: it is
worse by 0.111 and 0.096 nats, closing 96.3 % and 96.7 % of the gap against the predictive
code's 98.8 %), and at L = 64 the difference is +0.0258 with an interval that includes
zero. The condition requires matching at all three; it fails at two. Under the
out-of-fold estimator the same two lengths separate (+0.0780 [+0.0454, +0.1105] and
+0.0976 [+0.0698, +0.1255]) and L = 64 stays unresolved (−0.0080 [−0.0591, +0.0432]).

*Sign stability.* The sign of the gap at $R^\*$ is **positive in five of six analyses**
(the sixth, L = 64 out-of-fold, is −0.008 with an interval spanning zero): the predictive
code is never worse at $R^\*$. What is *not* stable is the significance — it is resolved
only from L = 128 up — and what does *not* happen is growth: +0.026, +0.111, +0.096 nats
as context goes 156 → 188 → 236 tokens. **The advantage appears between L = 64 and
L = 128 and then flattens; it neither grows with context nor disappears.**

*Where the copy code still wins.* The rung below which the copy code is significantly
better moves the other way, and monotonically: 1 024 bits at L = 64, 512 at L = 128, 256
at L = 256. Longer context shrinks the region where copying the state's variance beats
coding its predictions, but it does not raise the height the predictive code reaches.

**K2 — "$R^\*$ is not small": fail (the kill does not fire), re-derived.**

Beyond $R^\*$ the predictive curve falls **0.0501 / 0.0522 / 0.0526 nats**
(1.14 % / 1.16 % / 1.18 % of the gap) as the budget goes from 8 192 to 16 368 bits, and
the out-of-fold estimator gives 1.76 % / 1.50 % / 1.50 %. A plateau exists at every
length: the condition — "keeps falling beyond $R^\*$ with no plateau" — is not met.
$R^\*$ is also *invariant* at 8 192 bits across these three lengths, which is the
opposite of the growth K2 was built to detect; whether the invariance survives past
236 tokens is C1's first clause (§19.8), not a result here.

The caveat that limits this verdict, stated in §19.4 and not papered over: the plateau is
~5× the target's entropy, so what plateaus is the estimator, and the claim "the predictive
code reaches the full-context loss at $R^\*$" is *not* established here. K2 answers only
whether the curve stops falling, and it does.

**K3 — "no compression at all": fail (the kill does not fire), and this one is
floor-independent.**

The registered threshold is $R \ge 0.5\,(2Ldb)$ for *both* codes. Measured, per length:

| L | $R^\*$ | $R^\*$ / per-position state (49 152 bits) | KV bits at the measured mean context | $R^\*$ / KV |
|---|---|---|---|---|
| 64 | 8 192 | 16.7 % | 15 307 154 | 0.0535 % |
| 128 | 8 192 | 16.7 % | 18 465 700 | 0.0444 % |
| 256 | 8 192 | 16.7 % | 23 174 320 | 0.0353 % |

The predictive code saturates at 8 192 bits — one sixth of the single-position state — and
the copy code needs 12 288 bits to enter the same band at all three lengths. Against the
KV a transformer must actually carry at the measured context (2·L·3072·16 bits) that is
0.05 % or less, and the ratio *falls* with length because the state's budget is fixed
while the KV grows. A rank-$r$ linear code cannot require more than its rank ceiling, so
this verdict does not depend on the estimation floor: **the state is compressible to a
fraction of itself for the next-token target.** That is a statement about the
next-token distribution only: at the same 8 192 bits the code that wins this
comparison loses recall against a variance-preserving code (§19.9). The ratio to the KV falls with
context only because the KV grows while the state does not; the state's own
requirement neither falls nor rises across 156–236 tokens. *[rev. 2026-11-17]*

### 19.6 What this settles, and what it does not

Settled by this measurement:

- **The hidden state is not irreducible** (K3). 8 192 of 49 152 bits retain 98 % of the
  achievable loss reduction (the 2 %-of-gap band the registration uses), and the ratio
  to the KV carried at that context falls with context because the KV grows.
- **The curve plateaus** (K2), at a bit budget that does not grow across the measured
  range.
- **The capacity advantage is real but small, and it is not a length effect.** At matched
  bits the predictive code is never worse at $R^\*$ and is significantly better by
  0.078–0.111 nats at L ≥ 128; the copy code is significantly better below 256–1 024
  bits and at least as good for decisions at L ≤ 128.
  §12 predicts that the predictive code reaches the optimum at $R^\*$ while the
  copy needs $R \propto L$. What F5 shows is a gap of 0.6–2.5 % of the total loss
  reduction that does not grow with $L$, and a copy-code requirement of 12 288 bits
  to reach the same band at all three lengths, which does not grow with $L$ either.
  The *direction* of §12 survives; the *magnitude* it implies does not appear.

Not settled:

- **The entropy target: not attained at $R^\*$, attained at the ceiling — and the
  attribution differs at the two rates.** Two computations settle what the previous
  version of this paragraph left open. Neither needs a model run: both come from the
  same `h_t` rows and targets already on disk.

  *The oracle.* Every code in §19 decodes linearly from a rank-$r$ bottleneck, so its
  output lies in an $r$-dimensional affine subspace of the 1 024-dimensional log-prob
  space. The best such subspace can be chosen *with knowledge of the targets* (fitted on
  p01–p20, applied to p21–p32), which makes it an upper bound on any code of the family
  at $16r$ bits and therefore a lower bound on the achievable CE at that rate. At
  $r = 1\,023$ the projection is (almost) the identity on the target space, so the
  oracle returns $H(Y \mid X)$ and the construction checks itself.

| L | $H(Y\mid X)$ | 256 bits | 1 024 | 2 048 | 4 096 | **8 192** | 12 288 | 16 368 |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.3224 | 5.7685 | 3.4030 | 2.5408 | 1.8176 | **1.2304** | 0.7228 | 0.3224 |
| 128 | 0.2838 | 5.8543 | 3.7041 | 2.5539 | 1.7525 | **1.0799** | 0.6683 | 0.2838 |
| 256 | 0.2719 | 5.9714 | 3.5723 | 2.5143 | 1.7600 | **1.1026** | 0.6878 | 0.2827 |

| L | $H(Y\mid X)$ | oracle CE at $R^\*$ | oracle / $H$ | measured CE at $R^\*$ | measured − oracle | oracle at 16 368 | measured at 16 368 | measured − oracle there |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.3224 | 1.2304 | 3.82 | 1.6406 | 0.4101 | 0.3224 | 1.5904 | 1.2680 |
| 128 | 0.2838 | 1.0799 | 3.81 | 1.4884 | 0.4085 | 0.2838 | 1.4362 | 1.1524 |
| 256 | 0.2719 | 1.1026 | 4.05 | 1.4588 | 0.3562 | 0.2827 | 1.4061 | 1.1235 |

  At **8 192 bits the oracle scores 1.2304 / 1.0799 / 1.1026 nats — 3.82× / 3.81× /
  4.05× $H$** — so *no* member of the family (target-fitted, variance-preserving or
  literal RRR) attains the entropy target at $R^\*$ on these three lengths. The binding
  constraint is **rate**: the target needs about 1 000 of its 1 024 dimensions (768 dims
  still cost 0.6683–0.7228 nats above $H$; 1 023 dims reach it). The estimator supplies
  0.3562–0.4101 nats of the 1.19–1.32-nat excess at $R^\*$ — 30–34 % — so it is not
  what blocks attainment there.

  At the **rank ceiling the attribution reverses.** The oracle reaches $H$ (0.3224 /
  0.2838 / 0.2827; the 0.0107-nat shortfall at L = 256 is the 1 023-dim truncation of a
  1 024-dim target space, not a modelling error), while all three codes coincide at
  1.5904 / 1.4362 / 1.4061 — 1.2680 / 1.1524 / 1.1235 nats above it. **Every nat of the floor's distance above $H$ is therefore estimator
  error** — the 4.83–5.17× multiple is not a rate limit and not a state-dimension
  limit: the state carries what is needed, and the linear readout does not extract
  it.

  *And the estimator residual is not sample-count-limited.* Across the three lengths the
  ceiling excess falls only from 1.2680 to 1.1235 as $n$ goes from 9 027 to 27 650. A
  $1/n$ law would have taken it to 0.414; the exponent fitted to that 3.06× range is
  0.11, so halving the residual would need about 600× the positions (≈ 17 million). The
  binding constraint for the floor is the **linear readout from a rank-1 023 projection
  of a 3 072-float state** — a decoder-class constraint, which a nonlinear readout could
  in principle remove and which §19.8 now makes the next test.

  *Scope, unchanged.* These are 156–236-token contexts. The oracle is exact at the
  ceiling because the *target*, not the context, has been fully spanned; that is a
  statement about a 236-token prefix and nothing here says what happens at 1k–32k
  tokens. Nor does anything here measure a nonlinear decoder: the bound is a bound on
  linear decoders, which is exactly the family §15 registered. *[rev. 2026-11-17b]*
- **The context range: nothing here is long context.** §15 registers
  $L \in \{\text{1k}, \text{8k}, \text{32k}\}$. The corpus supplies mean contexts of
  156, 188 and 236 tokens — a 1.5× span, from 8 streams of ≤ 256 generated tokens on
  ~100-token prompts. The three lengths are real, measured and independent in content,
  but they sit inside the first 240 tokens of a passage. Every clause measured here is
  a property of a ≤ 236-token prefix: the 8 192-bit plateau, the 12 288-bit copy
  requirement, the 2 %-of-gap gap, and the *absence* of growth in either. None of it is
  extrapolated to 1k, 8k or 32k tokens in this document, and the copy code could not
  show $\Theta(t)$ growth at any length in any case, because its ceiling is the state's
  own 49 152 bits.
- **Nonlinear codes.** Both codes are linear, as §15's caveat says. A nonlinear
  predictive code is untested and nothing here refutes it.
- **The target.** Reproducing a 64-way distribution is strictly stronger than making the
  decision, so 8 192 bits is an upper bound on what a decision needs; the top-1 column is
  the decision-level reading, and it does not favour the predictive code at L ≤ 128.

### 19.7 Reproduction and verification

```
python harness/run_f5_capture.py --problems 1-32 --max-tokens 64  --prefix _f5L64
python harness/run_f5_capture.py --problems 1-32 --max-tokens 128 --prefix _f5L128
python harness/run_f5_capture.py --problems 1-32 --max-tokens 256 --prefix _f5L256
python harness/f5_compare.py --prefix _f5L64  --out results/f5L64_c1024.json
python harness/f5_compare.py --prefix _f5L128 --out results/f5L128_c1024.json
python harness/f5_compare.py --prefix _f5L256 --out results/f5L256_c1024.json
python harness/f5_compare.py --prefix _f5L64  --cv 2 --out results/f5L64_cv2.json
python harness/f5_compare.py --prefix _f5L128 --cv 2 --out results/f5L128_cv2.json
python harness/f5_compare.py --prefix _f5L256 --cv 2 --out results/f5L256_cv2.json
python harness/f5_doc_tables.py    # writes the tables above from the artifacts
python harness/f5_stage.py --prefix _f5L64  --ref results/f5L64_c1024.json
python harness/f5_stage.py --prefix _f5L128 --ref results/f5L128_c1024.json
python harness/f5_stage.py --prefix _f5L256 --ref results/f5L256_c1024.json
python harness/f5_oracle.py --prefix _f5L64   # oracle rank-r ceiling, §19.6
python harness/f5_oracle.py --prefix _f5L128
python harness/f5_oracle.py --prefix _f5L256
python harness/f5_verify.py        # exits 0 iff §19 reproduces from the raw sidecars
python harness/f5_recall.py --prefix _f5L64 --ref results/f5L64_c1024.json  # §19.9
python harness/f5_recall.py --prefix _f5L128 --ref results/f5L128_c1024.json
python harness/f5_recall.py --prefix _f5L256 --ref results/f5L256_c1024.json
python harness/f5_mixed.py --prefix _f5L64 --ref results/f5L64_c1024.json      # §19.9b
python harness/f5_mixed.py --prefix _f5L128 --ref results/f5L128_c1024.json
python harness/f5_mixed.py --prefix _f5L256 --ref results/f5L256_c1024.json
python harness/f5_mixed.py --prefix _f5L64 --alphas 1.0,0.9,0.0 \
    --rungs 8,16,32,64,96,128,192,256,384,512,768,1023 --out results/f5L64_sweep.json   # §19.9c
    (repeat for _f5L128 / _f5L256, and with --lag 4 --out ..._sweep4.json)
python harness/f5_sweep_table.py   # results/f5_sweep.md
python harness/f5_doc_check.py     # exits 0 iff every table cell above matches the artifacts
python harness/theory_check.py     # exits 0; checks 1-11 are unaffected by this section
```

`f5_verify.py` re-parses the sidecars, re-joins on `(stream, step)`, rebuilds the
candidate set and the censoring floor, and recomputes per length: paired rows, kept rows,
target entropy, no-state CE, the copy code's test and train CE at $r = 64$, and the
out-of-fold copy CE at $r = 256$ under the same two-fold split. Only the copy code is
re-derived numerically — it needs no ridge and no target-fitted encoder — so the
verification shares no estimator code with the analysis. Two flags exist for the
machine this was run on: `--fast` builds the copy basis from the Gram matrix (identical
to the SVD basis to $10^{-7}$ nats, and it keeps each length inside one command window)
and `--no-cv` skips the out-of-fold checks. `f5_stage.py` exists because the full
ladder's train CE for L = 256 did not finish in one window: it re-derives the artifact's
test CE at $r = 64$ and $r = 1023$ first (all six rows match to $10^{-4}$) and only then
records the train CE, so the floor table's train column has one provenance at all three
lengths.  fits the target's own top-r subspace on p01–p20 and projects the
p21–p32 rows onto it: that is the best any rank-r linear decoder can do, known
targets and all, and it is the bound §19.6 quotes.

### 19.8 The next claim, stated so that it can fail

§19.6 now pins the loss to two different causes at two different rates, which changes
which claim is sharpest. At 8 192 bits the wall is **rate**: the oracle subspace scores
1.0799–1.2304 nats and no linear decoder can beat it. At the rank ceiling the wall is
the **linear readout**: the oracle reaches $H$ and the measured code is 1.1235–1.2680
nats above it. The second decides whether the design is viable, and it is decidable in
closed form — same captures, same targets, a different decoder, no LLM run:

> **N1 (the nonlinear-readout claim).** At the same 8 192-bit budget, a decoder that is
> nonlinear in the state — still closed-form and untrained, e.g. ridge on quadratic or
> random features of $h$ — beats the linear-oracle bound and closes a measurable part of
> its gap to $H$.

**Direction and magnitude that would count as support.** CE at $R^\*$ **≤ 1.00 nats at
all three lengths**, for the same 512 coefficients × 16 bits stored. That is at least
0.08 nats below the best possible linear decoder at every length (0.23 at L = 64) and
still 3.10–3.68× $H$; the no-state CE is 5.87–5.94, so the bar is far from free.

**What would refute it.** Every nonlinear closed-form readout landing at or above the
linear oracle (1.0799–1.2304). That would prove the 512-dimensional target subspace,
not the readout's linearity, is the wall — and would force the carrier's budget up to
12 288–16 368 bits, where the table above says the oracle costs 0.6683–0.7228 and
0.2827–0.3224 nats respectively.

**Why this replaces C1 as the sharpest.** C1 — the separation claim, that the copy
code's requirement grows with context while the target-fitted one stays bounded — is
still open and still the right long-context test, but it needs a capture the runner has
failed three times, and §19.6 has just shown its copy half is capped at 16 368 bits by
the state's own dimension, so it can only ever be a lower bound. N1 needs no capture at
all and decides a larger question: whether a rank-512 carrier can reach the target's
entropy at 8 192 bits when the readout is allowed to be nonlinear. C1 is second in
line.

**What the current artifacts already give N1.** The bar it must clear
(1.0799–1.2304), the floor it cannot beat at any rate (0.2827–0.3224 at the ceiling),
and the linear result it must improve on (1.4588–1.6406). A nonlinear readout landing
between 1.00 and 1.08 would be the informative middle: proof that linearity costs most
of the gap while rate costs the rest. *[rev. 2026-11-17b]*

### 19.9 Recall at matched bits: the CE-optimal code is not the memory code

The CE ladder asks how well a rank-$r$ code predicts the next token. That is the axis the
long-context literature says hides recall failure (standard perplexity correlates
−0.11 with LongBench, +0.31 with LongEval, +0.33 with RULER). The sidecars carry the
emitted id per position, so the same question can be asked on the same rows: can the code
recover the token emitted $k$ steps back? Same three bases, same 16-bit coefficients, same
least-squares decoder; only the kept subspace differs. `recall-fitted` is the oracle basis
for this task — the ridge readout onto the one-hot of $\text{tok}_{t-k}$.

| L | lag | chance | copy | pred | recall-fitted |
|---|---|---|---|---|---|
| 64 | 1 | 0.101 | 0.881 | 0.827 | 0.918 |
| 128 | 1 | 0.112 | 0.803 | 0.725 | 0.891 |
| 256 | 1 | 0.115 | 0.784 | 0.704 | 0.898 |
| 64 | 4 | 0.072 | 0.389 | 0.337 | 0.403 |
| 128 | 4 | 0.112 | 0.280 | 0.261 | 0.306 |
| 256 | 4 | 0.113 | 0.256 | 0.239 | 0.283 |

At $R^\* = 8\,192$ bits ($r = 512$), top-1 accuracy on held-out problems.

1. **The CE-optimal code is the worse memory code, at every length and every rung.** At
   8 192 bits the predictive basis recovers 0.827 / 0.725 / 0.704 of lag-1 tokens against
   the copy basis's 0.881 / 0.803 / 0.784 — a 5–8 point loss for a code that is
   simultaneously 0.03 nats better on the CE ladder (§19.3). Fitting one code to both targets recovers the memory at a 1–3 % CE cost
   (§19.9b), so the split is a property of the objective, not of the code: what
   §4 asks for is a state trained on both, not two separate states.
2. **Recall degrades with context at fixed bits**: copy 0.881 → 0.784 and predictive
   0.827 → 0.704 as mean context goes 156 → 236 tokens. The capacity × length trade of
   the fixed-state literature appears inside a real 3B model's hidden state.
3. **Beyond lag 4 there is no linearly decodable token identity at all.** At lag 16 and
   64 *every* arm is at or below the marginal predictor — including the recall-fitted
   oracle (0.063–0.075 against chance 0.11). The limit there is the state, not the code.
4. The oracle leaves 4–9 points at lag 1 (0.891–0.918 against copy 0.784–0.881), so the
   copy code is near, but not at, the best recall a rank-512 code allows.

Scope: linear decoders, candidate sets restricted to ids seen ≥ 20 times in training,
one model, 156–236-token contexts. This is the axis the CE verdicts of §19.5 do not
cover: K3's "the state is compressible" is a statement about the next-token
distribution, and this section shows the same compression costs memory. *[rev. 2026-11-17c]*

### 19.9b The split is in the objective, not in the code

Fitting *one* code to a mix of the two targets — $M(\alpha) = [\sqrt{\alpha}\,\hat
Y_{\text{ce}}, \sqrt{1-\alpha}\,\hat O_k]$ with each block unit Frobenius norm, the
exact rank-$r$ solution of $\alpha\|Y_{\text{ce}} - PY_{\text{ce}}\|^2 +
(1-\alpha)\|O_k - PO_k\|^2$ at the §19 $\lambda$ — shows the dissociation above is not a
property of rank-512 codes:

| L | copy CE | copy recall | α=1 CE | α=1 recall | α=0.9 CE | α=0.9 recall | CE cost |
|---|---|---|---|---|---|---|---|
| 64 | 1.7617 | 0.884 | 1.7212 | 0.849 | 1.7404 | 0.914 | 1.1 |
| 128 | 1.6344 | 0.804 | 1.5342 | 0.753 | 1.5697 | 0.854 | 2.3 |
| 256 | 1.5654 | 0.780 | 1.4667 | 0.737 | 1.5124 | 0.852 | 3.1 |

At $r = 512$; $\alpha = 0$ (pure recall) reaches 0.911 / 0.857 / 0.857 at CE 4.98 / 4.92 /
4.73, so one code at $\alpha = 0.9$ recovers 99–100 % of the recall ceiling for 1–3 % CE.

1. **The CE cost of memory is small at 8 192 bits and grows with context**: +0.019 /
   +0.036 / +0.046 nats (+1.1 / +2.3 / +3.1 %) to buy back 6.5–12 points of lag-1 recall,
   at 156 / 188 / 236 tokens.
2. **The mixed code dominates the PCA copy code on both axes at every length** (CE 1.74 /
   1.57 / 1.51 against 1.76 / 1.63 / 1.57, recall 0.91 / 0.85 / 0.85 against 0.88 / 0.80 /
   0.78), so variance preservation is not the right prior even for memory.
3. The pure-recall code is unusable for prediction (CE 4.7–5.0), so the two axes are far
   apart when each is taken alone; the near-flat stretch from $\alpha = 0.9$ to $0.25$ is
   where a real design would sit.

Read this as a correction to §19.9: a state fitted to the next-token target alone is a poor
memory, but a state fitted to both is both, so §4's split is a statement about objectives,
not about how many codes a state can hold. Fitted on the lag-1 covered train rows, so these
CE values sit ~0.05–0.09 nats above the §19.3 ladder at the same rank; the arms are
internally comparable. *[rev. 2026-11-17d]*

### 19.9c What memory costs in bits

Sweeping the rank at $\alpha = 0.9$ and asking for the cheapest code that reaches 90 % of
that length's full-rank recall (bits $= 16r$, grid $r \in \{8, \dots, 1023\}$):

| lag | L | ceiling | copy | α=1 | α=0.9 | α=0 |
|---|---|---|---|---|---|---|
| 1 | 64 | 0.917 | 4096 | 8192 | 2048 | 1024 |
| 1 | 128 | 0.861 | 8192 | 12288 | 3072 | 1536 |
| 1 | 256 | 0.861 | 8192 | 12288 | 4096 | 1536 |
| 4 | 64 | 0.403 | 6144 | 6144 | 1536 | 512 |
| 4 | 128 | 0.296 | 6144 | 6144 | 1536 | 512 |
| 4 | 256 | 0.283 | 8192 | 12288 | 3072 | 512 |

Memory is cheapest in a code fitted to carry it: 1.5–4 kbit against 4–8 kbit for the
variance code and 6–12 kbit for the code fitted to prediction alone, at both lags and
every length. Lag 1 flatters every arm by autocorrelation — copy needs 4 kbit at lag 1 but
6–8 kbit at lag 4 — and the ordering survives that correction, so the advantage is memory
rather than recency. Read with §19.9b the exchange runs both ways: at 8 192 bits memory
costs 1–3 % CE, while adding prediction to a memory code costs 2–4× the bits (512–1536 →
1536–4096 for 90 % of the lag-4 ceiling).

Ceilings are this fit's full-rank recall; the lag-1 values sit below §19.9's $r = 512$
oracle at L = 128 and 256 (0.891 and 0.898) because the fit here uses the lag-covered
rows only. *[rev. 2026-11-17e]*

 

---

## 20. Prior art, in brief

Not a survey — only what changes what we may claim. Checked 2026-11-17.

- **Matched bits, PCA vs target-fitted: already done.** MatryoshkaKV (ICLR 2025) compares
  a PCA KV basis against one distilled through KL to the teacher's next-token
  distribution at the same rank and fp16 storage: Llama-2-7B at 50 % budget scores 43.07
  (PCA) vs 59.11 (distilled) against a 61.16 baseline. §19 is the same idea on a bit
  budget with problem-clustered intervals, a bounded hidden state and a 2 % effect; the
  *direction* is not ours. SAKI (2026) and AATC (2026) do matched-rank score- and
  output-optimal versions; KVTC (ICLR 2026) reaches ~20× on the KV with plain PCA +
  waterfilling.
- **Compressing the state is the field's default.** Palu, xKV, SALS, DLA, MLA. K3's
  "8 192 of 49 152 bits" is a different object (the hidden state, not the KV) but not a
  new kind of claim.
- **The recall capacity law is known.** Ω(N) bits of state for N associations (Based
  Thm 3.1; Zoology H.6), `err > 1 − |S|/D^L` (Jelassi, ICML 2024), a sharp threshold at
  `d²/(n log n) = 2` (2026), and RWKV-7 recalling 72.9 % of 256 pairs with an 8 192-dim
  state (≈0.55 bits/dim). So a fixed state cannot win recall by scaling, which is why
  §4.2 routes memory through an index rather than a bigger carrier.
- **Perplexity is the wrong metric for that axis** — the reason §19.9 exists. Hybrid
  linear attention matches LM accuracy across every ratio while RULER goes 0.256 → 0.436;
  Triadic Linear Attention ties on PPL (11.25 vs 11.11) and loses recall 26.2 vs 41.6;
  compressed KV can keep LongBench QA while RULER retrieval falls 92.7 → 6.1.
- **The memory-component literature is dense and none of it is ours.** Titans, SDM, SSE,
  Larimar, Engram/DSE (whose sparsity-allocation law is the closest existing formal
  statement to the cost law in §4), Memory Layers at Scale. Each supplies a component;
  none proposes three roles with three growth axes, which remains the memo's framing
  claim rather than a mechanism claim.
- **The sufficiency statement is classical in ingredients.** Belief State Transformer
  (ICLR 2025) proves a next-token-optimal predictor that is not a belief state; causal
  states (Shalizi–Crutchfield) are the minimal sufficient statistic for the whole future.
  §11–§12 is a clean special case, not a new theorem.

**What is ours, after this check:** the oracle bound on rank-$r$ linear decoders of the
next-token target and its per-context rate–distortion curve (§19.6) — no match found;
the estimator/rate attribution at the two rates; the three-length invariance of $R^\*$
with problem-clustered intervals; and now the CE-versus-recall dissociation measured
inside one model at matched bits (§19.9). Everything else in the memo has prior art and
should be cited, not claimed. *[rev. 2026-11-17c]*

---

## 21. The parity test (executed)

**Criterion, fixed before the run.** The same held-out problems (p21–p32) and the same §19
rows; bits counted identically on both sides: parameters at deploy precision, per-position
state, and every cache that grows with context. Parity means capability within 0.01 nats of
CE or 0.02 of recall; a win also requires strictly fewer bits. Verdicts are per slice, never
for the model as a whole.

**Systems.** The LLM is Llama-3.2-3B-Instruct Q4_K_M (28 layers, 3,072 hidden, 8 KV heads,
128 head dim, 128,256-token vocabulary); its capability is scored on its own sampled
continuation, so its CE is $H(Y \mid X)$ by construction. The candidate is plan v0, closed
form and untrained: an order-4 interpolated backoff n-gram over emitted ids as carrier, plus
a verbatim index of the last 16 ids. It sees only generated tokens — the artifacts carry no
prompt ids — which handicaps it and is part of the record. There is no workspace in v0.

**Prediction.**

| L | LLM (own belief) | n-gram carrier | LLM bits/pos | candidate bits/pos | verdict |
|---|---|---|---|---|---|
| 64 | 0.3224 | 4.6377 | 1.429e+08 | 433.2 | lost |
| 128 | 0.2838 | 4.8061 | 1.724e+08 | 429.1 | lost |
| 256 | 0.2719 | 4.7580 | 2.163e+08 | 412.0 | lost |

**Memory.** The LLM's fixed state alone (49 152 bits/position) cannot recall; its transcript
can, at the KV price. Both baselines are priced; the index is one system.

| L | lag | fixed state alone | index | LLM transcript | LLM bits/pos | candidate bits/pos | verdict |
|---|---|---|---|---|---|---|---|
| 64 | 1 | 0.913 | 1.000 | 1.000 | 1.429e+08 | 433.2 | supported |
| 64 | 4 | 0.324 | 1.000 | 1.000 | 1.429e+08 | 433.2 | supported |
| 64 | 16 | 0.048 | 1.000 | 1.000 | 1.429e+08 | 433.2 | supported |
| 128 | 1 | 0.904 | 1.000 | 1.000 | 1.724e+08 | 429.1 | supported |
| 128 | 4 | 0.284 | 1.000 | 1.000 | 1.724e+08 | 429.1 | supported |
| 128 | 16 | 0.064 | 1.000 | 1.000 | 1.724e+08 | 429.1 | supported |
| 256 | 1 | 0.907 | 1.000 | 1.000 | 2.163e+08 | 412.0 | supported |
| 256 | 4 | 0.280 | 1.000 | 1.000 | 2.163e+08 | 412.0 | supported |
| 256 | 16 | 0.069 | 1.000 | 1.000 | 2.163e+08 | 412.0 | supported |

**Verdicts.**

1. **Prediction: lost, by 4.3–4.5 nats.** The carrier is not empty (the no-state marginal is
   5.94 nats at L = 64, so the table removes 1.1–1.3), but it is nowhere near the model's own
   belief.
2. **Memory: supported at both baselines.** The index recalls every lag exactly for 412–433
   bits/position, against 49 152 for the fixed state that only reaches 0.913 / 0.324 / 0.048,
   and against 1.43–2.16 $\times 10^8$ for the transcript that actually can. Same
   capability as the transcript, $\sim 10^5$ fewer bits.
3. **Joint behaviour: lost.** Because prediction fails, the whole-behaviour comparison fails:
   the candidate matches memory but not language, so it is not a cheaper substitute for the
   model.

**What this establishes, and what it does not.** It establishes the mission's claim for the
memory slice only: memory held as addresses costs roughly five orders of magnitude fewer bits
than the same memory held as a transcript, at equal capability. It does not establish a
cheaper general representation, and it cannot at this scale — every carrier that can be fitted
here sees 10,080–31,286 tokens across 160 sequences, so the prediction gap is dominated
by data rather than by architecture. Two further caveats: the CE target is the model's own sampled continuation,
which flatters the LLM (a human-text CE would narrow the absolute gap without closing it at
this corpus size), and the index is a lookup, so this row prices the memory *role*, not a
standalone model. The honest unit of comparison is row 3. *[rev. 2026-11-17f]*

---

## 22. The decomposition, and the density it buys

*This section audits where a transformer spends its parameter bits. The derivation from
requirements is §23; it is the design argument, and this is the supporting evidence.*

§21 established what was already obvious (addressed memory is smaller than a transcript) and
nothing about weights. This section is the weight axis: what a 3B model's parameter bits are
actually doing, which parts are redundant, and what a design that keeps only the load-bearing
parts would cost.

### 22.1 Three roles, three fractions

| role | job | share of parameters | why it is redundant |
|---|---|---|---|
| composition | mix context; the learned operator | 29 % (attention) | ALBERT: sharing attention across layers is ≈ free (+0.1 / −0.7 avg points); recursive depth + LoRA recovers most of the rest (ICLR 2025) |
| knowledge | key→value memory (\(\text{Geva}^\dagger\)) | 59 % (FFN) | one wide *shared* FFN recovers most of a per-layer set; FFNs quantise to 3 bit at ≈ no loss |
| addressing | vocabulary in and out | 12 % (embeddings) | factorisable: \(V\!\times\!d \to V\!\times\!E + E\!\times\!d\) (ALBERT), tied |

The state is not in this table: it costs 49 152 bits/position at run time (§19), compresses
6× for prediction, and is where the earlier sections were aimed. Weights are a different
budget and behave differently.

### 22.2 Two density questions that must not be conflated

**(a) Compress the same knowledge into fewer bits.** Bounded by the weights' own entropy.
Three-bit vector quantisation is near-lossless (QTIP: 5.30 vs 5.12 WikiText-2 PPL on Llama-2-7B);
two bits is not (AQLM 6.64, older 2-bit scalar methods far worse). Against the deployed 4.5-bit
Q4_K_M this is a 1.7× win and no more. Packing alone cannot produce a dense architecture.

**(b) Move knowledge out of the weights.** A dense matrix stores an association at \(d b\)
bits, because a rank-\(d\) operator holds about \(d\) of them; an index stores one at
\(\log V + \log k \approx 20\) bits. RETRO measured the exchange: a 7.5B model retrieving
from a 2T-token store matches models ~25× its size, with the retrieval parameters under 10 %
of the model, and the gain is "comparable to multiplying the parametric model size by ~10×".
But the store is 7 bytes per token (infini-gram), so this is knowledge kept as *data*: a win
on weight bits, a loss on total system bits unless the corpus is already held. The two routes
are not rivals — the weight route wins wherever one association generalises across many
contexts, the index wherever the string is unique.

### 22.3 The design, priced

Shared operator library + indexed knowledge + cheap per-depth modulation, all at 3-bit packing
so the win is not bought from quantization:

| component | bits at 3-bit | after the lever | factor | lever |
|---|---|---|---|---|
| embeddings | 1.182e+09 | 2.017e+08 | 5.9 | factorise to E=512, tied |
| attention | 2.819e+09 | 4.027e+08 | 7.0 | share across 7 layers, rank-8 modulators |
| FFN | 5.637e+09 | 8.054e+08 | 7.0 | one wide shared FFN, rank-8 modulators |
| total | 9.638e+09 | 1.410e+09 | 6.8 | |

6.8× against the *same* model at the same 3-bit packing, 11.5× against the 4.5-bit model on
disk. That is a bound *conditional* on §22.5's redundancy assumption, not a measurement of a
trained design. The three levers are independent in the ledger and each has its own published constant,
which is why the total is quoted per component rather than as one number.

### 22.4 Why it still meets the requirements

A dense model needs: a learned compositional operator (a table cannot predict — §21 measured
4.5 nats of failure), shared structure so it generalises, an addressable store for what does
not generalise, gradient flow to everything, and a state that is a belief state. The design
keeps composition in weights (shared, not tabular), keeps generalising knowledge in a wide
shared FFN, moves only unique-string knowledge to an index, trains through continuous
modulators, and leaves the state as §12's carrier.

### 22.5 What must be true, and the first thing to measure

The load-bearing assumption is that layer *groups* are functionally redundant at 3B scale in a
decoder-only LM. The published constants come mostly from BERT-scale encoders and MT models;
the closest decoder-only evidence is recursive depth with LoRA on Gemma 1B/2B. If grouping
fails, the 6.8× collapses to the embedding and packing factors (≈2.4×).

That assumption is decidable read-only, on the checkpoint already on disk: spectra and pairwise
operator similarity across layer groups, and the reconstruction error of a shared basis at a
1/7 rank budget. If a shared basis reconstructs each layer group at small error, the ledger
holds; if it does not, it does not — and the answer costs an SVD, not a training run.

\(^\dagger\) Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, EMNLP 2021.
*[rev. 2026-11-17g]*

---

## 23. What a language model must be

§22 audits a transformer's parameter budget. That is the wrong starting point: an architecture
should be derived from what a language model must do, not from where a transformer happens to
put its weights. Six requirements, each with its reason, then the structure they force.

### 23.1 The requirements

**R1 — A state is forced only under four assumptions, and all four are choices.** The standard
result — the prefix can influence the prediction only through a fixed-size accumulator — holds
under (i) a single left-to-right pass, (ii) constant compute per token, (iii) parameters fixed at
inference, (iv) a finite state. Attention is the case where (iv) is dropped and the accumulator
is the prefix itself. So "recurrence or attention" is not the space; it is **one corner** of it,
and quoting it as exhaustive is a paraphrase of a framework rather than a decomposition. Each
dropped assumption opens a family with a different answer to the density question:

| dropped | family | what it buys | what it costs |
|---|---|---|---|
| (i) single pass | iterative refinement over the whole sequence (fixed point, diffusion) | depth becomes compute; one cell, no per-depth parameters | iterative sampling; implicit gradients |
| (ii) constant compute | adaptive halting, latent thinking | compute proportional to difficulty | halting is discrete; credit assignment |
| (iii) fixed parameters | fast weights, in-context updates | state tracking from a small weight delta instead of a wide state | stability, forgetting |
| (iv) finite state | attention over the transcript | exact recall of everything | O(t) bits and compute |
| the state's *type* | typed state: stack, set, permutation, graph | capability a dense vector can only smear | differentiable data-structure ops are hard to train |

The last row is the one that was missing, and it matters most for density. A dense vector needs
$n$ dimensions to represent the symmetric group $S_n$ faithfully — the defining representation
is $n$-dimensional — so tracking $n$ entities in a vector costs $\Theta(n)$ width and $n^2$
weights in every operator that touches it. Writing the permutation out costs $n \log n$ bits and
no width at all. The state's *encoding* changes the bit complexity of state tracking, not just
its convenience, and width is the term that dominates the ledger ($d^2$ per operator). A typed
state can therefore shrink the rule quadratically in the capability it has to hold — the largest
single lever I have found that is not in the literature I was reciting.

**R2 — Expressive power outside TC⁰.** Log-precision transformers *and* log-precision SSMs are
both provably in TC⁰ (Merrill et al., 2024): at any width and depth they cannot compose
permutations — the $S_5$ word problem — and so cannot track entities, code state or narrative
state. A predictor that can only be TC⁰ is not a language model in the required sense. The escape
is input-dependent state transitions, which provably raise the class.

**R3 — Non-commutative, sign-changing transitions.** Finite-precision linear recurrences whose
transition eigenvalues lie in $[0,1]$ cannot recognise PARITY at *any* length
(Sarrof et al., 2024; Grazzi et al., 2025). Adding negative eigenvalues fixes parity,
non-triangular products are needed for mod-3 counting, and products of $I - vv^\top$ with
eigenvalues in $[-1,1]$ can represent every regular language. So a decay-only diagonal is
excluded by a theorem, and the minimal sufficient cell is matrix-valued with rotation-like
spectra — a *rotation*, not a leak.

**R4 — Content addressing.** Mamba's selection argument: input-dependent selectivity is what
separates a predictor from a counter, and it is what selective copy and induction require.
§21 measured the alternative directly: a count model with no learned rule is 4.5 nats worse
than the LLM. Reads must be by content, not position, or every association costs a full
$d$-dimensional slot and does not transfer to unseen contexts.

**R5 — Gradient stability.** $\partial s_T/\partial s_0 = \prod_t J_t$, so training over $T$
steps needs the product bounded: unit-modulus spectra or a residual path. Note that this is the
same constraint as R3 — expressiveness wants a rotation and trainability wants a rotation — which
is why the $I - vv^\top$ form is not a coincidence.

**R6 — Density: no duplicated rule.** R1–R5 constrain the *rule*; none of them mentions depth,
per-position operators or per-layer copies. A parameter set that stores one rule $L$ times is
strictly dominated by one that stores it once and iterates. This one follows from the
requirements rather than from a measurement — and §22's sharing evidence says real transformers
are dupicated-rule systems.

### 23.2 What is forced, what is free

| forced by the requirements | left free |
|---|---|
| content addressing (R4) | width $d$ |
| rotation-like spectra (R3, R5) | the state's encoding: vector, stack, set, graph |
| bounded Jacobian product (R5) | rank $r$ of the update |
| one rule, iterated (R6) | store size; in weights or outside |
| | iterations $T$ — compute, not bits |
| | whether it is a state at all, or the transcript |
| | tokeniser, and what the rule is trained on |

Density is a claim about the right-hand column only. The forced column is where a design can be
*wrong*, but not where bits are won.

### 23.3 The bit-minimal member

| quantity | value |
|---|---|
| state, $d = 1024$, fp16 | 16384 bits |
| read/write maps, $2dr$, $r = 64$ | 131 072 params |
| gates + addressing map | 196 736 params |
| **rule total at 3-bit** | **327 808 params = 9.834e+05 bits** |
| rule against the 3B at the same packing | 9801× |
| store, per association | 19 bits (id 17 + slot 2) |
| FFN bits expressed as associations | 2.97e+08 |

The rule is of order $10^6$ bits. Everything a 3B spends above that is knowledge, duplication,
or both — and the two are separable: §22 shows the duplication (attention sharing is nearly
free, FFN sharing costs a little) while the stores literature shows the knowledge
(RETRO's 10× parametric gain from a retrieval store, at 7 bytes per token of data).

### 23.4 What the accounting predicts

- **The rule is cheap and the knowledge is not.** If the requirements are right about how much
  rule a language needs, a matched-CE design should fit its rule under $10^6$ bits. A trained
  design needing far more would refute this section's premise.
- **The store dominates total bits above $2.97\times10^8$ associations.** Below that the weights
  are the denser place for knowledge; above it, an index is. Real LLMs sit near the crossover,
  which is why retrieval is worth ~10× at 7B and less at small scale.
- **The trap, stated in advance:** R2 is a *capability* requirement, not a CE requirement, and
  the two can diverge — the same way perplexity hid recall failure in §19.9. If TC⁰ models reach
  CE parity, R2 binds only on the tasks that need state tracking, and a CE-only evaluation will
  declare the wrong requirements load-bearing.

### 23.5 The falsifiers

1. **R2/R3.** If a TC⁰, non-rotational cell (positive-eigenvalue diagonal recurrence) reaches
   matched CE *and* matched state-tracking on permutation-structured tasks, the requirement is
   mis-stated for language at scale. The literature says it cannot do the latter, so this is a
   cheap check on any new design.
2. **R6.** If a 3B's layer groups do not admit a shared basis at small reconstruction error, the
   duplication is not there and the density thesis loses its main term (§22.5's read-only gate).
3. **R4.** If a model without content addressing matches CE, addressing is not required for
   prediction — only for memory, and §19.9c already prices that role separately.

The first thing worth doing is not a training run: it is the read-only check of R6 on the
checkpoint on disk, because it decides whether the density claim has a large term at all.
*[rev. 2026-11-17h]*

---

## 24. What the state's encoding costs

§23.2 left the state's type free. It can be priced. Fix a cell whose state must carry $S$ bits
(worst case over inputs), coordinates at $p_s$ bits, weights at $p_w$ bits, and compare one
operator — the minimum any recurrent cell needs.

**Dense vector.** Holding $S$ bits needs width $d \ge S/p_s$, and the operator acting on that
width is $d \times d$:
$$B_{\text{dense}} = p_w (S/p_s)^2 .$$
**Typed state.** An explicit encoding — a permutation, a stack, a graph — costs the state itself
plus a pointer per coordinate: $B_{\text{typed}} = S + p_w S/p_s$.

The ratio is $p_w S/p_s^2$, so the dense encoding is cheaper only below

$$S^\* = p_s^2/p_w = 85 \text{ bits}$$

at fp16 coordinates and 3-bit weights. Every interesting state is far above that line, and the
penalty grows linearly in the amount of state:

| S (bits) | width d | dense rule bits | typed bits | ratio |
|---|---|---|---|---|
| 1024 | 64 | 12288 | 1216 | 10 |
| 10240 | 640 | 1228800 | 12160 | 101 |
| 30332 | 1896 | 10784448 | 36020 | 299 |
| 49152 | 3072 | 28311552 | 58368 | 485 |

**The permutation case, exactly.** Tracking $n$ entities is tracking a permutation, whose state
is $\log_2 n! \approx n\log_2 n - 1.44n$ bits: 30 332 bits for $n = 3\,000$. A nonlinear vector
could hold that in 1 896 coordinates. A *linear* cell cannot: faithful representations of $S_n$
have dimension $\ge n-1$, so it needs $d \ge 2999$ — which is, to within rounding, the 3 072
width of the model on disk. That operator costs 26 982 003 bits to carry 30 332 bits of group
state, against 39 332 bits written out: **686×**.

So a 3B's width is what a *linear representation of three thousand entities* demands, while the
information actually held is 30 kbit. The overhead is structural, not wasteful tuning: a dense
operator spans all $d^2$ linear maps in order to implement a structured action whose own
dimension is $n \log n$.

**Two assumptions that keep this honest.** First, *exactness*: the $\Omega(n)$ width bound is
for exact composition. If only statistical recall is required, a dense vector's associative
capacity ($\sim d^2 / \log d$ items) is a far better exchange rate, and the typed advantage
shrinks. Second, *structure*: $d^2$ buys arbitrary linear maps while a typed state buys
structured ones — the argument holds only where the needed state update is a group action, a
push/pop or an edge insert, which is what R2 and R3 claim about language. A capability whose
state update is genuinely unstructured would falsify the lever, and that is the test to look
for.

### 24.1 What survives from that

One fact is load-bearing and the rest is not: exact tracking of $n$ entities forces $d \ge n-1$
(faithfulness of the representation), so a model that tracks inside a dense residual stream pays
$n^2$ per operator for $n$ tracked entities — 2 998× at $n=3\,000$. That is the constraint the
architecture in §25 is built to respect.

Capacity comparisons between memory schemes are out of scope. The object here is an architecture
that replaces a transformer, not a memory that augments one, and a capacity law about recall says
nothing about whether a cell can compute. The only lesson taken is cheap and negative: dense
superposition is efficient *given a nonlinear readout*, so §25 keeps a dense path where structure
is unavailable instead of forbidding density.

---

## 25. The architecture these constraints leave

Four roles, and what each costs a transformer in bits (§22):

| role | transformer mechanism | bits |
|---|---|---|
| know | FFN, read as a key-value memory | 5.637e9 |
| route | QKᵀ, d² per layer | 2.819e9 |
| embed | token and position embeddings | 1.182e9 |
| track | residual stream | 0 — it is d |

The route row is the opportunity. 2.819e9 bits are spent computing an *address*, and a hash
computes an address for zero. A transformer pays $d^2$ because its address is a learned bilinear
form; if the address is a table index instead, the parameters left over are the knowledge itself.

### 25.1 The cell

State is typed: a permutation $\pi$ over $K=64$ slots, a stack of depth $\le 256$, and a gist
$g \in \mathbb{R}^{64}$ for whatever is not structured. For token $x_t$, iterate $i = 1..T$
with $T=8$:

$$u_i = H(x_t, g_{i-1}, i)$$ $$r_i = C[u_i]$$ $$m_i \sim \mathrm{Gumbel}\{\text{hold, push, pop,
swap, bind}\}$$ $$g_i = \tanh(W_g\,[r_i ; g_{i-1} ; \mathbf{1}_{m_i}])$$ $$C[u_i] \leftarrow
C[u_i] + \alpha\,(r_i - C[u_i])$$ $$\mathrm{logits} = g_T W_o$$

$H$ is a fixed hash with no parameters; $r_i \in \mathbb{R}^{32}$; $W_g \in \mathbb{R}^{96\times64}$;
$W_o \in \mathbb{R}^{32\times 128256}$. The write is a delta rule, so the codebook adapts inside a
sequence without backpropagation through time. Nothing in the cell is $d^2$: the only matrices are
96×64 and the token table.

**Why iterate instead of stack.** R6 said one rule, iterated. Depth becomes compute, so parameters
do not multiply by depth — the thing that makes a 28-layer transformer 28 copies of a 9.4M-weight
object.

**Why the state is typed.** §24's bound: an exact permutation over 64 slots costs $64\log 64 = 384$
bits of state and $O(K)$ parameters, while a dense stream faithful to $S_{64}$ needs $d \ge 63$ and
$63^2$ weights *per operator*. The cell can afford to be exact because it does not have to be wide.

### 25.2 Why the table is the dense place for knowledge

A codebook of $M$ buckets holding $n$ keys has $n(n-1)/2M$ colliding pairs. At $M = 2^{24}$:
0.6% of keys collide at $n=10^5$, 5.8% at $10^6$, 44.9% at $10^7$. A fixed hash therefore cannot
hold a language's contexts, and that is the point of the controller: it emits the address from
(token, gist, step), so buckets are allocated by frequency and collisions pile up in the tail. The
tail is then what the typed state and the 6 720-parameter map handle without the table.

The density argument is a counting one. A fact is 32 dims at 3 bits = 96 bits, and 1.6106e9 bits
of codebook hold 1.68e7 facts — the same order as the 6.8×-deduplicated transformer's 1.4e9 bits.
The transformer does not hold more knowledge. It holds that knowledge plus the routing that finds
it.

### 25.3 The ledger

| component | weights | bits |
|---|---|---|
| codebook C | 536870912 | 1.6106e9 |
| token map W_o | 4104192 | 1.2313e7 |
| controller, action head, write gate | 6720 | 2.0160e4 |
| **total** | | **1.6229e9** |

5.94× fewer bits than the same capability at 3 bits, 9.95× fewer than the deployed Q4_K_M, and
54016 multiply-accumulates per token against 6.43e9 — 118955× less work. The compute figure is the
one that matters: this is not a compressed transformer, it is a different order of work per token.

### 25.4 Why it can be trained, and where that claim is weak

The write is one gradient step on $\lVert r - C[u]\rVert^2$, so the codebook learns locally. The
actions are discrete and pass gradient straight through to a 5-way head. $W_g$ has 6 720
parameters, so the optimisation that must succeed is 6 720-dimensional — not $3\times10^9$.

Weak: discrete heads collapse onto `hold`, and a codebook can dead-end in the first epoch. Entropy
bonuses and a uniform initial write are the usual mitigations; neither is verified here, and this
is the part of the design I would not defend as settled.

### 25.5 The falsifier

Matched bits, matched tokens, the §19 ladder. Carrier to beat: the n-gram index at
4.6377/4.8061/4.7580 nats (32/64/128); oracle bound 1.2304/1.0799/1.1026. Proposed gate, as a
dated amendment to `PREREGISTRATION.md`: **CE ≤ 2.5 nats at 64-token contexts within 1.623e9 bits**,
else the architecture is falsified as a replacement.

The diagnostic that decides it before the CE does is the **novel-address rate** — the fraction of
eval tokens whose (token, gist) address was unseen in training. §19 already showed where carriers
die: 4.6377 nats in the tail against 0.3224 for the head. Above roughly 20% novel addresses the
table cannot carry the sequence, and the composition falls to 6 720 parameters, which cannot do it.
That is the failure mode I expect, and it is measurable in one pass over the eval set with no
training at all.

### 25.6 What this is not

Not a memory attached to an LM. There is no attention, no FFN, no transformer anywhere in the cell:
the parameters are a table plus 6 720 weights, and the computation is eight passes of a 96×64 map
over a typed state. The honest risk, once: 6 720 parameters of composition is very little, and
language may need more than a table and a small controller. That is a claim about the long tail,
and §25.5 tests it for the price of one pass. *[rev. 2026-11-17i]*

---

## 26. Where the bits actually are: WikiText-2, measured

Before any architecture: a fixed corpus, and the conditional entropy of the next token split
into the four buckets that determine what a model must contain. WikiText-2 word level
(2 051 910 train / 213 886 valid / 241 211 test tokens, 33 277 types, 707 documents). No
training: counts and two online-adapted scalar mixing weights. Locality is interpolated
Kneser-Ney at order 3 ($D = 0.75$), over continuation bigram and unigram counts; the
referential stages add a document cache and an in-document match model (`harness/f12_entropy.py`).

| stage | bits/token | gain |
|---|---|---|
| unigram | 9.8795 | — |
| KN trigram | 7.9916 | 1.8878 |
| + document cache | 7.9318 | 0.0598 |
| + match model | 7.6458 | 0.2860 |

**Locality is 19.1% of the entropy.** And it is a short-range effect at this data scale:
with Jelinek-Mercer weights tuned on the validation split, a 6-gram model's weights for orders
3–6 all came back zero. The reason is sparsity — 2 M training tokens over 33 k types — not a
property of language, so this is a floor on locality for a table of this size, not a ceiling.

| class | tokens | share | KN-3 | after cache+match |
|---|---|---|---|---|
| novel (first use in document) | 104273 | 43.2% | 10.6525 | 10.5731 |
| repeat (already used) | 136938 | 56.8% | 5.9655 | 5.4167 |

**The referential gap is measured and it is large.** Knowing that a token is a repeat is worth
4.69 bits (10.65 against 5.97), yet the two referential mechanisms recover 0.35 of it — 7%. The
cache knows the word has occurred, the match model knows what followed the last identical
context, and neither can say *which* earlier word is now due. That is a slot-filling problem,
not a smoothing problem, and it is the strongest measured case for a content-addressed store:
several bits per token sit behind identifying the referent.

**43% of test tokens are novel** and cost 10.65 bits each. No table helps them; they are
predicted from world knowledge and composition, which is what weights are for. So the residual
7.65 bits is not one bucket but two, and the larger half is the novel one.

**What this changes.** Locality is not where the bits are: my earlier assumption that a
table-first design would capture the bulk of the entropy is wrong on this corpus. A table-only
model sits at 7.65 bits/token (perplexity ≈ 200) against a unigram 9.88. The measured ordering
for where design effort belongs is (1) identifying referents, (2) predicting novel tokens,
(3) local smoothing last. One caveat that cuts the other way: at word level, repeats are
under-represented relative to subword tokenisation, so on a BPE vocabulary the referential
share would be larger, not smaller. *[rev. 2026-11-17i]*
