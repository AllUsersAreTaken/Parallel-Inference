# Pre-registration - softmax-endpoint landscape readouts

Frozen **2026-11-17, before any readout data was collected**. Stages 0-5 only.
Stage 6 (small language model) is out of scope until stages 2-4 pass; it needs a
trained hybrid backbone and is not attempted here.

The question, in one line: at matched output width, do readouts that expose the
retrieval landscape (log-partition, covariance, rank-2 Hessian, descent path)
beat the plain softmax endpoint `PV` under ambiguity, or is any gain explained by
width / multi-temperature diversity?

Implementation: `harness/readout_landscape.py`. Analysis is CPU (the only
installed PyTorch is a `+cpu` build); the harness selects CUDA when present.

## Frozen setting

- `d` in {64, 128}, `d_v = d`. `N = round(alpha * d)`, `alpha` in {0.5, 2, 5, 20}.
- Pattern correlation `rho` in {0, 0.3, 0.6, 0.9}: `k_n = normalize(sqrt(rho) u +
  sqrt(1-rho) e_n)`, `u, e_n ~ N(0, I_d)`. Realized mean off-diagonal correlation
  is logged per cell.
- Cue: `q = normalize(k_target + sigma * normalize(noise))`, `sigma` in
  {0, 0.3, 0.6, 1.0, 1.5}. Values `V` are independent random unit vectors.
- Temperature: main `tau = 1.0 * sqrt(d)` (so logits have O(1) spread). The
  multi-temperature control pair is `tau * {0.5, 2.0}`. A `tau` grid
  `{0.25, 0.5, 1, 2} * sqrt(d)` and a learned-`tau` variant are reported as a
  sensitivity analysis, not tuned per cell.
- Path: `beta_t = tau * (1/32)^((T-t)/T)`, `T` in {3, 5, 8}.

## Frozen readouts

| ID | Vector |
|---|---|
| R0 | `PV` |
| R0w | width-only control: `R0` padded, plus `cos(R0 @ G)` random Fourier features (fixed `G`, seed 17) to fill the common width |
| R1 | `[P_1V, P_2V]` at `tau*0.5`, `tau*2.0` |
| R2 | `[P_1V, P_2V - P_1V]`, same two temperatures (= linear reparametrisation of R1) |
| R3 | `[PV, logZ, H(p), max p]` |
| R4 | `R3 + [trace Cov_p(k), diag Cov_p(k)]`, diagonal by `E_p[k^2] - mu_k^2` |
| R5 | `R3 + [top-2 singular values, left vectors, right vectors]` of `Cov_p(v,k)`, via 8 power iterations with deflation |
| R6t3 / R6t5 / R6t8 | `[x_0..x_T, E_0..E_T]`, `x_t = P(beta_t)V`, `E_t = -logZ(beta_t)` |
| R7 | `[PV, (1/N) sum_n sigmoid(s_n) v_n]` |

Common width `W` = the largest nonzero readout dimension among active readouts
(d=64: 585 with T=8; d=128: 1161). Every readout is zero-padded on the right to
`W`. The learned head is identical for all readouts: `W -> 64 -> d_v`, GELU,
Adam, lr 3e-3, 100 steps, 4 memory banks x 12 cues per step, no weight decay,
cosine logits scaled by 10, cross-entropy over 96 candidates (target + the 8
highest-scoring wrong-value memories + random fill; all `N` candidates at eval).
The control R0w is a diagnostic, not one of R0-R7.

Data split: 16 training memory banks x 12 cues and 10 held-out banks x 12 cues
per cell, disjoint bank seeds. Five seeds per cell (1000-1004) drive banks and
head init. Metrics: top-1 value accuracy and recall@5 among all `N` candidates.

Statistics: per-cell mean over 5 seeds; contrasts are cluster-bootstrapped over
cells (10k resamples) because seeds within a cell are not independent evidence.

## Frozen gates

**G1 - sanity and equivalence.** PASS requires all of:
- (a) every closed form matches autograd in float64: `dlogZ/dtau = E_p[r]`,
  `d2logZ/dtau2 = Var_p[r]` (heat capacity), `dPV/ds_i = p_i(v_i - mu_v)` so
  `Cov_p(v,k) = sum_i (dPV/ds_i) (k_i - mu_k)^T`, and `d2logZ/dq2 = tau^2 Cov_p(k)`;
  max relative error < 1e-8.
- (b) power-iteration top-2 vs `torch.linalg.svd`: singular values within 1e-4.
- (c) R2 vs R1 head accuracy: `|Delta| <= 1.0 pp` over the reference cells
  (`d=64`, `rho` in {0, 0.6}, `alpha=5`, `sigma` in {0, 0.6}, 5 seeds = 20 cells)
  and the 95% cluster-bootstrap CI of the paired difference contains 0.
  Failure means the width-matched setup is broken, not that the hypothesis failed.

**G2 - retrieval fidelity.** Hard region `H = {rho >= 0.6, sigma >= 0.6}`.
For each landscape readout `k` in {R3, R4, R5, R6t3, R6t5, R6t8}, `Delta_k` =
mean over `H` cells of `acc_k - acc_R0`. PASS if some `k` has `Delta_k >= +3.0 pp`
with bootstrap CI lower bound > 0, **and** the same `k` beats R0w by >= 2.0 pp
(otherwise width explains it), **and** R1's own `Delta_H < +3.0 pp` (otherwise
multi-temperature diversity explains it).

**G3 - ambiguity task.** Cue equidistant from two correlated memories, then a
geometric hint. PASS if some landscape readout beats R0 by >= +5.0 pp in
after-hint accuracy with CI excluding 0, and its ECE is not worse than R0's by
more than 0.02.

**G4 - information probes.** Linear probe (ridge, lambda 1e-2 on standardised
padded features, closed form) on each raw readout predicts the runner-up memory
identity (regression to the runner-up value, scored by nearest-value match).
PASS if some landscape readout's probe accuracy beats R0's by >= +5.0 pp with CI
excluding 0. Binary ambiguity and noise-level probes are reported as secondary.

**G5 - path analysis.** On `rho=0.9`, `alpha=5`, `sigma=0.5` instances, the
fraction of heat-capacity local maxima within 1/16 of the beta span of a
top-1 leader change must be >= 1.5x the uniform-placement null (95th percentile
of 2000 shuffles).

**Stage 6** runs only if G2, G3 and G4 all pass.

## Commands (frozen)

```
python harness/readout_landscape.py sanity
python harness/readout_landscape.py sweep      # stage 2 + the R1/R2 reference cells
python harness/readout_landscape.py tau        # tau grid + learned tau
python harness/readout_landscape.py ambig      # stage 3
python harness/readout_landscape.py probe      # stage 4
python harness/readout_landscape.py path       # stage 5
python harness/readout_landscape.py report
```

Outputs go to `results/readout_landscape/`. If G2 fails, G3-G5 are still
executed but labelled **diagnostic only** - the registered protocol says stage 3
needs stage 2 to pass.

## Amendments

**AMENDMENT 1 - 2026-11-17, before any data collection.** G1(b) as written ("power
iteration vs `torch.linalg.svd`, singular values within 1e-4") is not achievable
and was mis-specified. Measurement: on random matrices and on a realistic
`Cov_p(v,k)`, the *first* triplet is exact (relative error ~1e-16), but the
deflated second one is only defined up to the gap between sigma_2 and sigma_3.
In the realistic covariances here sigma_2/sigma_3 is close to 1, so 8 iterations
leave a 2e-3 relative error on sigma_2 and no iteration count fixes it -- any
top-2 of a near-tie is arbitrary inside the tie.

Resolved as follows, still before data:
- G1(b) is checked on a rank-2 matrix with well-separated values (sigma_1 = 5,
sigma_2 = 3) built from random orthonormal factors, at the same 8 iterations R5
uses. Bar: singular values within 1e-4, the second triplet's residual within
1e-4, and the 2-D right subspace's projector within 1e-4.
- The degenerate cases (random matrices; the actual `Cov_p(v,k)`) are reported as
diagnostics, not gates: sigma_1 error, sigma_2 error, and the sigma_2/sigma_3 gap.
- R5 keeps 8 power iterations as frozen above. (see Amendment 2 for the
  temperature and sample-budget fixes, also pre-data.)

**AMENDMENT 2 - 2026-11-17, before any readout contrast was recorded.** Three
setting bugs found while building and smoke-testing the harness. No R0-vs-landscape
accuracy number existed when these were fixed; the only runs were 1-2 cell
construction checks, and their numbers are not part of the results.

1. **Key normalisation.** The mixing `k = sqrt(rho) u + sqrt(1-rho) e` used
   unnormalised noise rows, so realised correlation was 0.12 instead of 0.9.
   Fixed: both parts are unit vectors. Realised rho is logged per cell and now
   reads 0.90.
2. **Temperature.** `tau = sqrt(d)` is far too cold in the correlated, high-N
   corner: at rho=0.9, N=5d the target's score margin is ~0.09, so p is nearly
   uniform (measured: cue-dependent std of `PV` is 1.4e-3 against 0.2 for `logZ`)
   and every readout sits at the floor. tau is now chosen **per cell on training
   cues only**, from the frozen grid `{0.25,0.5,1,2,4,8} x sqrt(d)`, minimising
   the target's mean negative log-likelihood under `p` -- a proper scoring rule
   with a finite optimum, so it cannot be pushed to the degenerate tau->infinity
   that a hard margin criterion would pick. Chosen `tau/sqrt(d)` is logged.
   R1/R2 use `0.5x` and `2x` the chosen tau.
3. **Sample budget and standardisation.** The pilot budget (192 cues) left the
   wide readouts under-fitted rather than uninformative (R5 at 0.07 while the
   same endpoint block inside R3 scored 0.62; 3000 steps did not help, 3072 cues
   did). Budget is now 32 banks x 64 cues = 2048 cues for d=64, 600 steps;
   d=128 gets 16 x 16 = 256 cues at 300 steps because R5's covariance costs
   ~4x per sample. Readouts are standardised per dimension on train banks, with
   a 1e-3-relative std floor so a genuinely constant dimension cannot be
   amplified into unit-scale rounding noise.
4. **d=128 replication grid** narrowed to `alpha` in {0.5, 5}, `rho` in {0, 0.9},
   `sigma` in {0, 0.6} so the whole sweep fits in one overnight-free session;
   the hard region (rho>=0.6, sigma>=0.6) is still represented.

5. **R6 and G5 event definition.** R6 was first built as a temperature ladder on a
   fixed cue, `x_t = P(beta_t)V`. That is degenerate as a *path*: `argmax
   softmax(beta r) = argmax r` for every beta > 0, so no leader can ever change and
   "paths split between competing memories" is unsatisfiable by construction
   (checked: 0 splits, 0 heat-capacity maxima in 48 cues). R6 is now the
   modern-Hopfield retrieval descent, `s_0 = cue`, `s_{t+1} = K^T softmax(beta_t K
   s_t)`, `x_t = p_t^T V`, with beta annealed from tau/64 to tau; the state moves,
   so splits exist (measured 1.0 per trajectory). By the same measurement, the heat
   capacity rises monotonically along this descent, so "local maximum of C" is an
   empty event set too. G5's event is therefore **the step where C changes fastest**
   (steepest reorganisation), and the gate asks whether that step lands within 1/16
   of the trajectory on a leader change, against a uniform-placement null.

6. **R6 attractor and beta start (supersedes item 5's schedule).** Two further
   measurements on the descent map, both before any R6 number was recorded:
   - The uncentred map has an attractor at the shared component: iterate
     std collapses to 1e-5 after one step and cosine to the true value goes
     0.32 -> 0.06 -> -0.03, identical for every cue. Keys are therefore centred
     (`K - mean(K)`) in the descent, which is the standard treatment for
     correlated patterns.
   - The anneal start matters more than the number of steps. Starting at
     `tau/32` leaves p diffuse, so the update `s <- p^T K_c` lands near 0 and the
     path dies (same 1e-5 collapse). `tau/2` and `tau/4` converge to the target in
     one step (cosine 1.000 at every iterate) -- no path at all. `tau/8` is the
     only measured setting that keeps a multi-step trajectory (cosine
     0.997 -> 0.95 -> 0.82 -> 0.70). BETA_LO is therefore 1/8.
   - Recorded as an expected outcome, not a bug: at rho=0.9, sigma=0.6 this map
     settles on a spurious fixed point (cosine to the true value 0.05) for every
     starting cue. If R6 fails, that mechanism is the reason, and it is a
     property of the descent as specified, not of the harness.

Gates G1-G4 are unchanged; G5 keeps its 1.5x bar with the event redefined in item 5. Its features are therefore exact
when the landscape is well separated and a deterministic (but arbitrary inside
the tie) basis when it is not; that is a property of what R5 reports, and is
recorded when the results are written up.

7. **Protocol repair after the first full run (POST-DATA; 2026-11-17 late).** The
   first complete sweep/tau/ambig/probe run produced numbers, and inspecting them
   showed the comparison itself was invalid in two independent ways. This item
   changes the *protocol*, not the gates: G1-G5 keep the thresholds frozen on
   2026-11-17, and no threshold is moved to fit the new data. Every number from the
   superseded run is marked as such in the results file and none of it is carried
   forward unlabelled.

   a. **Temperature was pinned at the grid edge.** Amendment 2's rule (minimum
      target NLL) is monotone in tau: softmax's argmax does not move with tau, so
      the likelihood of the true index never turns around and the rule saturated at
      its own maximum, `c=8`, in 349 of 360 cells. The sweep was therefore an
      under-tempered comparison in exactly the correlated corner H1 is about. New
      rule: per cell, on **train-only** banks (seeds 900+/950+, disjoint from both
      the train and test banks), train a small R0 head at each
      `c in {0.5,2,8,32,128} x sqrt(d)` and take the **smallest** c whose accuracy
      is within 0.5 pp of the best -- the left edge of the accuracy plateau. The
      whole accuracy curve is logged per cell, so any saturation is visible in the
      results file rather than hidden. The plateau rule gives the endpoint its own
      best temperature without driving it to a degenerate hard-argmax readout, where
      the landscape features it is being compared against would vanish by
      construction. The `tau` mode additionally reports R0/R3/R5 along a 9-point
      curve from `0.5` to `128 x sqrt(d)` with 5-seed CIs, so no conclusion depends
      on the selector's choice.

   b. **Width was a first-order confound.** Amendment 3 padded every readout with
      zeros to the widest readout's raw dim (585 at d=64). That is input dilution,
      not information: measured directly, at 3072 cues and rho=0.9, sigma=0.6, R0w
      (raw `[PV, cos(PV,G)]`) scored 0.478 against R0's 0.699, and in the first full
      sweep R0w averaged 28.9 against R0's 91.4. New rule: each readout is
      standardised per dimension on train banks (1e-3 relative std floor, as in
      Amendment 3) and then mapped to a **common 128-dim head** by a fixed random
      matrix: orthonormal columns when the readout is wider than 128
      (Johnson-Lindenstrauss compression, norms and inner products preserved in
      expectation) and an isometric embedding when it is narrower, scaled by
      `sqrt(128/k)` so every readout enters at unit per-dimension variance. The map
      is seeded from the readout's name, never fitted, so it cannot move
      information between arms. This supersedes "padded or projected" in the frozen
      readout specification, and it makes `width-matched` mean 128 dimensions of
      head input for every arm, including R0w and the 585-dim path readouts.

   c. **Head capacity and budget.** The shared head is widened from 64 to 128
      hidden units (two layers, unchanged shape otherwise), because equal *input*
      width is worthless if the head cannot fit the widest arm. Training budget is
      equal for every readout by construction (same steps, same batches, same
      optimiser, one loss over all arms). Steps are 450 at d=64 (down from 600:
      the wider head converges sooner) and 225 at d=128, with d=128 keeping 16 x 16
      = 256 cues; d=64 keeps 32 x 64 = 2048 cues. Every arm in a cell sees exactly
      the same cues, banks and steps.

   d. **Probe power.** Stage 4 probed a 320-way runner-up identity with 500 training
      cues, which cannot be learned by any readout, so a null there said nothing
      about the readout. Probes now use `N = 2d` (128 at d=64), `n_tr=2000`,
      `n_te=500`, and the chance rate `1/N` is printed beside every accuracy.

   e. **What this does not fix.** The ambiguity task's 2-way choice is a hard
      credit-assignment problem for a shared-trunk head; if every arm stays at or
      below chance on it, that is a limitation of the task's learnability in this
      harness and G3 is then uninformative rather than evidence for H0. This is
      stated in the results file rather than repaired by tuning the task.

8. **Stage 3 retired and re-specified; G1c cause decomposed; probe widened (POST-DATA;
   2026-11-17, after Amendment 7's run).** Written after seeing G3 fail and G1c miss,
   so every number quoted here is a *reason for a protocol change*, not evidence for
   H1. No frozen threshold is moved: G1c keeps its 1 pp bar and its CI clause, and the
   new task below reuses frozen G3's thresholds unchanged.

   a. **Why the frozen ambiguity task could not test the hypothesis (measured).** The
      cue is the midpoint of a pair, and the disambiguating hint is a 2-dim one-hot
      saying "member a" or "member b". The head emits a value-space vector that is
      scored against value candidates, and the pair's two members are exchangeable in
      everything the head sees, so no function of (readout, hint) can recover which
      member is which. Three arms on identical inputs, same head, same budget
      (`ambig_check`): real labels, labels permuted across pairs (a true null), and an
      oracle that additionally receives the value-space direction `V[tgt]-V[oth]` --
      the one thing the design never supplies. Measured on 10 cells (rho in {0.6,0.9},
      sigma in {0.3,0.6}, 5 seeds): R0 real ~25%, permuted ~0.2% (chance over 96
      candidates, i.e. the head learns nothing when the labels carry no signal), oracle
      ~38% (pair 46%), the choice rate given the pair found going from 0.74 to 0.82,
      i.e. oracle-real +22.3pp [+18.2,+26.7]. So the frozen task is not strictly
      information-free -- its labels are learnable at 15.4% against 0.2% for the
      permuted null -- but its absolute level is dominated by how often the pair is
      even found (20.7%), which is why a +5 pp gate there had almost no dynamic range.
      The repair below is justified by that, not by a claim of unlearnability.

   b. **The repaired task.** The hint is now a second cue in the same space,
      `q1 = unit(q0 + delta*(k_tgt - k_oth))`, and every readout is computed for both
      cues and concatenated, so the head receives the same object a later retrieval
      step would. Nothing else changes: same banks, pairs, candidates, head, budget,
      target definition. `delta` was set from a training-free ceiling scan run before
      the fitted comparison (fraction of pairs where `argmax(q1)` is already correct:
      100% at delta=0.3 for three of the four cells and 72% at rho=0.9, sigma=0.6;
      52-100% at delta=0.15). Two values, 0.15 and 0.3, are reported, so the gate is
      not read at a single difficulty.

   c. **G3b (new, pre-registered here, thresholds copied from frozen G3).** On the
      repaired task, a landscape readout passes if it beats R0 by >=+5 pp with a 95%
      CI above 0 and does not raise ECE by more than 0.02. Separately, the task counts
      as *learnable* only if R0's 95% CI lower bound exceeds 5x chance. A run where the
      task is learnable but no arm gains 5 pp is a real negative for H1's ambiguity
      branch; a run where R0 sits at the ceiling is reported as such, because a +5 pp
      gain is arithmetically impossible there.

   d. **G1c cause decomposition (`g1c2`).** `R1 = [PV1, PV2]` and
      `R2 = [PV1, PV2-PV1]` share the same temperatures, seeds, data and budget by
      construction (one `readout_vectors` call, one head stack, one loss), so the
      residual cannot come from those. Two further fits on the same cells separate the
      remaining candidates: the same MLP with 3x steps (convergence) and a **linear**
      head with 1200 steps (basis dependence). A linear head cannot distinguish the two
      arms at all -- `span{PV1, PV2} = span{PV1, PV2-PV1}`, so the reachable function
      class is identical and a converged linear fit must give equal accuracy. Whatever
      the linear arm shows is the floor; the remainder in the MLP arms is the effect of
      a GELU nonlinearity being applied in different coordinates.

   e. **G4 power.** The runner-up probe was widened from 5 to 10 banks per seed (25 ->
      50 banks) because its +3.67 pp [+2.51, +4.90] miss against a +5 pp bar is a
      precision question. The bar is unchanged; if the wider interval still excludes
      5 pp the miss is reported as a miss.

## Amendment 9 — 2026-11-17 — the axis claim: non-inferiority plus incremental information

Written before any L-stage result existed.  Amendments 1-8 and every frozen gate
G1-G5 are untouched: they remain the record of the retrieval question, and the H1
verdict in the report stands unchanged.  This amendment does not reinterpret them.
It registers a *different* claim, because retrieval accuracy was the wrong test of it.

Retrieval accuracy asks whether landscape information helps find the basin.  The axis
claim is that the landscape exposes a structure axis (ambiguity, noise, superposition,
crowding) that the endpoint cannot represent, and that some computation needs it.  A
readout can be informative about that axis and useless for basin-finding; the two
questions have different labels, so they need different tests.

L1 (retrieval guard, non-inferiority).  The endpoint keeps its full width (R0 -> 128
in every block set) and the landscape enters only as an 8-dim side channel: logZ,
entropy, maxp, top-2 margin, heat capacity, total key variance, sigma1 of Cov(v,k),
Frobenius norm of Cov(v,k).  Arms E, E+S, E+N share one head stack, one width and one
budget.  N1: E+S - E >= -1.0 pp top-1 with 95% CI lower bound > -1.0, over the hard
corner, the easy corner and near capacity (alpha in {2,5,20} at rho=0.9).

L2 (conditional information).  Per task, five equal-width block sets: [E], [E|S],
[E|N], [S], [S res E].  Metric: AUROC for binary/multi-label, R^2 for regression.
A1: [E|S] - [E|N] >= +0.03 with CI > 0 on at least one task, over 50 banks
(5 seeds x 10), paired within bank.

Capacity-matched null.  N is S with cue rows permuted.  The plan first called for a
rotated copy of S, but a rotation is invertible and therefore carries the same
information, so it would not have been a null at all.  The correction makes every
test strictly harder and was made before any L2 data existed.

A2 orthogonality: the increment from [E | S res E] must be >= 60% of the naive
increment from [E | S], i.e. the gain is not the endpoint in a different encoding.

L3 tasks, whose labels depend on the axis by construction (m = 2, rho = 0.9,
sigma = 0.5, N = 2d, 2000 train / 500 test cues per bank):
  a superposition  which memories form the mixture (multi-label AUROC, chance 2/N)
                   and the mixing weights given the support (R^2).
                   A3: gain >= 0.03 AUROC on support recovery.
  b selective      is the endpoint's top-1 correct?  AUROC and AURC.
                   A4: [E|S] - AUROC(logZ) >= +0.02 with CI > 0.  maxp and logZ
                   (energy) are prior art and both already live in R3's scalars, so
                   beating the endpoint is not the bar; beating energy is.
  c novelty        corrupted stored memory vs random cue.  A5: over energy >= +0.02.
  d crowding       mean top-10 key cosine at the landing memory.  A6: R^2 >= 0.10.

Candidates: S3 (scalars), S4 (key variance), S5 (rank-2 Hessian/covariance), S6t5
(descent path), RUN (V of the runner-up).  RUN is an identity readout, not a
landscape one, and is excluded from any task whose label is the runner-up identity.

Adaptive tau control is registered but not run here: it is the one task that feeds
back into retrieval, and L1 already covers the retrieval question.

## Amendment 10 — 2026-11-17 — decision roles, not probe accuracy

Written before any R-stage result existed.  Amendments 1-9 and every frozen gate are
untouched.  §10/§11 asked whether landscape quantities *predict* a label the endpoint is
also trying to predict, and the answer was "barely, and mostly the cheap scalars".  That
test cannot detect a quantity that fills a role the endpoint has no scale for, because
the probe is free to imitate the endpoint.  This amendment registers three decisions and
scores the decision directly, so no probe stands between the quantity and the number.
Where the proposal itself contains a fitted gate (routing), the gate is trained on train
cues and scored on held-out ones like any other arm.

Common setting: d=64, rho=0.9, N and tau as in earlier stages (tau comes from the same
train-only selector; the tau question is not re-opened), 1500 train / 500 test cues per
bank, 5 seeds x 8 banks per cell, two noise regimes sigma in {0.5, 1.0} (endpoint top-1
98.4% and 43.7%).

**R1 routing.**  B=8 banks are a *partition of one memory cloud*, so every bank has the
same shared component and the only evidence for a bank is the specific memory it holds.
(Banks drawn from their own clouds are identifiable from the cue's shared component
alone, cos ~ sqrt(rho) = 0.95, which every router solves.)  Exactly one bank holds the
cue's source memory.  Under a uniform prior over memories the evidence for bank b is
Z_b / N_b, i.e. logZ - log N_b; that is the size correction and it is why bank sizes are
run both equal (32 each) and unequal ({16,64,32,16,64,32,16,64}).

Arms: `logZ` (argmax energy), `logZsz` (size-corrected), `maxp` (per-bank confidence),
`gate_S` (fitted gate on logZ, maxp, H, margin), `gate_H` (fitted gate on the Hessian
invariants alone), `gate_L` (fitted gate on all landscape scalars), `gate_PV` (fitted
gate on the endpoint vectors PV_b, the endpoint's only fair shot at a cross-bank
quantity).  Scored on: route accuracy on test cues, and end-to-end accuracy (route, then
retrieve inside the chosen bank) against the flat single-set baseline at B times the
reads.  R1 is informative if some landscape arm routes above `maxp` and `gate_PV`, or if
raw logZ is biased by bank size and the correction removes the bias.  A tie with `maxp`
is a tie: confidence is then already a comparable cross-bank statistic.

**R2 eviction.**  N=128 memories, keep M in {64, 32}; scores are label-free and computed
on train cues only.  `mass` = sum of attention over cues (incumbent).  `dlogZ` = the
exact first-order effect of removing key i on logZ, -log(1-p_i), which is *monotone in
mass*: it is reported as an equivalence check, not as a second method.  `dcov` = the
exact change in ||Cov_p(v,k)||_F when i is removed.  `shape` = key i's mass-weighted
alignment with the second singular direction of Cov_p(v,k) -- the memory that holds two
basins apart without ever winning.  Baselines: random, newest-index, and an oracle that
keeps the M memories with the highest test-time source count.  Scored on test top-1 at
the fixed budget, plus recall on the bottom mass quartile (the tail), plus the Spearman
correlation of each landscape score with mass.  R2 is informative if a landscape score
keeps more tail recall than mass at the same overall accuracy, or if mass already
matches the oracle ordering (in which case there is no eviction decision to improve).

**R3 steering.**  `pair`: cue mixes a memory and its nearest neighbour b; `triple`: a
memory and its two nearest neighbours.  Directions: v1 (top singular vector of
Cov_p(v,k)), d_ab = unit(k_a - k_b) -- the direction the top-2 identity already gives
you -- and random.  Steps alpha in {0.05, 0.1, 0.2, 0.4} in the unit-norm cue, then
renormalise.  Scored on: |cos(v1, d_ab)|; flip rate and clean-flip rate (the two ends
land on the pair, one each) for v1 vs d_ab vs random; and on triples, how many of the
three source memories the set {argmax q, argmax q+-v1, argmax q+-v2} covers, against
four random directions as the control.  R3 is informative if v1 separates the pair when
d_ab does not (e.g. three-way cases where the top-2 identity is wrong), or if +-v1,+-v2
covers three readings that a single difference direction cannot.  If |cos(v1,d_ab)| ~ 1
and the flip rates match, the honest conclusion is that the Hessian *is* the pair
direction -- a correct interpretation and no new capability.

**Not run here.**  Role 4 (continual writing by surprise/crowding) needs a stream and a
consolidation policy; R2 measures the same scoring problem statically, and the streaming
version is deferred rather than approximated.  Role 5 (landscape as a training signal)
needs the key matrix trained with an added term, which is a different harness.  Role 6
(landscape across tokens) needs a token-level model and a text corpus; this harness has
neither, and pretending a toy bank is a document would not measure the claim.

No thresholds are frozen for this block: it is exploratory, and the report states each
number with its CI and says plainly which comparisons came out as ties.
## Amendment 10b — 2026-11-17 — summary routing, whitened steering, eviction replication

Written before any 10b data existed.  Triggered by a correct objection to §12: the
routing arm routed by full logZ_b, and computing logZ_b reads every key in every bank,
so it costs exactly what flat retrieval costs.  The §12.1 "1/8 the reads" claim is
therefore wrong as stated: the accuracy comparison stands, the compute claim does not.
Only a router that estimates logZ_b from a per-bank summary can save anything.

**R1b summary routing.**  The cumulant expansion of the log-partition over bank b,

    log Z_b ~ log N_b + tau q.mu_b + (tau^2 / 2) q' Sigma_b q,

needs only a count, a mean and a key covariance per bank: O(d^2) per bank per cue
instead of O(N_b d).  Four routers on the same banks, with per-cue key-flop counts
reported for each: `full` (exact logZ_b, the accuracy reference at flat cost), `gauss`
(the expansion), `mean` (drops the covariance term), `sketch` (r = 8 k-means centroids
per bank, log N_b + logsumexp_j(tau q.c_j + log(n_j/N_b))).  Two configurations:
(B=4, N_b=256) where a summary can pay, and (B=8, N_b=32) where the bank is smaller
than the covariance and it cannot.  sigma in {0.5, 1.0}, 5 seeds x 4 banks.

Predictions.  (i) The expansion is a second-moment approximation to a quantity that is
set by the largest score, so it should break where the partition is dominated by one
key.  Reported directly: the per-cue spike measure max_i s_i - logZ_b, correlated with
the signed Gaussian error (true minus approximation) and with its magnitude.  A positive
signed correlation means the Gaussian under-estimates exactly when the partition is
spiky.  (ii) The cost ordering flips with N_b/d: at N_b=32, d=64 the Gaussian quadratic
form is more expensive than the scan it replaces, and a 4-centroid sketch costs the same
as the scan; at N_b=256 the ratio is 4x for Gauss and 32x for the sketch.  (iii) The
sketch should beat the Gaussian when the bank is multi-modal and lose when it is not;
the centroids are the next order of detail after the covariance.

**R2b eviction replication.**  The dcov result rests on one cell among many arms, so it
is repeated on fresh banks (seed offset 7000, no overlap with §12) at sigma=1.0, M=64,
the cell where it was strictly better than mass.  Cost is measured, not asserted: the
mass sum runs on the posterior that the retrieval pass already produced, while dcov needs
a per-key bilinear form, O(N d d_v) per cue.  Both are reported so the result can be read
as quality per unit of compute.

**R3b whitening and sharpening.**  Two explanations for the steering failure, separated:
  a. The background key covariance is not isotropic, so the total covariance is dominated
     by the shared direction.  Test: the generalized eigenproblem
     Cov_p(k) v = lam Sigma_global v (whiten by Sigma_global^{-1/2}), direction w1.
     Prediction: a near no-op.  At rho=0.9 the global covariance is ~rho uu' + (1-rho) I,
     which is isotropic *within* the residual subspace where k_a - k_b lives, so
     whitening rescales that subspace uniformly and cannot change the eigenvector order.
     If w1 matches h1 in alignment, that is the explanation rather than a failure.
  b. The posterior is diffuse, so the covariance is not near rank one.  Test: sharpen the
     posterior (beta x tau, beta in {1,2,4}), and report the eigenvalue separation
     lam1/lam2 alongside the alignment and the clean-flip rate, with d_ab's clean flip at
     the same temperature as the control.  Prediction: if diffuseness is the cause,
     lam1/lam2 rises with beta and the h1 clean rate approaches d_ab's; if lam1/lam2
     rises while the alignment stays near 0.6, then the dominant direction is a different
     memory's residual and the disambiguation axis is not the top eigenvector at all.

No thresholds are frozen.  Everything is reported as a measured number with its CI, and
any comparison that comes out as a tie is called a tie.
## Amendment 10c — 2026-11-17 — cell-probe routing and real activation keys

Written before any 10c data existed.  Two objections to §12.6, both correct, drive this.

**R1c cell-probe (IVF) routing.**  §12.6 concluded that second-moment summaries cannot
recover logZ because logZ is set by the largest scores, not by the mean and variance.  The
right sublinear estimator keeps the sharpness instead of smoothing it: a coarse quantiser
(k-means cells, r=16) scores only the centroids, then the exact scores are computed for the
memories in the top n_probe cells.  The estimator is

    log Z_b ~ logsumexp_{i in probe}(s_i) + log(N_b / n_probed_b),

the importance-corrected sum, which is unbiased for a random probe and keeps banks with
different probed sizes comparable.  n_probe in {1,2,4,8,16}; flops counted as r centroid
dots plus one dot per probed memory, per bank per cue.  Four banks of 256 memories, sigma
in {0.5, 1.0}, 5 seeds x 4 banks.

Prediction: routing accuracy should climb toward the full-scan reference as n_probe grows,
and the interesting number is where it saturates relative to the flop saving.  The binding
constraint at low n_probe is whether the probe contains the target memory at all
(`probe_recall`); end-to-end accuracy cannot exceed it.  Expect saturation by n_probe ~ 4-8
at sigma=0.5 and a much worse curve at sigma=1.0, where no router separates the banks well.

**R1d real activation keys.**  Every geometric claim in §12 (isotropic residual subspace,
no scale separation, whitening is a no-op) is a property of the correlated-pattern
generator.  Real activations are strongly anisotropic, so those claims may not transfer.
Keys are the captured hidden states of Phi-4-mini (d_raw = 3072, 9 streams of ~983 rows),
PCA-reduced to d=64 and renormalised.  Banks are the streams, so routing asks which stream
a novel hidden state came from; exactly one bank holds it and all banks come from the same
model and layer, so the shared component cannot give the answer away.  tau is set so the
median top1/top2 cosine gap is 2 logits, measured on train cues.

Predictions, stated so they can fail: (i) the geometry is anisotropic (top-64 singular
share well above 64/3072), so the specific reason whitening was a no-op in §12.8 does not
apply here; if whitening helps anywhere, it is here; (ii) logZ still routes above maxp,
but PV should carry more bank identity than the toy's ~15-20%, because real values are
informative about their source; (iii) steering: if the failure was the generator rather
than the idea, the whitened direction should close some of the gap to k_a - k_b.

Eviction is NOT run on real keys: the capture carries no request-frequency distribution,
and with uniform targets an eviction score is capped at the budget fraction, which §12's
quarantine already established is an uninformative design.  Roles 4-6 and adaptive tau
remain unrun.

Caveat that bounds the claim: these are hidden states (the input to an attention layer),
not W_k h.  No HF-format weights are on disk (models/ holds GGUF only), so a literal
attention-key capture is not available here.  The test settles activation geometry, not
the key projection, and W_k would rotate it.
## Amendment 10d — 2026-11-17 — controls on the routing result

**Written after the runs it describes.**  These are new measurements rather than changes to a
frozen gate, and I am not going to dress them as predictions: the design below was chosen
from the objections raised after §13, and the outcomes are already known.  What follows is
the protocol as executed, so the controls can be checked and repeated.

Three defects and three controls.

**D1 determinism.**  `torch.pca_lowrank` is randomised.  Two calls on the same matrix
returned projected keys differing by up to 1.5 in a single coordinate, and tau (set from the
median top-2 score gap) moved between 506 and 556 across draws, so every number in §13.2-13.3
depended on the draw, not only on the data.  Replaced by the exact Gram eigendecomposition
`eigh(Xc'Xc / n)`, sign-canonicalised.  Verified: two calls now differ by 0.0 exactly.

**D2 geometry normalisation.**  The old loader normalised the singular-value shares by the
sum of the *truncated* spectrum returned by `pca_lowrank` (80 components), not by the total.
Every "share of variance" figure in §13 was therefore overstated.  Corrected shares are
recomputed against the full spectrum.

**D3 PCA fit set.**  PCA was fit on all rows, including the rows used as test cues.  The
loader now takes `exclude_test`, which drops each stream's last quarter from the fit.  Both
variants are run so the size of the leak is measured rather than asserted.

**C1 matched-cost ANN.**  On the same cues and banks, compare probed-logZ routing against an
ordinary IVF search over the union: one shared quantiser with B*r cells, top `n_probe_total`
cells read globally, routing taken from the bank of the single best probed memory.  Flops
count centroid dots (B*r*d per-bank, r_tot*d shared) plus one dot per probed memory, so the
two allocations are compared at equal cost.  Prediction: if the shared budget matches the
per-bank curve, energy routing is ANN retrieval and the landscape quantity adds nothing.

**C2 per-bank vs shared quantiser.**  Same comparison read the other way: per-bank cells
allocate reads spread over banks, a shared quantiser concentrates them where the best match
is.  Prediction: if per-bank quantisers do not beat the shared one at matched cost, the
per-bank structure is doing no work.

**C3 incumbents on real banks.**  maxp was never a valid cross-bank router, so the toy's
13 pp margin was against the weakest rival.  Add the mean router (one dot per bank), a
k-means 8-centroid router, and the Gaussian (mean+covariance) logZ estimate, on the same
cues.  Prediction: with nine different documents the mean router should reach 80-90%, cutting
the claimed margin from +86 pp to ~+10 pp.

**C4 maxp mechanism.**  Test "near-duplicates flatten the owning bank's max" directly: count
rows per bank with cosine > 0.9 to the cue (local density), correlate with maxp across
(cue, bank) pairs, and rerun routing after greedily deduplicating each bank at 0.95.
Prediction: negative rho(maxp, density), and maxp recovering above chance after dedup.
## Amendment 10e — 2026-11-17 — role 1, the annealed ladder as a scale-space

Two corrections to the record first, then the protocol.

**Correction 1: the incumbents were path data.**  maxp, the top-2 gap and logZ are all
functions of the *first* softmax step, i.e. of the trajectory's opening, not of the memory
geometry alone.  §12-§14 read "landscape features lose to the incumbent" as evidence about
local scalars vs local scalars, but maxp and the top-2 gap are stage-0 path quantities.  The
line between "landscape features" and "the path" was never drawn, so the conclusion should
read: *stage-0 path information beats the later local scalars*.  Whether the path itself
carries something the first step and the endpoint do not is a different question, and until
this amendment it was untested.

**Correction 2: every role so far was recall.**  Routing, eviction and steering all score
"which memory/direction".  R6 was the only path test and its iterates were near-duplicates of
the endpoint.  Role 1 is the first test here whose metric is not retrieval.

**Protocol.**  Annealed mean-shift on the real hidden states,

    x_{t+1} = unit( sum_i softmax(beta * z_i) k_i ),   z = (x.K - mean_i)/std_i

with beta in score-standard-deviation units.  This is not cosmetic.  Annealing in raw logit
units at the retrieval temperature from the routing runs leaves **no coarse end at all**:
measured at beta = 0.031*tau the iterate is already 0.992-aligned with its own fixed point and
at perplexity 9, because the softmax is one-hot at that temperature.  Dividing the score field
by its own standard deviation restores a diffuse end (perplexity 8711 at beta=0.25).

Cues are rows from the nine capture files, noised at sigma=0.5.  Labels: **coarse** = which
file (9 problems, chance 11.1%), **fine** = position within its stream, as a quartile (chance
25%) and as a regression.  Readouts are ridge on the iterate, fit on cues from even-numbered
streams and scored on odd-numbered ones, so no test cue shares a stream or its near-duplicates
with training.  Stream identity itself is not a usable coarse label under that split -- a
72-way classifier cannot name a stream it never saw -- so the coarse label is the file.

Criterion, fixed before the run: the coarse readout should saturate at a much smaller beta than
the fine one; the iterates must stay clear of the endpoint over that range (R6's failure mode);
and granularity, not accuracy, is the quantity that must order with beta.

Measured (2000 cues, 72 streams, d=64, tau=452.3):

| beta | perplexity | cos to endpoint | file % | pos quartile % | pos R2 |
|---|---|---|---|---|---|
| 0.25 | 8711 | 0.223 | 18.9 | 32.2 | +0.081 |
| 0.5 | 7885 | 0.397 | 29.1 | 33.9 | +0.124 |
| 1 | 3805 | 0.647 | 34.5 | 36.7 | +0.123 |
| 2 | 344 | 0.840 | 44.9 | 38.9 | +0.219 |
| 4 | 58 | 0.944 | 69.0 | 45.3 | +0.364 |
| 8 | 15 | 0.985 | 83.5 | 48.3 | +0.495 |
| 16 | 7.8 | 0.996 | 88.2 | 51.6 | +0.568 |
| 32 | 4.8 | 0.999 | 90.4 | 52.7 | +0.591 |
| 64 | 3.1 | 1.000 | 91.1 | 53.0 | +0.595 |

Beta at 90% of each readout's own ceiling: **file 16, position quartile 16, position R2 16.**
Identical.  The criterion fails.

Three things follow, and they are all negative for role 1 as stated:

1. **No scale separation.**  There is no temperature at which the coarse label is at ceiling
   while the fine one is not, which is the signature a dendrogram would produce.  All three
   granularities become readable together and saturate together.
2. **The ordering, where there is one, runs backwards.**  Measured against its own ceiling the
   coarse label is the *most* beta-sensitive quantity on the ladder (18.9 -> 91.1, a 4.8x
   range) and the fine position the least (32.2 -> 53.0, 1.6x).  So the coarse structure is
   not recovered first; it is the structure that needs the most sharpening.
3. **R6's hazard recurs.**  cos(iterate, endpoint) passes 0.95 at beta=8 and 0.999 by beta=32,
   so the whole useful part of the ladder is a near-copy of the fixed point.  The 90%
   saturation at beta=16 sits inside that band.

What the ladder does show is that the iterates are informative at all: a 64-dim iterate
recovers the source problem at 91% from 9 classes, and position quartile at 53% from a 25%
base, both on held-out streams.  That is a representation result about the hidden states, not
about annealing: the same information is available at the endpoint, and the ladder's only
contribution is to destroy it as beta falls.
## Amendment 10f — 2026-11-17 — ladder vs path, and labels that actually nest

Two flaws in §15, both correct.

**The §15 ladder was multi-temperature fixed points, not a trajectory.**  Each level ran
feedback steps from the same cue and recorded only the converged iterate.  The trajectory at
fixed beta -- where each retrieval moves the query and the next retrieval uses the moved
query -- was never measured.

**The §15 labels could not support a dendrogram claim.**  Position in a stream is a continuous
drift; with participation ~75 the geometry is continuous, so annealing there is kernel
smoothing at different bandwidths.  A hierarchy needs discrete nested labels.  The capture
carries `off`/`len` character offsets into the decoded text, so token surface forms are
recoverable exactly with no tokenizer (verified: slicing and re-concatenating reproduces each
stream's text).

Also adopted: the axis is **log effective neighbourhood size**, not beta.  Beta is not
comparable across cues once the score field is standardised per cue; perplexity is.

Protocol:

- **A. Feedback vs ladder.**  Ladder: converged iterate at beta in {0.25 ... 64} (std units,
  6 steps).  Path: fixed beta in {4, 8, 16, 32}, six single steps from the cue, every iterate
  probed.  Comparison is at *matched log perplexity*, not matched beta.
- **B. Discrete nested labels.**  token class (4: space / numeric / punctuation / word) ->
  subclass (12: word-length bucket, digit count, newline flag) -> identity (token type, 268
  observed).  Same readout and split: ridge on the iterate, fit on even streams, scored on odd.

Verdict rules, fixed before the run: if the trajectory at matched perplexity is no better than
the ladder, the path carries multi-temperature information and nothing more.  If class
saturates at a smaller neighbourhood than identity, granularity tracks scale on labels that
nest; if all three saturate together, §15's negative was about the path, not about the labels.

Measured (2000 cues, 72 streams, d=64, 982 train / 1018 test):

- **The trajectory is indistinguishable from the ladder at matched bandwidth.**  Log10
  perplexity is degenerate across the converged levels (3.30-3.34 for beta=2..64), so
  single-nearest matching breaks ties arbitrarily.  Compared against the *range* of ladder
  readouts within +-0.03 log10 perplexity, the trajectory falls inside that range at every
  step and every label, with a largest excursion of +0.2 pp (class 96.5 vs 96.4 at beta=16,
  t=1).  The cue itself (t=0, log10 perp 3.41) has no ladder point within the band: the ladder
  never visits its scale.
- **It converges in one step.**  At beta=16 the t=1 iterate reproduces the ladder's beta=16
  row to the decimal (96.4 / 79.1 / 47.9), and later steps are stationary.
- **It does not traverse scale.**  Log10 perplexity along the path moves 3.41 -> 3.33 and then
  stops; the fixed-point entropy is nearly beta-invariant (3.30-3.34 across beta=4..64).
- **The cue already contains the labels.**  Before any iteration: class 94.2, subclass 74.1,
  identity 43.5.  The best point anywhere on the ladder: 96.4 / 79.1 / 48.2.  The entire
  apparatus adds **+2.2 / +5.0 / +4.7 pp**.
- **No granularity ordering**, even with labels that nest: beta-at-90%-of-ceiling in log10
  perplexity is 3.33 (class), 3.33 (subclass), 3.30 (identity).

Conclusions: the path is *not* a separate information channel at this geometry -- the feedback
structure equals the temperature ladder and neither beats the starting point by more than a few
points.  §15's negative was therefore not an artefact of continuous labels; it repeats on
discrete nested ones, and now has a better explanation than "not hierarchical": the readouts are
already saturated at t=0, so a coarse-to-fine sweep has nothing to add.
## Amendment 10g — 2026-11-17 — can a memory shape the route without changing the destination?

§16's equivalence holds "on these readouts", and that qualifier is the whole point: matching
effective neighbourhood size matches the *spread* of the weight vector, not which memories carry
the weight.  A linear probe is blind to that difference; an intervention is not.

Protocol.  Fix the cue and the temperature, zero one memory's weight (scores set to -1e9, so
indexing stays aligned), rerun the descent, and record

    d1    = 1 - cos(x_1, x_1^-j)              motion of the FIRST update
    dlate = max_{t>=2} 1 - cos(x_t, x_t^-j)   motion of any LATER step
    dT    = 1 - cos(x_T, x_T^-j)              whether it settles in the same place
    same  = nearest memory at T unchanged

"Route-only influence" is d1 ~ 0, dlate large, destination unchanged: a memory that did not
shape the first step and did not decide the answer, yet bent the path between them.  That is the
version of provenance an endpoint readout cannot produce.

Two settings, and the criterion before the run was d1 <= 0.01, dlate >= 0.05, destination
unchanged, at a temperature where the baseline takes >= 4 steps to settle.

**A. Real hidden states** (120 cues, 9 files, d=64, T=5, three interventions per cue):

| beta | late-entering memories | max late motion | d1<=0.01 | of those dlate>=0.05 | destination unchanged |
|---|---|---|---|---|---|
| 4 | 75/120 | 3.5e-2 | 334/336 | 1 | **0** |
| 16 | 25/120 | 5.1e-3 | 310/338 | 0 | **0** |

So silent memories are the norm (334 of 336 are silent at step 1) and late *motion* exists, but
the one case with dlate >= 0.05 changed the destination.  Widening to dlate >= 0.02: 32 rows
across both temperatures, **0 of 32 kept the destination**, mean dT 0.19 and 0.089.

**B. Constructed landscapes.**  A dense target cluster and a decoy placed *beyond* it -- far from
the cue (low weight at t=0), close once the iterate has rotated toward the target (the only
arrangement in which a memory can matter late and not early).  Five attractor sizes x two
temperatures: the baseline settles in 1-2 steps for any mass >= 32, and the decoy's maximum
influence anywhere in the sweep is dlate = 6.4e-4.  A second family (a chain of bridges along the
cue-to-target arc) did produce a 12-step transient, but there dlate = 0.0996 came with a changed
destination, and the influence shrank to 0.004 once the memory was far from the endpoint.

**Zero route-only cases: 0/674 real interventions and 0/10 constructed configurations.**

The reason is structural, not a tuning failure.  At fixed beta the weights are a function of
proximity to the current iterate, and the iterate moves toward the weighted mean of memories that
*already* have weight.  Relevance therefore propagates by proximity from what is already relevant:
a memory can only acquire late weight if the iterate physically approaches it, and the iterate
approaches the mass that the first step already selected.  If a memory sits near that early mass,
it is not silent at step 1; if it is silent, the path does not go near it.  Any late weight strong
enough to bend the route is also strong enough to move the endpoint -- which is exactly what
0/32 stable destinations among the dlate >= 0.02 rows measures.

Consequence for the provenance idea: the question "which memory shaped the route without changing
the destination?" is well posed, and the intervention answers it empirically rather than by probe.
In this dynamics the answer is none, and the reason is that transient length and destination
fragility are the same parameter here.  The distinction the thread needed -- "the path has
information" versus "the first step already did" -- collapses in this map, because there is no
channel by which late information can enter that is not also a channel by which the answer changes.

### Amendment 10g — correction (same date)

Withdrawing two general claims from 10g, keeping the measurement.

`F_{-i}(x) - F(x) = p_i/(1-p_i) (F(x) - k_i)` was verified in float64 to 1e-15, so the effect of
removing a memory depends on the state: it vanishes near the destination (where F -> k_i) and is
largest at an intermediate state far from k_i.  A memory can therefore be inert at the cue,
influential mid-route, and inert again at the endpoint.  "Any late weight strong enough to bend the
route is strong enough to move the endpoint" and "transient length and destination fragility are the
same parameter" are consequences of a fixed-point argument that the state dependence invalidates.

What stands: a measured trade-off.  In the tested families (674 real interventions, 10 constructed
configurations), every memory with dlate >= 0.02 also carried nontrivial weight at convergence, so
0/32 kept the destination; the single dlate >= 0.05 case sat at p_i = 0.14 when it settled.  The
reporting error is also fixed: §17.2's "max late motion" is the baseline's step-to-step move, while
dlate is the displacement between trajectories, and the former does not bound the latter.  Future
runs report norm displacement alongside the cosine, since a radial pull moves ||F|| without moving
direction (cue 93: predicted 0.0144, measured d1 1.3e-4).
