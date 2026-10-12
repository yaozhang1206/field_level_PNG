# Stage 2 notes: field-level posterior of $f_{\rm NL}$ on DESI DR1 mocks

Written 2026-10-12 (in progress). Code: `src/posterior.py`, `src/sampler.py`, `scripts/run_sampler.py`; tests in `test/`;
chain summaries in `results/chain_*.json` (chain states `results/chain_*.npz` are kept locally, not in git).

---

## 1. What Stage 2 had to settle

Stage 1 ended with two numbers for the $160^3$ ($k_{\max} = 0.1$) DR1 configuration: the exact Gaussian-marginal Fisher of
the model linearised in $\varepsilon$ gave $\sigma(f_{\rm NL}) = 4.0$, while the strictly linear (two-point) model with the same
window gave 8.3 at $80^3$. The gap is information carried by the quadratic couplings ($b_{\phi\delta}\,\delta\phi_g$,
$f_{\rm NL}\phi_g^2$, 2LPT), and a Fisher matrix can over-credit it. Only sampling the posterior decides. The brief's Stage 2
go/no-go is on cost: $f_{\rm NL}$ correlation length $\le 10^3$ gradient evaluations and a $62.5\,h^{-1}$Mpc production run
$\lesssim 1$ GPU-month.

## 2. Posterior and sampler

$$\ln p(\varepsilon, \theta \mid d) = -\tfrac12\sum_x \frac{[d(x) - M(\varepsilon,\theta)(x)]^2}{\sigma^2(x)} - \tfrac12|\varepsilon|^2 + \ln p(\theta),$$
Gaussian noise with $\sigma^2 = \bar n V_{\rm cell}$ (a Poisson option exists; same Fisher, see Stage 1 notes), flat priors on the
free bias parameters unless stated, $b_\phi$ and $b_{\phi\delta}$ fixed by the universality relations. Mocks are drawn from
the same model at $f_{\rm NL,true} = 0$; chains start at the truth (we measure posterior width and mixing, not burn-in).

Sampler state: $[\varepsilon, \theta_{\rm w}]$ with $\theta_{\rm w} = (\theta - \theta_{\rm fid})/\sqrt{{\rm diag}\,F_{\theta\theta}}$ (Stage 1 Fisher
block). NUTS (BlackJAX 1.3) with Stan-style window adaptation of a diagonal mass matrix and the step size, run as a plain
Python loop so that no warmup positions are stored (BlackJAX's `window_adaptation.run` keeps all of them: 6.5 GB at
$160^3$, which was the first out-of-memory failure). Tree depth 7 or 8. Chains checkpoint every 250 samples and can be
resumed from the saved position, step size and mass matrix, so a $160^3$ chain is accumulated in 2-hour interactive
sessions. Gradients: reverse-mode view of the forward model (DISCO-DJ scatter `custom_vjp`), 14 ms at $80^3$ and 63 ms at
$160^3$ on one GH200.

Validation (`test/notes/posterior16.md`, `sampler16_gauss.md`, `sampler16_linear.md`):
- autodiff gradient vs finite differences: $10^{-7}$ relative (Gaussian), $10^{-3}$ (Poisson, round-off limited);
- with noise-free data the posterior Hessian in $\theta$ equals the Stage 1 $F_{\theta\theta}$ to $3\times10^{-15}$;
- exact Gaussian target ($\theta$ fixed, 4096-dimensional $\varepsilon$): chain means and variances vs the dense inverse
  precision: rms mean error 0.013 (= expected from ESS), variance ratio $1.000$ (5–95 %: 0.94–1.07);
- six mocks at $16^3$: per-mock $f_{\rm NL}$ widths 0.62–1.26, mean 0.77 ± 0.09 vs Fisher 0.64, posterior means with rms 0.80:
  calibrated, with the Fisher a fair but not exact predictor of single-mock widths;
- storage-free window loop vs BlackJAX's: 0.82 vs 0.80, $\tau$ 4.2 vs 3.7;
- identity mass matrix (no adaptation): $\tau = 81$ — the adapted diagonal mass (range 0.02–15) is essential.

## 3. Is the quadratic-coupling information real? Periodic boxes first

Same mock, same cell size as production (125 Mpc/$h$), NUTS under the strictly linear model and under the full 2LPT model
(`test/notes/stage2_smallbox_summary.md`):

| box | linear chain | full chain | ratio | Tier 1 Fisher | linearised Fisher | ratio |
|---|---|---|---|---|---|---|
| $16^3$, $L = 2000$, LRG | 13.3 | 9.3 | 1.42 | 11.2 | 8.6 | 1.30 |
| $32^3$, $L = 4000$, QSO+LRG | 3.01 | 2.35 | 1.28 | 2.82 | 2.02 | 1.40 |
| $32^3$, $L = 4000$, LRG | 3.17 | 2.47 | 1.28 | 2.83 | 2.06 | 1.37 |

In periodic boxes the posterior realises most of the Fisher-predicted gain. The tree-level real-space bispectrum (Tier 2)
is converged in binning (`test/notes/tier2_binning.md`) and still credits these couplings with $< 3$ %: that estimator is
simply not where the information sits.

## 4. DR1 configuration (`test/notes/stage2_dr1_chains80.md`)

DR1 window, QSO+LRG, six shells, $f_{\rm NL,true} = 0$, $b_\phi$ fixed; ± is the chain error from the ESS.

| grid | mock | model | samples | $\sigma(f_{\rm NL})$ | $\tau_{\rm int}$ | ESS |
|---|---|---|---|---|---|---|
| $80^3$ | 0 | linear | 3000 | 7.77 ± 0.33 | 11 | 269 |
| $80^3$ | 0 | full, chain A | 3000 | 6.97 ± 0.23 | 6.4 | 472 |
| $80^3$ | 0 | full, chain B | 2500 | 7.16 ± 0.21 | 4.1 | 608 |
| $80^3$ | 0 | full, priors $\sigma = 2$ on $b_2, b_K$ | 3000 | 6.89 ± 0.31 | 12 | 244 |
| $80^3$ | 1 | linear | 3000 | 7.45 ± 0.28 | 8.4 | 358 |
| $80^3$ | 1 | full | 2500 | 7.51 ± 0.20 | 3.4 | 730 |
| $160^3$ | 0 | linear | 3000 | 6.52 ± 0.17 | 4.3 | 698 |
| $160^3$ | 0 | full, resumed chain (4 sessions) | 3300 | 5.7 ± 0.4 | 24 / 10.5 (depth 7 / 8) | 166 |
| $160^3$ | 0 | full, independent chain | 1200 | 4.8 ± 0.6 | 40 | 30 |
| $160^3$ | 0 | **full, pooled** | 4500 | **5.6 ± 0.3** | | 171 |

**Result.** At the DR1 configuration the full model gains only 0–10 % over the two-point posterior at $80^3$ and $\approx 14$ %
at $160^3$, against 36–50 % predicted by the linearised Fisher. The field-level forecast for DR1 QSO+LRG at $k_{\max} = 0.1$ is
$\sigma(f_{\rm NL}) = 5.6 \pm 0.3$ with $b_\phi$ fixed and $\approx 6.8$ with the 40 % $b_\phi$ prior propagated ($\times1.22$): a 25–30 %
improvement over the DESI DR1 $P(k)$ result (9.2), not the factor two of the Stage 1 Fisher, and above the $\lesssim 5$
threshold. Priors of width 2 on $b_2$, $b_K$ do not change the width, so the broad flat-prior directions in the QSO shells are
not the limitation.

Interpretation of the box–DR1 difference: the coupling information is a modulation of small-scale power by the large-scale
potential, usable only where that large-scale field is itself well measured. In a periodic box with uniform $\bar n$ it is;
with the DR1 window and $n(z)$ the relevant large-scale modes are poorly constrained, while the linearised Fisher holds
them fixed at the fiducial. A masked $16^3$ box test was attempted but is inconclusive at that information level
(widths 24–37, ESS 40–64).

## 5. Cost and sampler comparison

| sampler | grid | gradients per ESS($f_{\rm NL}$) | wall-clock per ESS | notes |
|---|---|---|---|---|
| NUTS, window-adapted, depth 8 | $80^3$ full | $1.6\times10^3$ | 13 s | 255 leapfrog, $\tau$ 4–6 |
| NUTS, depth 7 / 8 | $160^3$ full | $3.0 / 2.7\times10^3$ | 190 / 170 s | 127 / 255 leapfrog, $\tau$ 24 / 10.5 |
| MCLMC (tuned $L$, step) | $80^3$ full | $1.4\times10^2$ | 2.7 s | posterior consistent with NUTS (std 7.4 vs 7.0–7.2) |
| MCLMC (300-step tuning, step 592) | $160^3$ full | $0.95\times10^2$ | 10 s | **biased**: mean $-16.7$ vs $-3.85 \pm 0.4$, std 4.4 vs 5.6 |
| MCLMC, step/2 (296), same $L$ | $160^3$ full | $2.5\times10^2$ | 26 s | mean $-4.2$, std 5.75: consistent with NUTS |
| MCLMC, step/4 (148), same $L$ | $160^3$ full | $4.3\times10^2$ | 45 s | mean $-3.5$, std 5.58: converged |

NUTS misses the brief's $10^3$-gradient target by a factor 2–3 at $160^3$. MCLMC at the auto-tuned step is biased, but at
half that step its mean and width agree with NUTS and are stable under a further halving: **the production sampler is
MCLMC at half the auto-tuned step, $\approx 250$ gradients ($\approx 26$ s) per effective $f_{\rm NL}$ sample at $160^3$**, i.e. 1000
effective samples in $\approx 7$ GPU-hours, against $\approx 2$ GPU-days with NUTS. The brief's Stage 2 cost targets are met
(correlation length $\le 10^3$ gradients; production $\ll 1$ GPU-month). Every production run should carry a short NUTS
reference chain to confirm the MCLMC step.

## 6. Practical findings

- BlackJAX window adaptation stores all warmup positions; at $160^3$ that is 6.5 GB and caused two OOM failures. The
  same adaptation in a Python loop costs nothing in memory and reproduces the result.
- NUTS tree buffers at depth 8 need $\approx 8$ GB at $160^3$; they fit once warmup storage is gone (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`).
- MCLMC's `mclmc_find_L_and_step_size` also stores its tuning trajectory (20 GB for 3000 steps at $160^3$); 300 steps fit.
- `srun` on the interactive partition intermittently fails with "Error configuring interconnect"; `sbatch` to the same
  partition works.
- Chain `.npz` files with the resume state are 63 MB at $160^3$; they are excluded from git.

## 7. Open items

- MCLMC step-size convergence at $160^3$ (running); then the production sampler decision.
- Second mock at $160^3$ (queued) to measure the fiducial-field scatter of the width (10 % at $80^3$).
- $b_\phi$, $b_{\phi\delta}$ sampled with the 40 % prior at a non-zero fiducial $f_{\rm NL}$.
- Poisson likelihood check at $160^3$ (6 QSOs per voxel).
- Stage 3 (model mismatch on N-body halo catalogues) and Stage 4 (systematics) are untouched.
