# DESI f_NL Field-Level Inference — Project Summary and Stage 1 Plan

This file is intended as the working brief for a Claude Code session on NCSA Delta / Delta AI.
Drop it in the repo root (optionally rename to `CLAUDE.md`).

---

## 1. Project goal

Constrain local primordial non-Gaussianity ($f_{\rm NL}^{\rm loc}$) with Bayesian field-level inference
(BORG-style forward modelling) applied to DESI DR1 quasars (QSO) and luminous red galaxies (LRG).

**Definition of "doable"** (decided in planning):
- Idealized (no systematics) forecast $\sigma(f_{\rm NL}) \lesssim 5$ at $k_{\max} = 0.1\,h/{\rm Mpc}$ with bias marginalized.
- Final target $\sigma(f_{\rm NL}) \le 6$ on DR1 QSO+LRG with $|\Delta f_{\rm NL}| < 2$ on contaminated mocks.
- Comparison baseline: DESI DR1 power-spectrum result $f_{\rm NL} = -3.6^{+9.0}_{-9.1}$ (Chaussidon et al. 2024).

---

## 2. Reference papers and the numbers we rely on

### 2.1 Chaussidon et al. 2024 — DESI DR1 P(k) PNG analysis (arXiv:2411.17623)
- Samples: 2,130,621 LRGs (0.4 < z < 1.1); 1,189,129 QSOs (0.8 < z < 3.1).
- Model: Kaiser + Lorentzian damping,
  $$P(k,\mu) = \frac{\left[b_1 + b_\phi f_{\rm NL}/T_{\Phi\to\delta}(k) + f\mu^2\right]^2}{\left[1 + (k\mu\Sigma_s)^2/2\right]^2}\,P_{\rm lin}(k) + s_{n0}.$$
- $b_\phi = 2\delta_c(b_1 - p)$, $\delta_c = 1.686$; $p = 1.6$ (QSO), $p = 1.0$ (LRG).
- Free parameters per tracer: $(f_{\rm NL}, b_1, s_{n0}, \Sigma_s)$. Cosmology fixed to Planck 2018.
- Data vector: $P_0$ and $P_2$ only. $k$ range: $0.003 < k < 0.08\,h/{\rm Mpc}$ (LRG $k_{\min}$ raised to 0.006).
  $\Delta k = 0.001$ ($\ell=0$), $0.002$ ($\ell=2$).
- FKP $P_0$: $3\times10^4$ (QSO), $5\times10^4$ (LRG) $(h^{-1}{\rm Mpc})^3$. Box for $P(k)$ estimation: $16000\,h^{-1}$Mpc, $6\,h^{-1}$Mpc cells.
- Effective redshifts (Table 3):
  - LRG: $z_{\rm eff}$(FKP) = 0.733; $z_{\rm eff}$(OQE, $\ell=0$) = 0.754
  - QSO: $z_{\rm eff}$(FKP) = 1.651; $z_{\rm eff}$(OQE, $p=1.6$, $\ell=0$) = 2.082; $\ell=2$: 1.989
- Linear bias evolution $b_1(z) = a(1+z)^2 + b$ (Table 2; $a\approx0.209$ LRG, $a\approx0.237$ QSO; $b$ values in paper).
- Fitted $b_1$ at $z_{\rm eff}$: LRG $\approx 2.06$ (FKP); QSO $\approx 2.44$ (FKP), $\approx 3.08$ (OQE).
- Mean angular densities (Table 1): LRG $\approx 530\,{\rm deg}^{-2}$; QSO $\approx 185\,{\rm deg}^{-2}$.
- Effective-area fractions (LRG, QSO): North (18,20)%, South-NGC (48,44)%, South-SGC (25,27)%, DES (9,9)%.
- **Statistical errors from clean EZmocks (what Stage 1 must reproduce):**
  - QSO DR1 OQE: $f_{\rm NL} = -3^{+13}_{-10}$ (Table 4)
  - QSO DR1 FKP: $f_{\rm NL} = 3^{+18}_{-13}$
  - LRG DR1 FKP: $f_{\rm NL} = 5^{+18}_{-11}$
  - QSO+LRG DR1 OQE: $f_{\rm NL} = 1.1^{+10.5}_{-7.9}$ (Table 5)
  - Y5 forecast combined: $\sigma \approx 6.5$–$7$
- Dominant systematic: imaging systematics; mitigated by template regression → angular integral
  constraint (AIC); radial integral constraint (RIC) from shuffled randoms. Both absorbed into the
  window matrix $W \to W - W^{\rm RIC} - W^{\rm AIC}$. These affect the central value and $k_{\min}$ only, NOT the
  statistical error.

### 2.2 Andrews et al. 2024/2026 — Euclid field-level f_NL forecast (arXiv:2412.11945; A&A 711, A94)
- BORG, 2LPT, box $L = 8000\,h^{-1}$Mpc, grids $32^3/64^3/128^3 \to \Delta L = 250/125/62.5\,h^{-1}$Mpc, $k_{\max} = 0.025/0.05/0.1$.
  (Published version extends to $\Delta L = 31.25$.)
- $f_{\rm NL}$ enters as $\Phi_{\rm NG} = \phi_g + f_{\rm NL}\left(\phi_g^2 - \langle\phi_g^2\rangle\right)$ on the primordial potential before the transfer function.
- Bias:
  $$\rho_g = \langle N\rangle\left[1 + b_1\delta + \tfrac{b_2}{2}\delta^2 + b_K K^2 + b_\phi f_{\rm NL}\phi_g + b_{\phi\delta} f_{\rm NL}\,\delta\,\phi_g\right],$$
  $b_\phi = 2\delta_c(b_1 - p)$; $b_{\phi\delta} = b_\phi - b_1 + 1 + \delta_c\left[b_2 - \tfrac{8}{21}(b_1 - 1)\right]$. Default $p = 0.55$.
- Gaussian likelihood with $\sigma^2 = \langle N\rangle$. RSD included. Single fixed redshift per catalogue (no light cone).
- Sampler: Gibbs; HMC on initial conditions, slice sampling on $f_{\rm NL}$ and bias. Burn-in $\sim 5000$ samples.
  $f_{\rm NL}$ autocorrelation length $\approx 10{,}000$ samples (when bias is sampled).
- Results ($p=0.55$): $\sigma(f_{\rm NL}) = 6.0$ (250), 4.0 (125), 2.3 (62.5). $p=1$ at 125: 6.3. Fixed bias at 125: 2.6.
  Sampling $b_\phi$, $b_{\phi\delta}$ with 40% prior at 125: 5.0.
- Mocks generated with the same forward model (no model-mismatch test).

### 2.3 Related 2026 work
- Andrews et al. 2026, BORG on Quaia quasars in a $(10\,h^{-1}{\rm Gpc})^3$ box (arXiv:2602.02363) — template for QSO geometry/light cone.
- Andrews et al. 2026, Quijote-PNG field-level $f_{\rm NL}$ on N-body halo catalogues (arXiv:2603.20855): $\sim1.3\times$ improvement over P+B at $k_{\max}\approx0.1$.
- McAlpine et al. 2026, Manticore-Deep (arXiv:2606.10020): COLA-based BORG, $256^3$ tiles at $4\,h^{-1}$Mpc, 0.5–1.5M CPU-h per tile — the cost reference for high-resolution field-level runs.

---

## 3. Decisions made during planning

1. **Zel'dovich vs 2LPT**: ZA is accurate enough for $f_{\rm NL}$ at $k \le 0.05$ and probably 0.1, but gives
   no meaningful speed-up (2LPT is $\sim2\times$ ZA per gradient and both are cheap at $80^3$–$160^3$ grids).
   Default to 2LPT; test ZA in Stage 3 for model mismatch.
2. **Bottleneck is chain length, not the forward model.** $f_{\rm NL}$ mixing ($\sim10^4$-sample correlation
   length) is the thing to fix: joint HMC/NUTS on (field, $f_{\rm NL}$, bias), Fourier-diagonal mass matrix,
   MCLMC (BlackJAX) as the candidate accelerated sampler.
3. **Tooling**: build the forward model on **DISCO-DJ** (List, Hahn, Flöss, Winkler; JAX end to end;
   arXiv:2510.05206 for the PM/LPT module, arXiv:2311.03291 for the Einstein–Boltzmann module).
   Chosen over JaxPM/pmwd because: (i) nLPT recursion gives ZA/2LPT/3LPT from one code path, so the
   Stage 3 ZA-vs-2LPT test is a one-parameter change; (ii) de-aliasing, sheet-based resampling and
   higher-order mass assignment suppress discreteness/aliasing power that b2/bK would otherwise
   absorb and leak into $f_{\rm NL}$; (iii) DISCO-EB provides differentiable $P_{\rm lin}$ and $T_{\Phi\to\delta}(k)$ consistent
   with the solver; (iv) memory-efficient adjoint for grids beyond $160^3$; (v) the user has verified
   its accuracy exceeds pmwd/JaxPM. Own JAX code for: PNG perturbation on the primordial potential,
   bias model, selection/light-cone, likelihood. Request BORG access through Aquila in parallel.

   **Stage 0 API checks for DISCO-DJ (do before Stage 2):**
   - Entry point: can the LPT module take the primordial potential (or white noise) so that
     $\Phi_{\rm NG} = \phi_g + f_{\rm NL}(\phi_g^2 - \langle\phi_g^2\rangle)$ is applied *before* the transfer function, and $\phi_g$ at the
     Lagrangian position is available for the $b_\phi f_{\rm NL}\phi_g$ and $b_{\phi\delta} f_{\rm NL}\delta\phi_g$ bias terms? If the API
     takes $\delta_L$ as input, implement the PNG step in JAX and hand $\delta_L$ to DISCO-DJ.
   - Differentiability of the chosen de-aliasing / resampling / mass-assignment options at the
     settings used (some discreteness-suppression schemes use sorting or nearest-neighbour ops
     that break or slow gradients).
   - Gradient cost and memory of 2LPT forward+adjoint at $160^3$ and $320^3$ particles on one GH200/A100.
4. **Hardware**: Stage 1 (Fisher) is CPU-only. Stage 2+ uses GPUs (Delta AI GH200 or Delta A100).
5. **Systematics strategy**: model imaging templates inside the forward model with sampled
   amplitudes (avoids the AIC) rather than pre-weighting the catalogue.
6. **Light cone**: QSOs span $a \approx 0.24$–$0.55$; use redshift shells (or a light-cone model) from the
   start, including in the Stage 1 Fisher.
7. **$b_\phi$**: run everything both with $b_\phi$ fixed (UMF relation) and sampled with a 40% Gaussian prior.
8. **ELGs excluded** (large-scale excess unresolved in DR1).

---

## 4. Four-stage feasibility plan (overview)

| Stage | What | Go/no-go | Hardware | Time |
|---|---|---|---|---|
| 0 | Define targets, DISCO-DJ API checks (see §3.3), BORG access | — | 1 GPU for API/timing checks | 1 wk |
| 1 | Fisher: reproduce DESI $P(k)$ $\sigma$, then field-level Fisher | field-level $\sigma \lesssim 5$ at $k_{\max} = 0.1$, bias marginalized | CPU | 3–4 wk |
| 2 | Self-consistent mock MCMC at $125\,h^{-1}$Mpc, QSO only; sampler comparison | $f_{\rm NL}$ corr. length $\le 10^3$; 62.5 production $\le\sim1$ GPU-month | GPU | 4–6 wk |
| 3 | Model mismatch on Abacus / Quijote-PNG halos; ZA vs 2LPT | $|\Delta f_{\rm NL}| < 0.5\sigma$ at $k_{\max} = 0.1$ | GPU | 4 wk |
| 4 | Imaging + radial systematics on contaminated mocks with template sampling | $|\Delta f_{\rm NL}| < 2$; $\sigma$ degraded $< 30\%$ | GPU | 6–8 wk |

---

## 5. Stage 1 — detailed plan

### 5.1 Purpose
Establish the information content of DESI DR1 QSO+LRG on f_NL with no systematics, in two steps:
(A) reproduce the DESI $P(k)$ statistical errors as a calibration of volume/$n(z)$/bias conventions;
(B) compute the field-level Gaussian Fisher under the full bias model to get the idealized $\sigma(f_{\rm NL})$.

### 5.2 Environment (Delta)
- Partition: CPU (`cpu` on Delta) for Part A and the closed-form checks. Part B's Schur-complement
  Fisher runs on CPU in < 1 h at $80^3$ but is minutes on one GPU; use a GPU for the $160^3$ runs.
- Python env: numpy, scipy, classy (CLASS) and/or DISCO-DJ (discodj; its DISCO-EB module can replace
  CLASS and gives autodiff derivatives of $P_{\rm lin}$ and $T_{\Phi\to\delta}$), astropy, healpy, fitsio/h5py, matplotlib, jax (CPU).
  Optional: cosmoprimo (DESI wrapper), pypower, mockfactory (DESI, for randoms/window).
  Use the same DISCO-DJ environment for Stage 1 and Stage 2 so cosmology/transfer conventions carry over.
- Memory: 16–32 GB for everything except the optional window-matrix build ($\sim100$ GB, few hours).
- Store large inputs (randoms) on scratch; keep repo small.

### 5.3 Inputs to obtain
1. DESI DR1 LRG and QSO clustering catalogues + randoms (public DR1 release). If unavailable
   immediately, start with the paper's Table 1 densities, Fig. 2 n(z) (digitize), and an approximate
   footprint (DR1 ≈ 7,500 deg²; use the Legacy Survey footprint masked to DR1 tiles if available).
2. Planck 2018 cosmology for CLASS: $\Omega_m=0.3153$, $\Omega_\Lambda=0.6847$, $\Omega_b=0.0493$, $h=0.6736$, $A_s=2.1\times10^{-9}$, $n_s=0.9649$.
   (DESI paper fixes Planck 2018; the Euclid paper uses the same.)
3. Bias values: $b_1(z)$ from DESI Table 2; $b_2$, $b_K$ from a reasonable prescription (e.g. Lazeyras et al.
   $b_2(b_1)$ fit; $b_K = -\tfrac{2}{7}(b_1-1)$ as a placeholder) — only needed for part B.

### 5.4 Part A — Reproduce DESI P(k) Fisher

**Tasks**
1. `cosmology.py`: wrap DISCO-EB (fallback: CLASS); return $P_{\rm lin}(k,z)$, $T_{\Phi\to\delta}(k,z) = \sqrt{P_{\rm lin}/P_\Phi}$, $f(z)$, $D(z)$.
   Cross-check DISCO-EB vs CLASS $P_{\rm lin}$ at $z_{\rm eff}$ to $\le0.5\%$ as a sanity test.
   $$P_\Phi(k) = \frac{9}{25}\,\frac{2\pi^2}{k^3}\,A_s\left(\frac{k}{k_{\rm pivot}}\right)^{n_s-1},\qquad k_{\rm pivot} = 0.05\,{\rm Mpc}^{-1}.$$
   Check units ($h$/Mpc vs 1/Mpc) carefully.
2. `model_pk.py`: implement Eq. 2.4 of Chaussidon et al.; Legendre multipoles $P_0$, $P_2$ via $\mu$-integration.
   Parameters $\theta = (f_{\rm NL}, b_1, s_{n0}, \Sigma_s)$. Verify $b_\phi(b_1,p)$.
3. `geometry.py`: $n(z)$ per tracer, $V_{\rm eff} = \int \left[\frac{\bar n(z)P_0}{1+\bar n(z)P_0}\right]^2 dV$, using the paper's $P_0$ values.
   First pass: analytic shell volumes × sky fraction. Second pass: from randoms.
4. `fisher_pk.py`: Gaussian covariance per $k$-bin of width $\Delta k$ for the multipoles
   $P_\ell$, $\ell\in\{0,2\}$, with FKP weighting integrated over the survey $n(z)$
   (Feldman–Kaiser–Peacock / Grieb et al. 2016):
   $${\rm Cov}\big[P_\ell(k), P_{\ell'}(k)\big] = (2\ell+1)(2\ell'+1)\,\frac{2(2\pi^2)}{k^2\Delta k}
     \int_0^1 d\mu\; L_\ell(\mu) L_{\ell'}(\mu)\, S(k,\mu),$$
   $$S(k,\mu) = \frac{\int dV\, n^4 w^4\,[P_g(k,\mu) + 1/n_{\rm obs}]^2}{\left(\int dV\, n^2 w^2\right)^2},
     \qquad w = \frac{1}{1 + nP_0},$$
   where $n = n_{\rm NX}(z)$ sets the FKP weights and $n_{\rm obs} = \bar c\,n_{\rm NX}$
   ($\bar c = A_{\rm eff}/A_{\rm full}$) is the completeness-reduced density that sets the shot noise.
   Because $n_{\rm obs} w_{\rm tot} = n_{\rm NX}/(1+n_{\rm NX}P_0)$ is independent of completeness,
   sample variance is set by the full tiled footprint ($A_{\rm full} = 9070\,{\rm deg}^2$) while shot
   noise is $1/(c\,n_{\rm NX})$. For uniform $n$ in volume $V$ this reduces to
   ${\rm Var}[P_0] = 2[P_g + 1/n]^2/N_{\rm modes}$ with $N_{\rm modes} = V k^2\Delta k/(2\pi^2)$.
   Bins in $k$ are independent (block-diagonal); the $\mu$ integral uses Gauss–Legendre on $[0,1]$.
   Derivatives $\partial P_\ell/\partial\theta$ by JAX autodiff; the covariance is held at the
   fiducial parameters ($f_{\rm NL}=0$, $s_{n0}=0$, $\Sigma_s = 3$ Mpc/$h$).
   $$F_{ij} = \sum_k \frac{\partial P^{T}}{\partial\theta_i}\,C^{-1}\,\frac{\partial P}{\partial\theta_j}.$$
   Report $\sigma(f_{\rm NL})$ marginalized over $(b_1, s_{n0}, \Sigma_s)$, and also with $b_1$ fixed.
5. Run configurations:
   - QSO, FKP weights, $z_{\rm eff}=1.651$, $b_1=2.44$, $p=1.6$, $0.003<k<0.08$
   - QSO, OQE weights, $z_{\rm eff}=2.082$, $b_1=3.08$, $p=1.6$ (use $V_{\rm eff}$ and $\bar n$ appropriate to the OQE-weighted sample; approximate by evaluating at $z_{\rm eff}$)
   - LRG, FKP, $z_{\rm eff}=0.733$, $b_1=2.06$, $p=1.0$, $0.006<k<0.08$
   - Combined QSO+LRG (independent, as in the paper)
6. Also compute $\sigma(b_\phi f_{\rm NL})$ (assumption-free) for comparison with Table 4 second block.
7. Plot $\sigma(f_{\rm NL})$ vs $k_{\min}$ and vs $k_{\max}$ to compare with their Fig. 8.

**Validation targets** (clean EZmock numbers, Tables 4–5): QSO OQE $\sigma \approx 10$–13; LRG FKP $\approx 11$–18;
combined OQE $\approx 8$–10.5. Fisher is symmetric; aim within $\sim20\%$ of the average of the asymmetric bounds.
If far off, check: $V_{\rm eff}$, $\bar nP$ at the peak, whether $P_2$ is included, $\Delta k$ binning, $\Sigma_s$ damping.

**Optional refinements** (only if the first pass disagrees): window-matrix convolution via pypower
on the DR1 randoms; redshift-shell integration instead of a single z_eff; OQE weighting applied
explicitly in V_eff.

### 5.5 Part B — Field-level Fisher

**Setup**
- Box: $L = 10{,}000\,h^{-1}$Mpc (QSOs at $z=3.1$ are at $\chi \approx 4{,}600\,h^{-1}$Mpc). Observer at centre.
- Grids: $80^3$ ($\Delta L=125$, $k_{\max}=0.05$) and $160^3$ ($\Delta L=62.5$, $k_{\max}=0.1$). Report both.
- Redshift shells: QSO 0.8–1.3, 1.3–2.1, 2.1–3.1 (match DESI imaging bins); LRG 0.4–0.6, 0.6–0.8, 0.8–1.1.
  Each shell has its own $b_1$, $b_\phi$, $\bar n$, $f$, $D$.
- Bias model: Euclid Eq. 4 ($b_1$, $b_2$, $b_K$, $b_\phi f_{\rm NL}\phi_g$, $b_{\phi\delta} f_{\rm NL}\delta\phi_g$), RSD via Kaiser at the field level.

**Method**

Model: $d(x) = M(\varepsilon, \theta)(x) + n(x)$, with $\varepsilon$ the white-noise field (unit Gaussian prior), $\theta$ the global
parameters ($f_{\rm NL}$, per-shell $b_1$, $b_2$, $b_K$, $s_n$; $b_\phi$, $b_{\phi\delta}$ either fixed by UMF or sampled with 40% prior),
and $n$ Gaussian with variance $\sigma^2(x) = \langle N(x)\rangle$ ($\sigma^2 \to \infty$, i.e. zero weight, outside the footprint).

Joint Fisher at the fiducial point: $F = J^{T} N^{-1} J + P$, with $J = \partial M/\partial(\varepsilon, \theta)$, $N^{-1} = {\rm diag}(1/\sigma^2)$,
$P$ = identity on the $\varepsilon$ block (white-noise prior) and the $\theta$-prior on the $\theta$ block.
Marginalize $\varepsilon$ by the Schur complement:

$$F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon} F_{\varepsilon\varepsilon}^{-1} F_{\varepsilon\theta},\qquad \sigma(f_{\rm NL}) = \sqrt{\left(F_{\rm marg}^{-1}\right)_{f_{\rm NL} f_{\rm NL}}}$$

$F_{\varepsilon\varepsilon}$ is $(N_\varepsilon \times N_\varepsilon)$ with $N_\varepsilon = 80^3$ or $160^3$ and is never formed. Everything is done with operators:

- `jvp_eps(v)` $= (\partial M/\partial\varepsilon)\,v$ (jax.jvp) $\to$ field
- `vjp_eps(w)` $= (\partial M/\partial\varepsilon)^{T} w$ (jax.vjp) $\to$ $\varepsilon$-vector
- `jac_theta` $= \partial M/\partial\theta_i$ for each $i$ (jax.jacfwd or $n_\theta$ jvps) $\to$ $n_\theta$ fields

Steps:
1. $F_{\theta\theta}[i,j] = \sum_x \partial_i M(x)\,\partial_j M(x)/\sigma^2(x)$ (+ $\theta$-prior).
2. $F_{\theta\varepsilon}$ row $i$ = `vjp_eps`$(\partial_i M/\sigma^2)$ ($n_\theta$ VJPs).
3. For each $i$, solve $(J^{T}N^{-1}J + I)\,v_i = F_{\varepsilon\theta}[:, i]$ by conjugate gradient
   (jax.scipy.sparse.linalg.cg), ${\rm matvec}(v) = $ `vjp_eps`$($`jvp_eps`$(v)/\sigma^2) + v$.
   Well conditioned; expect 50–200 iterations. Each iteration = one jvp + one vjp through 2LPT.
4. $F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon}\cdot[v_1 \ldots v_{n_\theta}]$; invert; read off $\sigma(f_{\rm NL})$.

Cost: $n_\theta \times \sim100 \times$ (jvp+vjp) $\approx$ a few thousand 2LPT forward/adjoint passes at $80^3$ $\to$ minutes on a
GPU, < 1 h on CPU. This is the Laplace approximation at the truth and includes the nonlinear
couplings present in the linearized model ($f_{\rm NL}\phi^2$, $b_{\phi\delta}\delta\phi$, 2LPT mode coupling). It depends on the
fiducial $\varepsilon$ realization: average $F_{\rm marg}$ over $\sim5$ realizations and report the scatter.

Closed-form cross-checks (must be implemented; they validate the operator code):
- **Tier 1 (Gaussian, two-point)**: set 2LPT $\to$ linear, drop $b_2$, $b_K$, $b_{\phi\delta}$. The Schur result must match
  $$F_{ij} = \sum_{\rm modes} \tfrac12\,\frac{\partial\ln\tilde P(k,\mu)}{\partial\theta_i}\,\frac{\partial\ln\tilde P(k,\mu)}{\partial\theta_j},\qquad \tilde P = P_g(k,\mu) + 1/\bar n,$$
  summed over every independent $k$-mode in the survey volume, per redshift shell. This is Part A without multipole
  compression and with shell-by-shell evolution of $(b_1 - p)$; expect only $\sim10$–20% tighter than Part A.
- **Tier 2 (tree-level bispectrum)**: add
  $$F_{ij} \mathrel{+}= \sum_{\rm triangles} \frac{\partial B_g/\partial\theta_i\;\partial B_g/\partial\theta_j}{{\rm Var}(B_g)},$$
  with $B_g$ = primordial local bispectrum (through $T_{\Phi\to\delta}$) + gravitational $F_2$ term ($b_1$, $b_2$, $b_K$) + $b_\phi/b_{\phi\delta}$
  cross terms, ${\rm Var}(B_g) \approx s\,V\,\tilde P(k_1)\tilde P(k_2)\tilde P(k_3)/N_{\rm tri}$ (RSD per Karagiannis et al. 2018).
  Tier 1 + Tier 2 is an approximation to the Schur result, not an independent target; a disagreement
  $> \sim30\%$ at $k_{\max} = 0.05$ indicates a bug in one of them.

Implementation notes:
- Stay in $\varepsilon$ (white noise), not $\delta_L$: the prior block is then the identity and CG converges fast.
  Parametrizing by $\delta_L$ makes the prior $P_{\rm lin}(k)^{-1}$ and the system ill-conditioned.
- Voxels outside the footprint / radial range: weight $1/\sigma^2 = 0$ exactly, never $\sigma^2 = 0$.
- $\sigma^2(x)$ spans $\sim3$ orders of magnitude across the QSO volume; check the CG tolerance against the
  smallest-weight shells.
- Build everything per redshift shell; $f_{\rm NL}$ is shared across shells, all bias/noise parameters are per shell.

Report $\sigma(f_{\rm NL})$: QSO only, LRG only, combined; $k_{\max} = 0.05$ ($80^3$) and 0.1 ($160^3$);
$p = 1.6$ and 1.0 for QSOs; $b_\phi$ fixed and sampled (40% prior); mean and scatter over $\varepsilon$ realizations.

**Suggested functions (fisher_field.py / fisher_closed.py)**
```
build_model(theta, eps, cfg)     -> predicted counts field   # DISCO-DJ 2LPT + PNG + bias + RSD + selection
jvp_eps, vjp_eps, jac_theta      -> operator wrappers
cg_solve(matvec, rhs, tol=1e-6)  -> jax.scipy.sparse.linalg.cg
schur_fisher(theta_fid, eps_fid, prior_theta, cfg) -> F_marg (n_theta × n_theta)
gaussian_mode_sum(cfg)           -> Tier 1 closed form (must match schur_fisher in the linear limit)
bispectrum_tree(cfg)             -> Tier 2 closed form (cross-check)
```

**Go/no-go**: combined $\sigma(f_{\rm NL}) \lesssim 5$ at $k_{\max}=0.1$ with bias marginalized and $b_\phi$ sampled with 40% prior.
If $\gtrsim 7$: reframe toward DESI Y5 and/or add CMB-lensing cross-correlation.

### 5.6 Outputs of Stage 1
- `results/fisher_pk.json`: $\sigma(f_{\rm NL})$ per configuration vs DESI Tables 4–5.
- `results/fisher_field.json`: $\sigma(f_{\rm NL})$ per shell/tracer/$k_{\max}$/$b_\phi$-treatment.
- Plots: $\sigma$ vs $k_{\min}$, $\sigma$ vs $k_{\max}$, per-mode information density $dF/d\ln k$ (shows where the signal is;
  expect $k\approx0.005$–0.02, $\bar nP\approx0.3$–0.5 for QSOs).
- A short memo (1–2 pages) with the go/no-go decision and the conventions (box, shells, binning)
  to carry into Stage 2.

### 5.7 Suggested repo layout
```
desi_fnl_fli/
  STAGE1_PLAN.md          (this file)
  env/requirements.txt
  src/
    cosmology.py
    model_pk.py
    geometry.py
    fisher_pk.py
    forward_model.py        (DISCO-DJ 2LPT + PNG + bias + RSD + selection; shared with Stage 2)
    fisher_field.py         (Schur-complement Fisher via jvp/vjp + CG)
    fisher_closed.py        (Tier 1 mode sum, Tier 2 tree-level bispectrum)
  configs/*.yaml           (tracer, z-range, b1, p, k-range, weights)
  scripts/run_fisher_pk.py, run_fisher_field.py
  results/
  notebooks/
```

### 5.8 Things to watch
- Units: CLASS returns Mpc, DESI quotes $h$/Mpc. $T_{\Phi\to\delta}$ normalization ($\Phi = \tfrac35\mathcal{R}$ convention) changes $b_\phi f_{\rm NL}$ by a constant factor if wrong.
- Sign convention of $f_{\rm NL}$ (LSS vs CMB) — DESI follows the Slosar et al. LSS convention.
- Shot noise at the QSO densities: $\bar nP \approx 0.5$ at the $P(k)$ peak; results are sensitive to $\bar n(z)$ at $z > 2$.
- $\Sigma_s$ is poorly constrained for QSOs (OQE fit returned 0); fix it or put a prior if it destabilizes the Fisher.
- Don't over-invest in the window matrix in Stage 1; $k_{\min}$ cut is enough for the go/no-go.