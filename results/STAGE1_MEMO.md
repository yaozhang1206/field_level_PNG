# Stage 1 memo: information content of DESI DR1 QSO+LRG on local $f_{\rm NL}$

Date: 2026-10-08. Code: this repository (commit TBD). Environment: Delta AI, conda env `pm`, DISCO-DJ 0.0.2, CLASS 3.4.

## 1. Summary and go/no-go

- **Part A (P(k) Fisher) reproduces DESI DR1.** With the public DR1 catalogues (n(z) from the NX column, geometric
  footprint 9,070 deg² from the randoms, completeness-reduced shot noise) the Gaussian $P_0+P_2$ Fisher gives
  $\sigma(f_{\rm NL}) = 15.5$ for QSO FKP (DESI clean mocks: 15.5), 20.0 / 13.1 for LRG with $k_{\min} = 0.006$ / 0.003 (DESI: 14.5),
  and 7.9–8.5 for QSO+LRG in the optimal-weighting limit (DESI OQE: 9.2). Conventions ($T_{\Phi\to\delta}$, $b_\phi$, volumes,
  shot noise) are therefore calibrated.
- **Part B (field level).** Closed-form Gaussian mode sum over the $160^3$ grid ($k_{\rm Ny} = 0.05\,h/{\rm Mpc}$, the "$k_{\max} = 0.1$"
  configuration of the brief) with $f_{\rm NL}$ + $(b_1, b_2, b_K)$ per shell marginalised: $\sigma(f_{\rm NL}) = 5.0$ (QSO+LRG) when all box
  modes down to $k_f = 6\times10^{-4}$ are counted, 7.4 with $k_{\min} = 0.003$. The tree-level bispectrum adds < 3 %.
  The exact Gaussian-marginal Fisher of the DISCO-DJ 2LPT forward model with the real DR1 window gives
  **$\sigma(f_{\rm NL}) = 5.7$ at $80^3$ ($k_{\max}=0.05$) and $4.0 \pm 0.2$ at $160^3$ ($k_{\max}=0.1$)** (QSO+LRG, $b_\phi$ fixed; see §4).
- **Decision: GO (marginal).** The exact Gaussian-marginal forecast at $160^3$ is $\sigma(f_{\rm NL}) = 4.0 \pm 0.2$ with
  $b_\phi$ fixed and $\approx 4.9$ after propagating a 40 % prior on $b_\phi$ ($\times1.22$), i.e. at the $\lesssim 5$ threshold and a
  factor $\approx 2$ below the DESI DR1 $P(k)$ result (9.2). Caveats that Stage 2 must settle: (i) the forecast rests on the model
  linearised in $\varepsilon$; the strictly linear model with the same window gives 8.3 at $80^3$, so 60 % of the Fisher
  information at $80^3$ (and more at $160^3$) comes from the quadratic couplings ($b_{\phi\delta}\,\delta\phi$, $f_{\rm NL}\phi^2$,
  2LPT) entering through the $f_{\rm NL}$-dependence of the covariance, which a Fisher matrix can over-credit; (ii) the
  largest-scale modes dominate, so results depend on the window and on the fiducial realisation at the 10 % level;
  (iii) no systematics. If the Stage-2 MCMC recovers only the two-point information, expect $\sigma \approx 7$–8 and
  the reframing branch of the brief applies.

## 2. Conventions carried into Stage 2

| item | choice |
|---|---|
| cosmology | Planck 2018 ($\Omega_m=0.3153$, $\Omega_b=0.0493$, $h=0.6736$, $A_s=2.1\times10^{-9}$, $n_s=0.9649$, $m_\nu=0.06$ eV); CLASS for $P_{\rm lin}$; DISCO-EB agrees to 1.1 % (0.5 % with massless $\nu$) |
| PNG convention | $\Phi = \tfrac35\zeta$ (LSS/Slosar), $P_\Phi = \tfrac{9}{25}\tfrac{2\pi^2}{k^3}A_s(k/k_p)^{n_s-1}$; $T_{\Phi\to\delta} = \tfrac23 k^2 T(k) D_{\rm md}(z)\,c^2/(\Omega_m H_0^2)$; $\Delta b = b_\phi f_{\rm NL}/T_{\Phi\to\delta}$ |
| bias | $b_\phi = 2\delta_c(b_1 - p)$, $p = 1.6$ (QSO), 1.0 (LRG); $b_1(z) = a(1+z)^2+b$ ($a,b = 0.237, 0.771$ QSO; $0.209, 1.415$ LRG); $b_2$ Lazeyras+16; $b_K = -\tfrac27(b_1-1)$; $b_{\phi\delta}$ Euclid Eq. |
| box / grids | $L = 10{,}000$ Mpc/$h$, observer at centre; $80^3$ ($\Delta L = 125$, $k_{\rm Ny} = 0.025$) and $160^3$ ($\Delta L = 62.5$, $k_{\rm Ny} = 0.05$) |
| shells | QSO 0.8–1.3, 1.3–2.1, 2.1–3.1; LRG 0.4–0.6, 0.6–0.8, 0.8–1.1; one $(a, b_1, b_2, b_K, b_\phi, b_{\phi\delta}, \bar n)$ per shell |
| selection | angular: fractional HEALPix (nside 64) coverage from the QSO randoms; radial: shell $\chi$ range; $\bar n = \bar c\,{\rm NX}(z)$ with $\bar c = A_{\rm eff}/A_{\rm full}$ |
| forward model | $\varepsilon \to \delta_L(k) \to \phi_g = \delta_L/T \to \Phi_{\rm NG} = \phi_g + f_{\rm NL}(\phi_g^2-\langle\phi_g^2\rangle) \to \delta_{\rm NG} \to$ DISCO-DJ 2LPT $\to$ Lagrangian weights $\to$ RSD (radial LOS) $\to$ CIC |
| likelihood | Gaussian, $\sigma^2 = \bar n V_{\rm cell}$ per voxel, zero weight off footprint |
| noise parameter | $s_n$ per shell decouples from the mean-model Fisher at the Gaussian level; not included |

## 3. Part A details

Method: Gaussian covariance of $(P_0, P_2)$ in $\Delta k = 0.001$ bins with the FKP-weighted variance
$$S(k,\mu) = \frac{\int dV\, n^4 w^4 \left(P + 1/n_{\rm obs}\right)^2}{\left(\int dV\, n^2 w^2\right)^2},\qquad w = \frac{1}{1+nP_0},$$
integrated over the DR1 $n(z)$; parameters $(f_{\rm NL}, b_1, s_{n0}, \Sigma_s)$, fiducial $f_{\rm NL} = 0$, $\Sigma_s = 3$ Mpc/$h$. Derivatives by jax autodiff.

| configuration | $\sigma(f_{\rm NL})$ marg. | $b_1$ fixed | DESI clean mocks (Tables 4–5) |
|---|---|---|---|
| QSO FKP ($z_{\rm eff}=1.651$, $b_1=2.44$, $0.003<k<0.08$) | 15.5 | 11.8 | $^{+18}_{-13}$ (mean 15.5) |
| QSO OQE-like single $z$ ($z=2.082$, $b_1=3.08$) | 8.6 | 6.6 | $^{+13}_{-10}$ (mean 11.5) |
| QSO 10 shells, $b_1(z)$ template (optimal-weighting limit) | 9.6 | 7.4 | $^{+13}_{-10}$ (mean 11.5) |
| LRG FKP ($z_{\rm eff}=0.733$, $b_1=2.06$, $k_{\min}=0.006$) | 20.0 | 13.1 | $^{+18}_{-11}$ (mean 14.5) |
| LRG FKP, $k_{\min}=0.003$ | 13.1 | 10.4 | |
| QSO+LRG FKP | 12.3 | | |
| QSO+LRG shells | 8.5 | | $^{+10.5}_{-7.9}$ (mean 9.2) |

Notes: the LRG number is sensitive to $k_{\min}$ (0.003 vs 0.006 changes $\sigma$ by $1.5\times$) and the DESI LRG posterior is
strongly asymmetric; the QSO numbers match to 3 %. The 'full' treatment (sample variance from the geometric
footprint, shot noise from the completeness-reduced density) was needed: with the effective area alone LRG gave 24.
Figures: `test/figures/fisher_pk_kcuts_full.png`, `fisher_pk_info_density_full.png`. Details: `test/notes/fisher_pk_full.md`.

## 4. Part B details

### 4.1 Closed forms (`test/notes/fisher_closed.md`)

| grid | tracers | Tier 1 ($P$, $k_{\min} = k_f$) | Tier 1 + Tier 2 | Tier 1, $k_{\min} = 0.003$ |
|---|---|---|---|---|
| $80^3$ | QSO / LRG / both | 8.4 / 7.1 / 5.4 | 8.4 / 7.1 / 5.4 | 11.5 / 14.1 / 8.9 |
| $160^3$ | QSO / LRG / both | 7.5 / 6.7 / 5.0 | 7.4 / 6.6 / 4.9 | 9.5 / 11.9 / 7.4 |

The PNG information is dominated by the largest scales (per-$\ln k$ information $\propto k^{-1}$ in the sample-variance
regime), so the closed form depends strongly on $k_{\min}$: the $k_f$ value counts modes with $\lambda \approx 10$ Gpc/$h$ that the
$\approx 5$ Gpc/$h$ survey cannot constrain, the 0.003 value is conservative. The tree-level (real-space) bispectrum adds
< 3 % at $k \le 0.05$.

### 4.2 Field-level Fisher with the DISCO-DJ 2LPT forward model (`test/notes/fisher_field.md`)

Both estimators, DR1 window, 6 shells, $(b_1,b_2,b_K)$ per shell marginalised, $b_\phi$ fixed (UMF), fiducial $f_{\rm NL}=0$:

| grid | tracers | exact Gaussian-marginal (Hutchinson) | Schur / profile (mean of 4 realisations) | closed Tier 1 ($k_{\min}=k_f$ / 0.003) |
|---|---|---|---|---|
| $80^3$ ($k_{\rm Ny}=0.025$) | QSO+LRG | **5.67** (5.40 ± 0.31, 6.00 ± 0.34; two realisations) | 3.9 ± 0.4 | 5.4 / 8.9 |
| $80^3$ | QSO | 8.99 ± 0.12 | 4.7 ± 0.6 | 8.4 / 11.5 |
| $80^3$ | LRG | 9.42 ± 0.41 | 7.3 ± 0.8 | 7.1 / 14.1 |
| $80^3$, QSO $p=1.0$ | QSO+LRG | 4.36 ± 0.23 | 2.5 | 5.4 / 8.9 |
| $160^3$ ($k_{\rm Ny}=0.05$) | QSO+LRG | **4.00 ± 0.18** (12 probes, 1 realisation) | 2.8 (1 realisation) | 5.0 / 7.4 |
| $80^3$, strictly linear model, DR1 window | QSO+LRG | 8.28 ± 0.21 | 3.4 | 5.4 / 8.9 |

(± = jackknife over Hutchinson probes; the two $80^3$ realisations differ by 10 %.) The $80^3$ single-tracer and $p=1.0$
runs used an unpreconditioned CG for $C^{-1}$; a re-run of the QSO+LRG case with the noise-preconditioned solver gave
5.32 ± 0.65 vs 5.40, so they stand. The $160^3$ run uses the preconditioned solver (CG residuals $\le 10^{-5}$); an
earlier unpreconditioned $160^3$ attempt gave 2.6 and was discarded. Cost on one GH200 after compilation: 10 s (80³)
/ 100 s (160³) per probe, 8 s per Schur solve at 80³; memory 6 / 17 GB.

**Linearisation budget.** Same realisation, same window, 80³: strictly linear model 8.28 → linearised 2LPT model 5.32
→ 160³ linearised 2LPT 4.00. The gain from 8.3 to 5.3 is the information the quadratic terms contribute through
$\partial C/\partial f_{\rm NL}$; the tree-level real-space bispectrum (Tier 2, $\Delta k = 0.002$) accounts for < 3 % of it,
so Tier 2 as implemented under-resolves the squeezed configurations ($k_L \sim 10^{-3}$, where $1/T \propto k^{-2}$).


### 4.3 Schur (profile) vs exact marginal: validation at 16³ (`test/notes/schur_dense16.md`)

Brute-force dense computation on a periodic $16^3$ box with the strictly linear model:

| | LRG ($\bar n = 3\times10^{-4}$) | QSO ($\bar n = 2.5\times10^{-5}$) |
|---|---|---|
| Schur complement, CG operators / dense | 0.777 / 0.777 | 14.93 / 14.93 |
| exact Gaussian-marginal $\tfrac12{\rm Tr}(C^{-1}\partial_iC\,C^{-1}\partial_jC)$, dense / Hutchinson (16 probes) | 0.614 / 0.617 | 26.73 / 26.66 |
| Tier 1 mode sum, exact / Laplace-weighted $\partial_iP\,\partial_jP/(2P(P+N))$ | 0.614 / 0.729 | 26.73 / 14.55 |

So (i) the jvp/vjp/CG operator code is exact, (ii) the Tier 1 mode sum equals the exact Gaussian-marginal
Fisher, (iii) the Schur complement of the joint $(\varepsilon, \theta)$ Fisher is the *profile* curvature, which differs from the
marginal by $(P+N)/(2P)$ per mode: pessimistic where $\bar nP \gg 1$ (LRG, $\times1.27$ in $\sigma$ here) and **optimistic by $\times1.8$ in $\sigma$
($\times3.3$ in $F$) for the shot-noise-dominated QSOs**. The Schur method of the brief therefore cannot be used as the
forecast; the exact Gaussian-marginal Fisher of the model linearised around the fiducial $\varepsilon$ (implemented as
`FieldFisher.fisher_marginal`, cost $\approx (n_\theta+1)\times n_{\rm probes}$ CG solves) is used instead, with the Schur numbers
reported for reference only.

At fiducial $f_{\rm NL} = 0$ any Fisher matrix is blind to the $b_\phi$ prior ($b_\phi$ enters only as $b_\phi f_{\rm NL}$); propagating
$f_{\rm NL} = A/b_\phi$ with a 40 % Gaussian prior on $b_\phi$ inflates $\sigma$ by $\approx \sqrt{1 + 3\cdot0.4^2} = 1.22$.

### 4.4 The three field-level Fisher estimators: what each one is

All three start from the same forward model $d = M(\varepsilon,\theta) + n$ with $\varepsilon\sim\mathcal N(0,I)$
(white-noise field) and $n\sim\mathcal N(0,N)$, $N = {\rm diag}(\sigma^2)$, and from the Jacobians at the
fiducial point $J_\varepsilon = \partial M/\partial\varepsilon$, $J_\theta = \partial M/\partial\theta$.

**(a) Joint Fisher and its Schur complement (the brief's method).** The Fisher matrix of the *joint*
posterior $p(\varepsilon,\theta\,|\,d)$ is
$$F = \begin{pmatrix} F_{\theta\theta} & F_{\theta\varepsilon} \\ F_{\varepsilon\theta} & F_{\varepsilon\varepsilon}\end{pmatrix},\qquad
F_{\theta\theta} = J_\theta^T N^{-1} J_\theta + P_\theta,\quad
F_{\theta\varepsilon} = J_\theta^T N^{-1} J_\varepsilon,\quad
F_{\varepsilon\varepsilon} = J_\varepsilon^T N^{-1} J_\varepsilon + I .$$
$F_{\theta\theta}$ alone is the curvature with the field held fixed at $\varepsilon_{\rm fid}$ ("unmarginalised":
bias and initial conditions known); it ignores the degeneracy between $\theta$ and $\varepsilon$ and is far too
optimistic. The Schur complement
$$F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta},
\qquad (F^{-1})_{\theta\theta} = F_{\rm marg}^{-1},$$
removes that degeneracy. It is the exact marginalisation *of a Gaussian* in $(\varepsilon,\theta)$, and it is
computed without approximation (CG operators agree with the dense $16^3$ matrix to all digits). But the joint
posterior is not Gaussian, because $\theta$ multiplies $\varepsilon$ in the model: the Schur complement equals the
curvature of the *profile* likelihood $\max_\varepsilon \ln[p(d|\varepsilon,\theta)p(\varepsilon)]$, which omits the
$\theta$-dependent volume term $\tfrac12\ln\det F_{\varepsilon\varepsilon}(\theta)$ of the true marginal. Per Gaussian
mode with signal $P$ and noise $N$,
$$F^{\rm Schur}_{ij} = \frac{\partial_iP\,\partial_jP}{2P(P+N)}\quad\text{versus}\quad
F^{\rm exact}_{ij} = \frac{\partial_iP\,\partial_jP}{(P+N)^2},$$
a ratio $(P+N)/(2P)$: too pessimistic ($\times\sqrt2$ in $\sigma$) for $\bar nP\gg1$ (LRG) and too optimistic for
$\bar nP\ll1$ (QSO, $\sigma$ under-estimated by $1.8\times$ at $16^3$). It also scatters strongly between
$\varepsilon$ realisations because the few largest modes dominate the $f_{\rm NL}$ information. It is reported for
reference only.

**(b) Exact Gaussian-marginal Fisher of the linearised model.** Linearising $M \approx M_0 + J_\varepsilon\varepsilon$
and integrating $\varepsilon$ out analytically makes the data Gaussian with covariance
$C(\theta) = J_\varepsilon J_\varepsilon^T + N$, whose Fisher matrix is
$$F_{ij} = \tfrac12\,{\rm Tr}\!\left[C^{-1}\partial_iC\,C^{-1}\partial_jC\right],\qquad
\partial_iC = (\partial_iJ_\varepsilon)J_\varepsilon^T + J_\varepsilon(\partial_iJ_\varepsilon)^T .$$
This is the true information of the marginal posterior $p(\theta|d)$ for a linear model (no profile shortcut, no
missing volume term); in a periodic box it reduces mode by mode to the Tier 1 sum $\partial_iP\,\partial_jP/(P+N)^2$,
and with the DR1 window the mask and $n(z)$ enter through $J_\varepsilon$ and $N$. Its only approximation for the
2LPT model is the linearisation in $\varepsilon$ about the fiducial field, which keeps the $\theta$-dependence of all
couplings but drops the non-Gaussian response of the field itself. It depends on $\varepsilon_{\rm fid}$ only through
that linearisation point, so one or two realisations suffice.

Two evaluations of (b) are used:
- *Dense marginal* ($16^3$ only): $J_\varepsilon$ is built explicitly as a $4096\times4096$ matrix, $C$ is inverted,
  $\partial_iJ_\varepsilon$ comes from a second forward-mode pass over $\theta$, and the trace is computed directly.
  Ground truth, exact to floating point.
- *Hutchinson marginal* ($80^3$, $160^3$): $C$ is never formed; the trace is estimated with Rademacher probes $z$,
  $$F_{ij} \approx \tfrac12\big\langle (C^{-1}z)^T\,\partial_iC\,(C^{-1}\partial_jC\,z)\big\rangle_z ,$$
  with $C^{-1}$ by conjugate gradient and $\partial_iC$ applied through nested forward/reverse-mode operators.
  Cost $\approx (n_\theta+1)\times n_{\rm probes}$ CG solves. At $16^3$ with 16 probes it reproduces the dense
  value to $<1\%$ (0.617 vs 0.614 for LRG, 26.66 vs 26.73 for QSO).

**(c) Closed forms (Tier 1, Tier 2).** Analytic mode sums over the grid with the same shells, volumes and
$n(z)$ but no survey window; Tier 1 is (b) in the linear, window-free limit and Tier 2 adds the tree-level
bispectrum. They bracket the window effect and validate the operator code.

**Which number is the forecast.** The Hutchinson marginal (b) with the DR1 window. The Schur values (a) are listed
for comparison with the brief, and the closed forms (c) for the dependence on $k_{\min}$.

## 5. Where the signal is

Per-mode information density peaks at $k \approx 0.003$–$0.01\,h/{\rm Mpc}$ for both tracers (`fisher_pk_info_density_full.png`),
i.e. $\bar nP \approx 0.4$–0.7 for QSOs at $z > 2$ and the sample-variance regime for LRGs; the high-$z$ QSO shell (2.1–3.1) and the
0.8–1.1 LRG shell carry the most $f_{\rm NL}$ information per shell ($\sigma \approx 10$ each, Tier 1).

## 6. Outputs

- `results/fisher_pk_full.json`, `results/fisher_closed.json`, `results/fisher_field_*.json`, `results/fisher_field.json`
- notes and figures for every test in `test/notes`, `test/figures`


---

## 9. Stage 2 progress (2026-10-11)

- Sampler: BlackJAX NUTS on $[\varepsilon, \theta_{\rm w}]$ (identity mass for $\varepsilon$, Fisher-whitened $\theta$). Validated on an
  exactly Gaussian 4096-dimensional target (means and variances to $\lesssim 1$ %); 6-mock $f_{\rm NL}$ widths average 0.77 ± 0.09
  vs Fisher 0.64 with calibrated posterior means (`test/notes/sampler16_gauss.md`).
- **Is the quadratic-coupling information real?** Small periodic boxes with the production cell size, same mock under the
  linear and the full 2LPT model (`test/notes/stage2_smallbox_summary.md`): posterior $\sigma$ ratio linear/full = 1.28–1.42,
  Fisher-predicted 1.30–1.40. Yes, at $\gtrsim 85$ % of the Fisher prediction. The Stage 1 caveat is largely retired; the DR1
  $80^3$ chains (queued) give the direct number.
- Tier 2 (tree-level bispectrum) is converged in binning and still adds < 3 %: the tree-level real-space bispectrum with
  Gaussian covariance is simply not where this information is; it sits in the modulation of small-scale power by the
  large-scale field, which the field-level posterior captures.
- Cost: NUTS needs 31 leapfrog steps per sample on the linear model and $\approx 255$ on the full model; $\tau_{\rm int}(f_{\rm NL})$
  = 2–8 samples. At $80^3$ this projects to $\sim 10$–25 s per sample on one GH200.
- **DR1 configuration, $80^3$, same mock under both models** (`results/chain_nuts_res80_*.json`): linear model
  $\sigma(f_{\rm NL}) = 7.8 \pm 0.3$ (3000 samples, ESS 269; Fisher 8.28); full 2LPT model $8.0 \pm 0.8$ (500 samples, ESS 57;
  linearised Fisher 5.3). Unlike the periodic boxes, the full model shows **no gain** over the two-point information at
  DR1; a width of 6 is disfavoured at $\approx 2.5\sigma$ by the short chain. Interpretation: the quadratic-coupling
  information is a modulation of small-scale power by the large-scale potential, which the posterior can only use if
  those large-scale modes are themselves well constrained; the DR1 window removes exactly those modes, while the
  linearised Fisher holds them fixed at the fiducial. Long chains (2500 samples, second mock, $160^3$) are queued to
  confirm. If confirmed, the realistic DR1 forecast is $\sigma \approx 7$–8 and the brief's reframing branch applies.
