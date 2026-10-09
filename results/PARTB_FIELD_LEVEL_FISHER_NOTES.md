# Part B: field-level Fisher analysis for DESI DR1 $f_{\rm NL}$ — detailed notes

Written 2026-10-09 for the Stage 1 record. Code: `src/forward_model.py`, `src/fisher_field.py`, `src/fisher_closed.py`,
`scripts/run_fisher_marginal.py`, `scripts/run_fisher_field.py`; validation in `test/`. All numbers quoted are in
`results/fisher_field_*.json` and `test/notes/`.

---

## 1. The question and the data model

**Question.** How well can a BORG-style field-level analysis of DESI DR1 QSOs and LRGs constrain local $f_{\rm NL}$,
with no systematics, bias marginalised, on a $10\,h^{-1}$Gpc box at $80^3$ ($k_{\rm Ny} = 0.025\,h/{\rm Mpc}$) and $160^3$
($k_{\rm Ny} = 0.05$)? The brief's go/no-go is $\sigma(f_{\rm NL}) \lesssim 5$ in the $160^3$ configuration.

**Data model.** For each tracer $t \in \{{\rm QSO}, {\rm LRG}\}$ the data are galaxy counts per voxel,
$$d_t(\mathbf{x}) = M_t(\varepsilon, \theta)(\mathbf{x}) + n_t(\mathbf{x}),\qquad
n_t \sim \mathcal{N}\!\left(0, \sigma_t^2(\mathbf{x})\right),\qquad \sigma_t^2 = \langle N_t(\mathbf{x})\rangle = \bar n_t(\mathbf{x})\,V_{\rm cell},$$
with $1/\sigma^2 = 0$ outside the footprint. The latent variables are the white-noise field $\varepsilon$ (one unit
Gaussian per Lagrangian grid cell, prior $\varepsilon \sim \mathcal{N}(0, I)$) and the global parameters
$$\theta = \big(f_{\rm NL};\ \{b_1^s, b_2^s, b_K^s\}_{s=1}^{6}\ [;\ \{b_\phi^s, b_{\phi\delta}^s\}]\big),$$
one bias set per redshift shell $s$ (QSO 0.8–1.3, 1.3–2.1, 2.1–3.1; LRG 0.4–0.6, 0.6–0.8, 0.8–1.1), $n_\theta = 19$
with $b_\phi, b_{\phi\delta}$ fixed by the universality relations, 31 when sampled.

Why Gaussian noise rather than Poisson: for parameters that enter only the mean, the Poisson Fisher matrix
$\sum_x \partial_i\mu\,\partial_j\mu/\mu$ is identical to the Gaussian one with $\sigma^2 = \mu$; the two differ only in
posterior shape. The noise variance is the unperturbed mean $\bar nV_{\rm cell}$, not $\mu(1+\delta_g)$; the
difference averages to $\mathcal{O}(\sigma_\delta^2)$ on the cell scale, a few percent in $F$.

Why both tracers share one $\varepsilon$: that is what the field-level likelihood does (multi-tracer), and the Fisher
for "QSO only" or "LRG only" is obtained by giving the other tracer's voxels zero weight, not by slicing the joint
matrix (slicing would keep the other tracer's $f_{\rm NL}$ information through the shared $F_{ff}$ entry; this was a
bug caught in the closed-form run).

---

## 2. The forward model $M(\varepsilon, \theta)$

### 2.1 Linear fields and PNG

$$\delta_L(\mathbf{k}) = \sqrt{P_{\rm lin}(k, z{=}0)\,(N_{\rm grid}/L)^3}\;\tilde\varepsilon(\mathbf{k}),\qquad
\phi_g(\mathbf{k}) = \frac{\delta_L(\mathbf{k})}{T_{\Phi\to\delta}(k, 0)},$$
$$\Phi_{\rm NG}(\mathbf{x}) = \phi_g + f_{\rm NL}\left(\phi_g^2 - \langle\phi_g^2\rangle\right),\qquad
\delta_{\rm NG}(\mathbf{k}) = T_{\Phi\to\delta}(k, 0)\,\Phi_{\rm NG}(\mathbf{k}),$$
with the Part A conventions $T_{\Phi\to\delta}(k,z) = \tfrac23 k^2 T(k) D_{\rm md}(z) c^2/(\Omega_m H_0^2)$ and
$P_\Phi = \tfrac{9}{25}\tfrac{2\pi^2}{k^3}A_s (k/k_p)^{n_s-1}$, so that $\Delta b = b_\phi f_{\rm NL}/T_{\Phi\to\delta}$ matches
Chaussidon et al. The normalisation $(N_{\rm grid}/L)^3$ is DISCO-DJ's GRF convention; it was checked by measuring
$P(\delta_L)$ on the grid (ratio to $P_{\rm lin}$: 0.98 at $k < 0.005$, the rest is the CIC window).

Why parametrise by $\varepsilon$ and not $\delta_L$: the prior block of the Fisher matrix is then the identity, which
makes the conjugate-gradient systems below well conditioned (eigenvalues $1 + \bar nP$). Parametrising by $\delta_L$
would give a prior $P_{\rm lin}^{-1}(k)$ spanning many orders of magnitude.

### 2.2 Displacement and RSD (DISCO-DJ)

The PNG linear density is handed to DISCO-DJ's nLPT solver as the potential $\hat\phi = -\delta_{\rm NG}/k^2$, which returns
$\psi_1, \psi_2$ (2LPT). For shell $s$ with scale factor $a_s$ (centre of the shell), growth $D_s$ and rate $f_s$:
$$\mathbf{x}(\mathbf{q}) = \mathbf{q} + D_s\psi_1 + D_s^2\psi_2,\qquad
\mathbf{x}_s = \mathbf{x} + f_s\left[(D_s\psi_1 + 2D_s^2\psi_2)\cdot\hat{\mathbf{r}}\right]\hat{\mathbf{r}},$$
with $\hat{\mathbf{r}}$ the line of sight from the observer at the box centre. The velocity displacement follows from
$\dot{\mathbf{x}} = \dot D\,\partial_D\mathbf{x}$ and $\dot D = fHD$, so $\Delta\mathbf{x}_{\rm RSD} = f\,D\,\partial_D\psi$; the factor
$n D^n$ for order $n$ is the EdS approximation that DISCO-DJ itself uses. Checked against Kaiser at the mode level
(test `forward_model_res80`: $P_s(k,\mu)/P_r(k,\mu)$ = 1.03 / 1.20 / 1.56 vs 1.03 / 1.21 / 1.62 in three $\mu$ bins).

Why 2LPT and not ZA: the brief's decision; at these grids both are cheap. Why no CIC deconvolution: the data would be
gridded the same way, so the window is part of the model, not an error.

### 2.3 Bias: Lagrangian weights

Each particle carries a weight evaluated at its Lagrangian position,
$$w(\mathbf{q}) = 1 + b_1^L\,\delta + \tfrac12 b_2^L\left(\delta^2 - \langle\delta^2\rangle\right) + b_K^L\left(K^2 - \langle K^2\rangle\right)
+ b_\phi f_{\rm NL}\,\phi_g + b_{\phi\delta}^L f_{\rm NL}\,\delta\,\phi_g,\qquad \delta \equiv D_s\,\delta_L(\mathbf{q}),$$
$K_{ij} = (\partial_i\partial_j/\nabla^2 - \delta_{ij}/3)\delta$, and the galaxy field is the CIC deposit of $w$ at $\mathbf{x}_s$.
The Eulerian parameters of the Euclid paper are converted on input:
$$b_1 = 1 + b_1^L,\quad b_2 = b_2^L + \tfrac{8}{21}b_1^L,\quad b_K = b_K^L - \tfrac27 b_1^L,\quad b_{\phi\delta} = b_{\phi\delta}^L + b_\phi .$$
Fiducial values per shell: $b_1(z) = a(1+z)^2 + b$ from DESI Table 2, $b_2$ from Lazeyras et al. 2016,
$b_K = -\tfrac27(b_1-1)$ (i.e. $b_K^L = 0$), $b_\phi = 2\delta_c(b_1 - p)$ with $p = 1.6$ (QSO) / 1.0 (LRG),
$b_{\phi\delta} = b_\phi - b_1 + 1 + \delta_c[b_2 - \tfrac{8}{21}(b_1 - 1)]$.

Why Lagrangian rather than Eulerian weights: painting Eulerian-biased densities in redshift space gives
$b_1 f\mu^2$ instead of $f\mu^2$ in the Kaiser term; Lagrangian weights advected with the particles give exactly
$\delta_g = (b_1 + b_\phi f_{\rm NL}/T + f\mu^2)\,\delta$ at linear order, and $\phi_g$ needs no growth factor because it is
primordial. (An early version scaled $\delta_L$ without $D_s$ in the weights; the mode-level test caught it.)

### 2.4 Selection

Expected counts per voxel, $\bar n_t(\mathbf{x})V_{\rm cell}$, are built once as static arrays: radial range of the shell in
$\chi$, $\bar n = \bar c\,{\rm NX}(z)$ (observed density = completeness $\bar c = A_{\rm eff}/A_{\rm full}$ times the
catalogue's completeness-corrected NX, as in Part A's "full" treatment), times the fractional HEALPix (nside 64)
coverage from the QSO randoms. Footprint: 9,070 deg$^2$. Voxels in the footprint: 46,090 (QSO) and 7,647 (LRG) at
$80^3$, 8× that at $160^3$.

### 2.5 Autodiff through DISCO-DJ

DISCO-DJ's particle scatter has either a `custom_vjp` (reverse mode) or a `custom_jvp` (forward mode,
`requires_jacfwd=True`) rule, and JAX cannot apply the other mode through either. The Fisher code therefore keeps two
views of the same model: `fm_fwd` for $J_\varepsilon v$, $\partial_\theta M$ and $\partial_\theta(J_\varepsilon v)$;
`fm_bwd` for $J_\varepsilon^T w$ and $\partial_\theta(J_\varepsilon^T w)$. The adjoint identity
$\langle w, J v\rangle = \langle J^T w, v\rangle$ was checked to 14 digits across the two views.

---

## 3. Three Fisher estimators

### 3.1 Schur complement of the joint Fisher (the method in the brief)

With $J = \partial M/\partial(\varepsilon, \theta)$, $N^{-1} = {\rm diag}(1/\sigma^2)$ and prior $P$ (identity on $\varepsilon$,
$\theta$-prior on $\theta$), the joint Fisher is $F = J^T N^{-1} J + P$ and
$$F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta},\qquad
F_{\varepsilon\varepsilon} = J_\varepsilon^T N^{-1} J_\varepsilon + I .$$
$F_{\varepsilon\varepsilon}$ ($N_{\rm grid}^3$ square) is never formed: for each $i$ the vector
$v_i = F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta_i}$ is obtained by CG with the matvec
$v \mapsto J_\varepsilon^T\!\left(N^{-1}J_\varepsilon v\right) + v$ (one jvp, one vjp per iteration; $\sim$100 iterations;
19 solves; 8 s at $80^3$, 160 s at $160^3$ after compilation).

**What it is.** For a Gaussian likelihood with a parameter-dependent mean, $J^TN^{-1}J$ is the exact expected Hessian,
so this is the curvature of the joint $(\varepsilon,\theta)$ log-posterior at the truth, and its Schur complement is the
curvature of the *profile* likelihood in $\theta$ (maximised over $\varepsilon$). It is not the curvature of the marginal.

**Per-mode value for a linear model.** Take one complex Fourier mode with $M = \sqrt{P_s(\theta)}\,\varepsilon$, noise $N$.
Then $u = \partial_\theta M = (\partial_\theta\sqrt{P_s})\varepsilon$, and the Schur complement reduces to
$u^T(P_s + N)^{-1}u$, whose expectation over $\varepsilon$ is
$$F^{\rm Schur} = \frac{(\partial_i P_s)(\partial_j P_s)}{2P_s\,(P_s + N)}\quad\text{per independent complex mode}.$$
The exact Gaussian-field Fisher for the same mode (parameters in the covariance) is
$$F^{\rm exact} = \frac{(\partial_i P_s)(\partial_j P_s)}{(P_s + N)^2},\qquad
\frac{F^{\rm Schur}}{F^{\rm exact}} = \frac{P_s + N}{2P_s}.$$
So the Schur estimate is optimistic where $\bar nP \ll 1$ (QSOs: $F$ too large by $\approx N/2P$) and pessimistic by a
factor 2 in $F$ where $\bar nP \gg 1$ (LRGs). The missing piece is the $\tfrac12\ln\det F_{\varepsilon\varepsilon}(\theta)$
"volume" term that marginalisation produces and a profile ignores. The brief's statement that the Schur result must
match the Tier 1 mode sum in the linear limit is therefore false except at $\bar nP = 1$; this was confirmed
numerically (§4).

### 3.2 Exact Gaussian-marginal Fisher of the linearised model (the estimator used for the forecast)

Linearise the model in $\varepsilon$ around the fiducial field: $M \approx M_0 + J_\varepsilon(\theta)\,\delta\varepsilon$. Marginalising
the Gaussian $\delta\varepsilon$ gives data with covariance $C(\theta) = J_\varepsilon J_\varepsilon^T + N$ (data space, $n_{\rm data}\times n_{\rm data}$),
and the Fisher matrix for parameters in a Gaussian covariance is
$$F_{ij} = \tfrac12\,{\rm Tr}\!\left[C^{-1}\partial_iC\;C^{-1}\partial_jC\right],\qquad
\partial_iC = (\partial_i J_\varepsilon)J_\varepsilon^T + J_\varepsilon(\partial_i J_\varepsilon)^T .$$
For a strictly linear model this is the exact marginal Fisher (it contains the log-det term). For the 2LPT model it
is exact for the model linearised at the fiducial $\varepsilon$; the information it credits to $\theta$ enters through the
$\theta$-dependence of $J_\varepsilon$, i.e. through the second derivatives $\partial_\theta\partial_\varepsilon M$, which is where the
quadratic couplings ($b_{\phi\delta}\delta\phi$, $f_{\rm NL}\phi^2$, 2LPT) act.

**Hutchinson estimator.** With Rademacher probes $z$ ($\langle zz^T\rangle = I$),
$$F_{ij} \approx \frac{1}{n_p}\sum_{p=1}^{n_p}\tfrac12\,\big(C^{-1}z_p\big)^T\,\partial_iC\,\big(C^{-1}\partial_jC\,z_p\big),$$
which needs, per probe, one solve $C^{-1}z$ and $n_\theta$ solves $C^{-1}(\partial_jC\,z)$, plus $n_\theta$ applications of
$\partial_iC$ to $C^{-1}z$. The operator $\partial_iC\,w = \partial_iJ_\varepsilon(J_\varepsilon^Tw) + J_\varepsilon(\partial_iJ_\varepsilon^Tw)$ uses
nested autodiff: $\partial_iJ_\varepsilon v = \partial_{\theta_i}[J_\varepsilon(\theta)v]$ is a jvp over $\theta$ of a jvp over $\varepsilon$
(forward view) and $\partial_iJ_\varepsilon^Tw$ is a jvp over $\theta$ of a vjp over $\varepsilon$ (reverse view). The parameter
direction is passed as a one-hot array so that one compiled program serves all $i$.

**Solver.** $C$ has eigenvalues from $\sigma^2 \sim 10^{-3}$ (edge voxels, noise floor) to $\sim 10^3$, so plain CG on $C$
stalls at the iteration cap. The solve is done on the noise-whitened system,
$$C^{-1}b = N^{-1/2}\left(N^{-1/2}J_\varepsilon J_\varepsilon^TN^{-1/2} + I\right)^{-1}N^{-1/2}b,$$
whose eigenvalues are $1 + \bar nP$ (at most $\sim 20$); the true residual $\|Cx - b\|/\|b\|$ is logged per probe
($\le 10^{-5}$ in all reported runs). An unpreconditioned $160^3$ attempt converged to a wrong 2.6 and was discarded;
an $80^3$ re-run with the preconditioned solver (5.32) confirmed the earlier $80^3$ numbers (5.40).

**Error bars.** Single-probe matrices can be indefinite, so the quoted ± is a jackknife over probes of $\sigma$ from the
mean Fisher. It measures estimator noise only. Cosmic variance of the fiducial field is measured separately by
repeating with a second $\varepsilon$ (10 % at $80^3$).

**Cost.** Once compiled: 10 s per probe at $80^3$, 100 s at $160^3$ (12 probes = 20 min); host memory 6 / 17 GB.
Memory was the first failure mode: each `FieldFisher` instance and each static-argument derivative closure compiled
new XLA programs that embed DISCO-DJ's scatter index tables, and two runs were OOM-killed at 120 GB. The fix is to
compile the operators once per (model, tracer set) with $\varepsilon$, $\theta$ and the noise as arguments (memory flat at
1.3 GB over four instances at $16^3$), which also removed recompilation and made the Schur solve 40× faster.

### 3.3 Closed forms (`fisher_closed.py`)

**Tier 1 (Gaussian two-point, Tegmark 1997).** Per shell $s$ and per independent (half-space) mode,
$$F_{ij} = \sum_s \int dV_s(z)\int_{\rm cube}\frac{d^3k}{(2\pi)^3}\,\frac{\partial_iP_g\,\partial_jP_g}{2\,[P_g(k,\mu;z) + 1/n_{\rm obs}(z)]^2},\qquad
P_g = \left[b_1 + \frac{b_\phi f_{\rm NL}}{T(k,z)} + f\mu^2\right]^2P_{\rm lin}(k,z),$$
the $\tfrac12$ being "$\tfrac12(\partial\ln P)^2$ per real degree of freedom" summed over the full $k$-space (equivalently
$\partial P\partial P/(P+N)^2$ per independent complex mode, 2 real dof). The $k$ integral runs over the modes of the grid
cube ($|k_i| \le k_{\rm Ny}$) from $k_{\min}$, with $\mu$ relative to one axis. The radial integral uses the Part A $n(z)$
tables (full footprint, observed density). A `laplace=True` option replaces the weight by $1/(2P(P+N))$ to give the
expectation of the Schur estimator for a linear model. Mode counting was cross-checked against Part A's multipole
Fisher on a box: unmarginalised $\sigma$ 0.516 vs 0.518.

Why both $k_{\min} = k_f = 2\pi/L$ and $k_{\min} = 0.003$ are reported: the per-$\ln k$ information density goes as
$\propto k^{-1}$ in the sample-variance regime (the $1/T \propto k^{-2}$ scale-dependent bias against $k^3$ mode counting),
so the mode sum is dominated by its lowest modes; with $k_f$ it counts $\lambda \approx 10\,h^{-1}$Gpc modes that a
$\approx 5\,h^{-1}$Gpc survey cannot constrain. The field-level estimators handle the window automatically; the
closed form only brackets them.

**Tier 2 (tree-level bispectrum, real space).**
$$B_g = c_1c_2c_3\,B_{\rm PNG} + \sum_{\rm cyc}c_1c_2\left[2b_1F_2(\mathbf{k}_1,\mathbf{k}_2) + b_2 + 2b_KS_2 + b_{\phi\delta}f_{\rm NL}\left(\tfrac1{T_1}+\tfrac1{T_2}\right)\right]P_1P_2,$$
$c_i = b_1 + b_\phi f_{\rm NL}/T_i$, $B_{\rm PNG} = 2f_{\rm NL}T_1T_2T_3(P_{\Phi,1}P_{\Phi,2} + {\rm cyc})$,
$S_2 = (\hat k_1\cdot\hat k_2)^2 - \tfrac13$, with
$${\rm Var}[B] = \frac{s_B\,(2\pi)^6\,\tilde P_1\tilde P_2\tilde P_3}{V\cdot 8\pi^2k_1k_2k_3\Delta k^3},\qquad
F_{ij} = \sum_{\rm triangles}\frac{\partial_iB\,\partial_jB}{{\rm Var}[B]},$$
$s_B = 6, 2, 1$ for equilateral, isosceles, general triangles, $\Delta k = 0.002$, $k < k_{\rm Ny}$, derivatives by autodiff.
It adds $< 3\%$ to Tier 1. This is now known to be an under-estimate (see §6), most likely because the squeezed
configurations with $k_L \sim 10^{-3}$, where $1/T \propto k_L^{-2}$ varies by 20× across one bin, are evaluated at bin
centres.

---

## 4. Validation chain

| test | result |
|---|---|
| forward model, $80^3$ box: $P_m/P_{\rm lin}$, $P_g/[(b_1 + b_\phi f_{\rm NL}/T)^2P_{\rm lin}]$ for $f_{\rm NL} = 0, \pm100$, RSD vs Kaiser | 0.98 at $k<0.005$; 1.01 / 1.03 / 1.16 at $k < 0.01$; Kaiser matched (§2.2) |
| dense $16^3$, linear model: CG Schur vs dense Schur | 0.777 vs 0.777 (LRG), 14.93 vs 14.93 (QSO) |
| dense $16^3$: dense $\tfrac12{\rm Tr}[\cdots]$ vs Tier 1 exact mode sum | 0.614 vs 0.614; 26.73 vs 26.73 |
| dense $16^3$: Hutchinson (16 probes) vs dense marginal | 0.617 vs 0.614; 26.66 vs 26.73 |
| dense $16^3$: Schur vs Laplace-weighted mode sum | 0.777 vs 0.729; 14.93 vs 14.55 (one-realisation scatter) |
| $80^3$ periodic box, linear model, 3 realisations: Schur vs Laplace-weighted sum | QSO 10.7–12.5 vs 11.3; LRG 0.56–0.72 vs 0.62 |
| $80^3$ DR1 window, linear model: Schur | QSO+LRG 4.2–5.1 vs 3.8 (window-free sum) |
| memory over 4 Fisher instances at $16^3$ | 1.25 → 1.29 GB |

The $16^3$ test is the decisive one: it is the only place where all four quantities (operator Schur, dense Schur,
dense marginal, mode sums) are computed independently on the same problem, and it exposed a factor $D^2$ bug in the
strictly linear *test* model (growth applied to the primordial-potential term) before any production run.

---

## 5. Results

DR1 window, 6 shells, $(b_1, b_2, b_K)$ per shell marginalised, $b_\phi$ fixed, fiducial $f_{\rm NL} = 0$, QSO $p = 1.6$
unless stated. Exact-marginal ± is the probe jackknife.

| grid | tracers | exact marginal | Schur (profile) | Tier 1 ($k_f$ / 0.003) |
|---|---|---|---|---|
| $80^3$ | QSO+LRG | 5.40 ± 0.31 (24 probes); 2nd realisation 6.00 ± 0.34; mean-$F$ 5.67 | 3.9 ± 0.4 (4 real.) | 5.4 / 8.9 |
| $80^3$ | QSO | 8.99 ± 0.12 | 4.7 ± 0.6 | 8.4 / 11.5 |
| $80^3$ | LRG | 9.42 ± 0.41 | 7.3 ± 0.8 | 7.1 / 14.1 |
| $80^3$, QSO $p = 1.0$ | QSO+LRG | 4.36 ± 0.23 | 2.5 | 5.4 / 8.9 |
| $80^3$, strictly linear model | QSO+LRG | 8.28 ± 0.21 | 3.4 | 5.4 / 8.9 |
| $160^3$ | QSO+LRG | **4.00 ± 0.18** (12 probes) | 2.8 | 5.0 / 7.4 |

With the 40 % Gaussian prior on $b_\phi$ propagated as $f_{\rm NL} = A/b_\phi$ (a Fisher at $f_{\rm NL} = 0$ is blind to it),
$\sigma \to 1.22\,\sigma$: $4.0 \to 4.9$ at $160^3$.

---

## 6. Interpretation

1. **Window.** Strictly linear model, DR1 window: 8.28, against 5.4 for the window-free sum with $k_{\min} = k_f$ and 8.9
   with $k_{\min} = 0.003$. The survey window removes the $\lambda \gtrsim 5\,h^{-1}$Gpc modes that dominate the naive sum;
   the field-level analysis sits at the conservative end of the closed-form bracket for two-point information.

2. **Schur vs marginal, nonlinear model.** On the same realisations the Schur numbers are 1.4–2.4× tighter than the
   exact marginal for QSOs and both tracers, as the $(P+N)/(2P)$ analysis predicts for shot-noise-dominated samples;
   for LRG alone the ordering is the opposite of the linear prediction (6.2 vs 9.4), which means that for the 2LPT
   model the two are genuinely different approximations rather than related by a per-mode factor. Neither is the true
   marginal of the non-Gaussian posterior; that needs Stage 2.

3. **Quadratic couplings.** Same realisation, window and grid: 8.28 (linear) → 5.32 (linearised 2LPT) at $80^3$, i.e.
   $1 - (5.32/8.28)^2 = 59\%$ of $F$ comes from terms beyond the two-point function, entering through
   $\partial C/\partial f_{\rm NL} \supset$ "small-scale power modulated by the local large-scale $\phi_g$" (the squeezed local-PNG
   signature via $b_{\phi\delta}$ and $f_{\rm NL}\phi^2$). At $160^3$ the gain grows (5.3 → 4.0) because the two-point
   information saturates while the number of modulated small-scale modes grows 8×. Three reasons to treat this as an
   upper limit on what a sampler will recover: the linearisation holds the modulating field fixed at its fiducial
   value; Fisher is a Cramér–Rao bound that non-Gaussian posteriors need not attain; and the tree-level Tier 2 as
   implemented credits these couplings with $< 3\%$, a disagreement that is attributed to its $k$ binning but not
   proven.

4. **Realisation scatter.** 10 % between two fiducial fields at $80^3$; the largest-scale modes dominate, so the few
   lowest $|\mathbf{k}|$ values of $|\tilde\varepsilon|^2$ matter. The Schur estimator scatters more (3.2–4.3) because it also
   depends on $\varepsilon$ through $\partial_\theta M$.

5. **$p = 1.0$ vs 1.6 for QSOs.** $b_\phi$ is 2–3× larger at $p = 1.0$, giving 4.36 vs 5.40 at $80^3$. This is a
   prior-dependent statement, not a measurement, and is the reason the $b_\phi$-sampled runs are deferred to Stage 2.

---

## 7. Choices made, and why

| choice | reason |
|---|---|
| exact Gaussian-marginal (Hutchinson) as the headline estimator | the brief's Schur complement is a profile curvature with a known, large, sample-dependent bias (§3.1, §4) |
| white-noise parametrisation | identity prior block → well-conditioned CG; required anyway for Stage 2 HMC |
| Lagrangian bias weights | correct Kaiser limit when particles are displaced; $\phi_g$ needs no growth factor |
| one $(a, b_1, b_2, b_K, b_\phi, b_{\phi\delta}, \bar n)$ per shell, shells as in the brief | light cone without a full time-evolution model; parameters per shell match how DESI fits evolve $b_1(z)$ |
| Gaussian noise, variance $\bar nV_{\rm cell}$ | identical Fisher to Poisson for mean parameters; cheap |
| noise floor $10^{-3}$ counts on partially covered edge voxels | avoids infinite weights; the preconditioner absorbs the resulting dynamic range |
| full footprint + $\bar c\,{\rm NX}$ density | the Part A treatment that reproduced DESI's LRG and QSO errors |
| CIC without deconvolution | window is part of the model; data would be gridded identically |
| fiducial $f_{\rm NL} = 0$ | matches the mocks DESI used; the $b_\phi$ prior is propagated analytically instead |
| separate runs per tracer set | slicing the joint matrix keeps the other tracer's $f_{\rm NL}$ information |
| Rademacher probes, 12–24 | validated at $16^3$; jackknife error $\sim 5\%$ at 24 probes |
| noise-preconditioned CG with logged true residuals | unpreconditioned solves stalled silently at $160^3$ |
| operators compiled once per model/tracer set | compiled DISCO-DJ scatter programs are large; per-instance compilation OOM-killed two runs |
| two autodiff views of the model | DISCO-DJ scatter supports only one AD mode per view |
| closed forms kept as brackets, not targets | the brief's "Tier 1 must match Schur" is false; Tier 2 under-resolves squeezed triangles |

---

## 8. What remains open (hand-off to Stage 2)

- Whether a sampler recovers $\sigma \approx 4$–5 (quadratic-coupling information) or $\approx 8$ (two-point only).
  Cheap intermediate check: Laplace approximation to the marginal posterior on a grid in $f_{\rm NL}$ (optimise over
  $\varepsilon$, add $\tfrac12\ln\det$ via the same trace machinery).
- Tier 2 with a finer low-$k$ binning, to resolve its disagreement with the linearised Fisher.
- A second (and third) fiducial realisation at $160^3$ (20 GPU-minutes each).
- $b_\phi$, $b_{\phi\delta}$ sampled with the 40 % prior at a non-zero fiducial $f_{\rm NL}$, once a sampler exists to
  compare against.
- Systematics (Stage 4) are absent from all numbers above.
