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
  **$\sigma(f_{\rm NL}) = $ SCHUR_RESULT** ($160^3$, QSO+LRG, $b_\phi$ fixed; see §4).
- **Decision:** GO_NOGO_TEXT

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

### 4.2 Schur-complement field-level Fisher (`test/notes/fisher_field.md`)

SCHUR_TABLE

Validation: see §4.3 (dense 16³ check) and `test/notes/fisher_field_linear_check_res80*.md` (80³, linear model, periodic box
and DR1 window). With the DR1 window the field-level results exceed the k_min = k_f mode sum because the window removes the
largest modes; they should be compared with the $k_{\min} \approx 0.003$ closed form.

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

## 5. Where the signal is

Per-mode information density peaks at $k \approx 0.003$–$0.01\,h/{\rm Mpc}$ for both tracers (`fisher_pk_info_density_full.png`),
i.e. $\bar nP \approx 0.4$–0.7 for QSOs at $z > 2$ and the sample-variance regime for LRGs; the high-$z$ QSO shell (2.1–3.1) and the
0.8–1.1 LRG shell carry the most $f_{\rm NL}$ information per shell ($\sigma \approx 10$ each, Tier 1).

## 6. Outputs

- `results/fisher_pk_full.json`, `results/fisher_closed.json`, `results/fisher_field_*.json`, `results/fisher_field.json`
- notes and figures for every test in `test/notes`, `test/figures`
