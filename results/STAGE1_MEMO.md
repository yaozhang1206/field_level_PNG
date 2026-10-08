# Stage 1 memo: information content of DESI DR1 QSO+LRG on local f_NL

Date: 2026-10-08. Code: this repository (commit TBD). Environment: Delta AI, conda env `pm`, DISCO-DJ 0.0.2, CLASS 3.4.

## 1. Summary and go/no-go

- **Part A (P(k) Fisher) reproduces DESI DR1.** With the public DR1 catalogues (n(z) from the NX column, geometric
  footprint 9,070 deg² from the randoms, completeness-reduced shot noise) the Gaussian P0+P2 Fisher gives
  σ(f_NL) = 15.5 for QSO FKP (DESI clean mocks: 15.5), 20.0 / 13.1 for LRG with k_min = 0.006 / 0.003 (DESI: 14.5),
  and 7.9–8.5 for QSO+LRG in the optimal-weighting limit (DESI OQE: 9.2). Conventions (T_Φ→δ, b_φ, volumes,
  shot noise) are therefore calibrated.
- **Part B (field level).** Closed-form Gaussian mode sum over the 160³ grid (k_Ny = 0.05 h/Mpc, the "k_max = 0.1"
  configuration of the brief) with f_NL + (b1, b2, bK) per shell marginalised: σ(f_NL) = 5.0 (QSO+LRG) when all box
  modes down to k_f = 6×10⁻⁴ are counted, 7.4 with k_min = 0.003. The tree-level bispectrum adds < 3 %.
  The Schur-complement (joint-Gaussian) Fisher with the DISCO-DJ 2LPT forward model and the real DR1 window gives
  **σ(f_NL) = SCHUR_RESULT** (160³, QSO+LRG, b_φ fixed, mean over realisations; see §4).
- **Decision:** GO_NOGO_TEXT

## 2. Conventions carried into Stage 2

| item | choice |
|---|---|
| cosmology | Planck 2018 (Ωm=0.3153, Ωb=0.0493, h=0.6736, As=2.1e-9, ns=0.9649, mν=0.06 eV); CLASS for P_lin; DISCO-EB agrees to 1.1 % (0.5 % with massless ν) |
| PNG convention | Φ = 3/5 ζ (LSS/Slosar), P_Φ = (9/25)(2π²/k³) As (k/k_p)^(ns−1); T_Φ→δ = (2/3) k² T(k) D_md(z) c²/(Ωm H0²); Δb = b_φ f_NL / T_Φ→δ |
| bias | b_φ = 2δc(b1 − p), p = 1.6 (QSO), 1.0 (LRG); b1(z) = a(1+z)²+b (a,b = 0.237,0.771 QSO; 0.209,1.415 LRG); b2 Lazeyras+16; bK = −2/7(b1−1); b_φδ Euclid Eq. |
| box / grids | L = 10,000 Mpc/h, observer at centre; 80³ (ΔL = 125, k_Ny = 0.025) and 160³ (ΔL = 62.5, k_Ny = 0.05) |
| shells | QSO 0.8–1.3, 1.3–2.1, 2.1–3.1; LRG 0.4–0.6, 0.6–0.8, 0.8–1.1; one (a, b1, b2, bK, b_φ, b_φδ, n̄) per shell |
| selection | angular: fractional HEALPix (nside 64) coverage from the QSO randoms; radial: shell χ range; n̄ = c̄·NX(z) with c̄ = A_eff/A_full |
| forward model | ε → δ_L(k) → φ_g = δ_L/T → Φ_NG = φ_g + f_NL(φ_g²−⟨φ_g²⟩) → δ_NG → DISCO-DJ 2LPT → Lagrangian weights → RSD (radial LOS) → CIC |
| likelihood | Gaussian, σ² = n̄ V_cell per voxel, zero weight off footprint |
| noise parameter | s_n per shell decouples from the mean-model Fisher at the Gaussian level; not included |

## 3. Part A details

Method: Gaussian covariance of (P0, P2) in Δk = 0.001 bins with the FKP-weighted variance
S(k,μ) = ∫dV n⁴w⁴ (P + 1/n_obs)² / (∫dV n²w²)², w = 1/(1+nP0), integrated over the DR1 n(z); parameters
(f_NL, b1, s_n0, Σs), fiducial f_NL = 0, Σs = 3 Mpc/h. Derivatives by jax autodiff.

| configuration | σ(f_NL) marg. | b1 fixed | DESI clean mocks (Tables 4–5) |
|---|---|---|---|
| QSO FKP (z_eff 1.651, b1 2.44, 0.003<k<0.08) | 15.5 | 11.8 | +18/−13 (mean 15.5) |
| QSO OQE-like single z (z 2.082, b1 3.08) | 8.6 | 6.6 | +13/−10 (mean 11.5) |
| QSO 10 shells, b1(z) template (optimal-weighting limit) | 9.6 | 7.4 | +13/−10 (mean 11.5) |
| LRG FKP (z_eff 0.733, b1 2.06, k_min 0.006) | 20.0 | 13.1 | +18/−11 (mean 14.5) |
| LRG FKP, k_min 0.003 | 13.1 | 10.4 | |
| QSO+LRG FKP | 12.3 | | |
| QSO+LRG shells | 8.5 | | +10.5/−7.9 (mean 9.2) |

Notes: the LRG number is sensitive to k_min (0.003 vs 0.006 changes σ by 1.5×) and the DESI LRG posterior is
strongly asymmetric; the QSO numbers match to 3 %. The 'full' treatment (sample variance from the geometric
footprint, shot noise from the completeness-reduced density) was needed: with the effective area alone LRG gave 24.
Figures: `test/figures/fisher_pk_kcuts_full.png`, `fisher_pk_info_density_full.png`. Details: `test/notes/fisher_pk_full.md`.

## 4. Part B details

### 4.1 Closed forms (`test/notes/fisher_closed.md`)

| grid | tracers | Tier 1 (P, k_min = k_f) | Tier 1 + Tier 2 | Tier 1, k_min = 0.003 |
|---|---|---|---|---|
| 80³ | QSO / LRG / both | 8.4 / 7.1 / 5.4 | 8.4 / 7.1 / 5.4 | 11.5 / 14.1 / 8.9 |
| 160³ | QSO / LRG / both | 7.5 / 6.7 / 5.0 | 7.4 / 6.6 / 4.9 | 9.5 / 11.9 / 7.4 |

The PNG information is dominated by the largest scales (per-ln k information ∝ k⁻¹ in the sample-variance
regime), so the closed form depends strongly on k_min: the k_f value counts modes with λ ≈ 10 Gpc/h that the
≈ 5 Gpc/h survey cannot constrain, the 0.003 value is conservative. The tree-level (real-space) bispectrum adds
< 3 % at k ≤ 0.05.

### 4.2 Schur-complement field-level Fisher (`test/notes/fisher_field.md`)

SCHUR_TABLE

Validation: on a strictly linear model in a periodic box the Schur result agrees with the mode sum weighted
by dP dP/(2P(P+N)) to LINCHECK_BOX; with the DR1 window the Schur result exceeds the k_f mode sum
because the window removes the largest modes (`test/notes/fisher_field_linear_check_res80*.md`).

### 4.3 Schur (profile) vs exact marginal: validation at 16³ (`test/notes/schur_dense16.md`)

Brute-force dense computation on a periodic 16³ box with the strictly linear model:

| | LRG (n̄ = 3e-4) | QSO (n̄ = 2.5e-5) |
|---|---|---|
| Schur complement, CG operators / dense | 0.777 / 0.777 | 14.93 / 14.93 |
| exact Gaussian-marginal ½Tr(C⁻¹∂C C⁻¹∂C), dense / Hutchinson (16 probes) | 0.614 / 0.617 | 26.73 / 26.66 |
| Tier 1 mode sum, exact / Laplace-weighted dP dP/(2P(P+N)) | 0.614 / 0.729 | 26.73 / 14.55 |

So (i) the jvp/vjp/CG operator code is exact, (ii) the Tier 1 mode sum equals the exact Gaussian-marginal
Fisher, (iii) the Schur complement of the joint (ε, θ) Fisher is the *profile* curvature, which differs from the
marginal by (P+N)/(2P) per mode: pessimistic where nP ≫ 1 (LRG, ×1.27 in σ here) and **optimistic by ×1.8 in σ
(×3.3 in F) for the shot-noise-dominated QSOs**. The Schur method of the brief therefore cannot be used as the
forecast; the exact Gaussian-marginal Fisher of the model linearised around the fiducial ε (implemented as
`FieldFisher.fisher_marginal`, cost ≈ (n_θ+1) × n_probes CG solves) is used instead, with the Schur numbers
reported for reference only.

At fiducial f_NL = 0 any Fisher matrix is blind to the b_φ prior (b_φ enters only as b_φ f_NL); propagating
f_NL = A/b_φ with a 40 % Gaussian prior on b_φ inflates σ by ≈ √(1 + 3·0.4²) = 1.22.

## 5. Where the signal is

Per-mode information density peaks at k ≈ 0.003–0.01 h/Mpc for both tracers (`fisher_pk_info_density_full.png`),
i.e. nP ≈ 0.4–0.7 for QSOs at z > 2 and the sample-variance regime for LRGs; the high-z QSO shell (2.1–3.1) and the
0.8–1.1 LRG shell carry the most f_NL information per shell (σ ≈ 10 each, Tier 1).

## 6. Outputs

- `results/fisher_pk_full.json`, `results/fisher_closed.json`, `results/fisher_field_*.json`, `results/fisher_field.json`
- notes and figures for every test in `test/notes`, `test/figures`
