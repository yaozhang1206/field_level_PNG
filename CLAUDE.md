# DESI f_NL Field-Level Inference — Project Summary and Stage 1 Plan

This file is intended as the working brief for a Claude Code session on NCSA Delta / Delta AI.
Drop it in the repo root (optionally rename to `CLAUDE.md`).

---

## 1. Project goal

Constrain local primordial non-Gaussianity (f_NL^loc) with Bayesian field-level inference
(BORG-style forward modelling) applied to DESI DR1 quasars (QSO) and luminous red galaxies (LRG).

**Definition of "doable"** (decided in planning):
- Idealized (no systematics) forecast σ(f_NL) ≲ 5 at k_max = 0.1 h/Mpc with bias marginalized.
- Final target σ(f_NL) ≤ 6 on DR1 QSO+LRG with |Δf_NL| < 2 on contaminated mocks.
- Comparison baseline: DESI DR1 power-spectrum result f_NL = −3.6 +9.0/−9.1 (Chaussidon et al. 2024).

---

## 2. Reference papers and the numbers we rely on

### 2.1 Chaussidon et al. 2024 — DESI DR1 P(k) PNG analysis (arXiv:2411.17623)
- Samples: 2,130,621 LRGs (0.4 < z < 1.1); 1,189,129 QSOs (0.8 < z < 3.1).
- Model: Kaiser + Lorentzian damping, P(k,μ) = [b1 + bφ f_NL / T_Φ→δ(k) + f μ²]² / [1 + (kμΣs)²/2]² · P_lin + s_n0.
- bφ = 2 δc (b1 − p), δc = 1.686. p = 1.6 (QSO), p = 1.0 (LRG).
- Free parameters per tracer: (f_NL, b1, s_n0, Σs). Cosmology fixed to Planck 2018.
- Data vector: P0 and P2 only. k range: 0.003 < k < 0.08 h/Mpc (LRG k_min raised to 0.006).
  Δk = 0.001 (ℓ=0), 0.002 (ℓ=2).
- FKP P0: 3e4 (QSO), 5e4 (LRG) (h^-1 Mpc)^3. Box for P(k) estimation: 16000 h^-1 Mpc, 6 h^-1 Mpc cells.
- Effective redshifts (Table 3):
  - LRG: z_eff(FKP) = 0.733; z_eff(OQE ℓ=0) = 0.754
  - QSO: z_eff(FKP) = 1.651; z_eff(OQE, p=1.6, ℓ=0) = 2.082; ℓ=2: 1.989
- Linear bias evolution b1(z) = a(1+z)² + b (Table 2; a≈0.209 LRG, a≈0.237 QSO; b values in paper).
- Fitted b1 at z_eff: LRG ≈ 2.06 (FKP); QSO ≈ 2.44 (FKP), ≈ 3.08 (OQE).
- Mean angular densities (Table 1): LRG ≈ 530 deg^-2; QSO ≈ 185 deg^-2.
- Effective-area fractions (LRG, QSO): North (18,20)%, South-NGC (48,44)%, South-SGC (25,27)%, DES (9,9)%.
- **Statistical errors from clean EZmocks (what Stage 1 must reproduce):**
  - QSO DR1 OQE: f_NL = −3 +13/−10   (Table 4)
  - QSO DR1 FKP: 3 +18/−13
  - LRG DR1 FKP: 5 +18/−11
  - QSO+LRG DR1 OQE: 1.1 +10.5/−7.9  (Table 5)
  - Y5 forecast combined: σ ≈ 6.5–7
- Dominant systematic: imaging systematics; mitigated by template regression → angular integral
  constraint (AIC); radial integral constraint (RIC) from shuffled randoms. Both absorbed into the
  window matrix W → W − W^RIC − W^AIC. These affect the central value and k_min only, NOT the
  statistical error.

### 2.2 Andrews et al. 2024/2026 — Euclid field-level f_NL forecast (arXiv:2412.11945; A&A 711, A94)
- BORG, 2LPT, box L = 8000 h^-1 Mpc, grids 32³/64³/128³ → ΔL = 250/125/62.5 h^-1 Mpc, k_max = 0.025/0.05/0.1.
  (Published version extends to ΔL = 31.25.)
- f_NL enters as Φ_NG = φ_g + f_NL(φ_g² − ⟨φ_g²⟩) on the primordial potential before the transfer function.
- Bias: ρ_g = ⟨N⟩[1 + b1 δ + b2 δ²/2 + bK K² + bφ f_NL φ_g + bφδ f_NL δ φ_g].
  bφ = 2δc(b1 − p); bφδ = bφ − b1 + 1 + δc[b2 − (8/21)(b1 − 1)]. Default p = 0.55.
- Gaussian likelihood with σ² = ⟨N⟩. RSD included. Single fixed redshift per catalogue (no light cone).
- Sampler: Gibbs; HMC on initial conditions, slice sampling on f_NL and bias. Burn-in ~5000 samples.
  f_NL autocorrelation length ≈ 10,000 samples (when bias is sampled).
- Results (p=0.55): σ(f_NL) = 6.0 (250), 4.0 (125), 2.3 (62.5). p=1 at 125: 6.3. Fixed bias at 125: 2.6.
  Sampling bφ, bφδ with 40% prior at 125: 5.0.
- Mocks generated with the same forward model (no model-mismatch test).

### 2.3 Related 2026 work
- Andrews et al. 2026, BORG on Quaia quasars in a (10 h^-1 Gpc)³ box (arXiv:2602.02363) — template for QSO geometry/light cone.
- Andrews et al. 2026, Quijote-PNG field-level f_NL on N-body halo catalogues (arXiv:2603.20855): ~1.3× improvement over P+B at k_max≈0.1.
- McAlpine et al. 2026, Manticore-Deep (arXiv:2606.10020): COLA-based BORG, 256³ tiles at 4 h^-1 Mpc, 0.5–1.5M CPU-h per tile — the cost reference for high-resolution field-level runs.

---

## 3. Decisions made during planning

1. **Zel'dovich vs 2LPT**: ZA is accurate enough for f_NL at k ≤ 0.05 and probably 0.1, but gives
   no meaningful speed-up (2LPT is ~2× ZA per gradient and both are cheap at 80³–160³ grids).
   Default to 2LPT; test ZA in Stage 3 for model mismatch.
2. **Bottleneck is chain length, not the forward model.** f_NL mixing (~10⁴-sample correlation
   length) is the thing to fix: joint HMC/NUTS on (field, f_NL, bias), Fourier-diagonal mass matrix,
   MCLMC (BlackJAX) as the candidate accelerated sampler.
3. **Tooling**: build the forward model on **DISCO-DJ** (List, Hahn, Flöss, Winkler; JAX end to end;
   arXiv:2510.05206 for the PM/LPT module, arXiv:2311.03291 for the Einstein–Boltzmann module).
   Chosen over JaxPM/pmwd because: (i) nLPT recursion gives ZA/2LPT/3LPT from one code path, so the
   Stage 3 ZA-vs-2LPT test is a one-parameter change; (ii) de-aliasing, sheet-based resampling and
   higher-order mass assignment suppress discreteness/aliasing power that b2/bK would otherwise
   absorb and leak into f_NL; (iii) DISCO-EB provides differentiable P_lin and T_Φ→δ(k) consistent
   with the solver; (iv) memory-efficient adjoint for grids beyond 160³; (v) the user has verified
   its accuracy exceeds pmwd/JaxPM. Own JAX code for: PNG perturbation on the primordial potential,
   bias model, selection/light-cone, likelihood. Request BORG access through Aquila in parallel.

   **Stage 0 API checks for DISCO-DJ (do before Stage 2):**
   - Entry point: can the LPT module take the primordial potential (or white noise) so that
     Φ_NG = φ_g + f_NL(φ_g² − ⟨φ_g²⟩) is applied *before* the transfer function, and φ_g at the
     Lagrangian position is available for the bφ f_NL φ_g and bφδ f_NL δ φ_g bias terms? If the API
     takes δ_L as input, implement the PNG step in JAX and hand δ_L to DISCO-DJ.
   - Differentiability of the chosen de-aliasing / resampling / mass-assignment options at the
     settings used (some discreteness-suppression schemes use sorting or nearest-neighbour ops
     that break or slow gradients).
   - Gradient cost and memory of 2LPT forward+adjoint at 160³ and 320³ particles on one GH200/A100.
4. **Hardware**: Stage 1 (Fisher) is CPU-only. Stage 2+ uses GPUs (Delta AI GH200 or Delta A100).
5. **Systematics strategy**: model imaging templates inside the forward model with sampled
   amplitudes (avoids the AIC) rather than pre-weighting the catalogue.
6. **Light cone**: QSOs span a ≈ 0.24–0.55; use redshift shells (or a light-cone model) from the
   start, including in the Stage 1 Fisher.
7. **bφ**: run everything both with bφ fixed (UMF relation) and sampled with a 40% Gaussian prior.
8. **ELGs excluded** (large-scale excess unresolved in DR1).

---

## 4. Four-stage feasibility plan (overview)

| Stage | What | Go/no-go | Hardware | Time |
|---|---|---|---|---|
| 0 | Define targets, DISCO-DJ API checks (see §3.3), BORG access | — | 1 GPU for API/timing checks | 1 wk |
| 1 | Fisher: reproduce DESI P(k) σ, then field-level Fisher | field-level σ ≲ 5 at k_max 0.1, bias marginalized | CPU | 3–4 wk |
| 2 | Self-consistent mock MCMC at 125 h^-1 Mpc, QSO only; sampler comparison | f_NL corr. length ≤ 10³; 62.5 production ≤ ~1 GPU-month | GPU | 4–6 wk |
| 3 | Model mismatch on Abacus / Quijote-PNG halos; ZA vs 2LPT | \|Δf_NL\| < 0.5σ at k_max 0.1 | GPU | 4 wk |
| 4 | Imaging + radial systematics on contaminated mocks with template sampling | \|Δf_NL\| < 2; σ degraded < 30% | GPU | 6–8 wk |

---

## 5. Stage 1 — detailed plan

### 5.1 Purpose
Establish the information content of DESI DR1 QSO+LRG on f_NL with no systematics, in two steps:
(A) reproduce the DESI P(k) statistical errors as a calibration of volume/n(z)/bias conventions;
(B) compute the field-level Gaussian Fisher under the full bias model to get the idealized σ(f_NL).

### 5.2 Environment (Delta)
- Partition: CPU (`cpu` on Delta) or an interactive node. No GPU needed.
- Python env: numpy, scipy, classy (CLASS) and/or DISCO-DJ (discodj; its DISCO-EB module can replace
  CLASS and gives autodiff derivatives of P_lin and T_Φ→δ), astropy, healpy, fitsio/h5py, matplotlib, jax (CPU).
  Optional: cosmoprimo (DESI wrapper), pypower, mockfactory (DESI, for randoms/window).
  Use the same DISCO-DJ environment for Stage 1 and Stage 2 so cosmology/transfer conventions carry over.
- Memory: 16–32 GB for everything except the optional window-matrix build (~100 GB, few hours).
- Store large inputs (randoms) on scratch; keep repo small.

### 5.3 Inputs to obtain
1. DESI DR1 LRG and QSO clustering catalogues + randoms (public DR1 release). If unavailable
   immediately, start with the paper's Table 1 densities, Fig. 2 n(z) (digitize), and an approximate
   footprint (DR1 ≈ 7,500 deg²; use the Legacy Survey footprint masked to DR1 tiles if available).
2. Planck 2018 cosmology for CLASS: Ωm=0.3153, ΩΛ=0.6847, Ωb=0.0493, h=0.6736, As=2.1e-9, ns=0.9649.
   (DESI paper fixes Planck 2018; the Euclid paper uses the same.)
3. Bias values: b1(z) from DESI Table 2; b2, bK from a reasonable prescription (e.g. Lazeyras et al.
   b2(b1) fit; bK = −(2/7)(b1−1) as a placeholder) — only needed for part B.

### 5.4 Part A — Reproduce DESI P(k) Fisher

**Tasks**
1. `cosmology.py`: wrap DISCO-EB (fallback: CLASS); return P_lin(k,z), T_Φ→δ(k,z) = sqrt(P_lin/P_Φ), f(z), D(z).
   Cross-check DISCO-EB vs CLASS P_lin at z_eff to ≤0.5% as a sanity test.
   P_Φ(k) = (9/25)(2π²/k³) As (k/k_pivot)^(ns−1), k_pivot = 0.05 Mpc^-1. Check units (h/Mpc vs 1/Mpc) carefully.
2. `model_pk.py`: implement Eq. 2.4 of Chaussidon et al.; Legendre multipoles P0, P2 via μ-integration.
   Parameters θ = (f_NL, b1, s_n0, Σs). Verify bφ(b1,p).
3. `geometry.py`: n(z) per tracer, V_eff(k) = ∫ [n̄(z)P0/(1+n̄(z)P0)]² dV, using the paper's P0 values.
   First pass: analytic shell volumes × sky fraction. Second pass: from randoms.
4. `fisher_pk.py`: Gaussian covariance per k-bin,
   Cov[P_ℓ, P_ℓ'](k) from the standard Kaiser-Gaussian formulae with shot noise 1/n̄,
   N_modes = V_eff k² Δk / (2π²). Numerical derivatives ∂P_ℓ/∂θ (or JAX autodiff). F = Σ_k ∂P^T C^-1 ∂P.
   Report σ(f_NL) marginalized over (b1, s_n0, Σs), and also with b1 fixed.
5. Run configurations:
   - QSO, FKP weights, z_eff=1.651, b1=2.44, p=1.6, 0.003<k<0.08
   - QSO, OQE weights, z_eff=2.082, b1=3.08, p=1.6  (use V_eff and n̄ appropriate to the OQE-weighted sample; approximate by evaluating at z_eff)
   - LRG, FKP, z_eff=0.733, b1=2.06, p=1.0, 0.006<k<0.08
   - Combined QSO+LRG (independent, as in the paper)
6. Also compute σ(bφ·f_NL) (assumption-free) for comparison with Table 4 second block.
7. Plot σ(f_NL) vs k_min and vs k_max to compare with their Fig. 8.

**Validation targets** (clean EZmock numbers, Tables 4–5): QSO OQE σ ≈ 10–13; LRG FKP ≈ 11–18;
combined OQE ≈ 8–10.5. Fisher is symmetric; aim within ~20% of the average of the asymmetric bounds.
If far off, check: V_eff, n̄P at the peak, whether P2 is included, Δk binning, Σs damping.

**Optional refinements** (only if the first pass disagrees): window-matrix convolution via pypower
on the DR1 randoms; redshift-shell integration instead of a single z_eff; OQE weighting applied
explicitly in V_eff.

### 5.5 Part B — Field-level Fisher

**Setup**
- Box: L = 10,000 h^-1 Mpc (QSOs at z=3.1 are at χ ≈ 4,600 h^-1 Mpc). Observer at centre.
- Grids: 80³ (ΔL=125, k_max=0.05) and 160³ (ΔL=62.5, k_max=0.1). Report both.
- Redshift shells: QSO 0.8–1.3, 1.3–2.1, 2.1–3.1 (match DESI imaging bins); LRG 0.4–0.6, 0.6–0.8, 0.8–1.1.
  Each shell has its own b1, bφ, n̄, f, D.
- Bias model: Euclid Eq. 4 (b1, b2, bK, bφ f_NL φ_g, bφδ f_NL δ φ_g), RSD via Kaiser at the field level.

**Method**
At k ≤ 0.1 with a Gaussian likelihood the field-level Fisher for the global parameters, after
marginalizing the field, reduces per mode to the Gaussian-field information plus the tree-level
bispectrum contribution. Implement in two tiers:
1. **Tier 1 (two-point, mode-by-mode)**: for each k-mode and shell, information on f_NL from the
   galaxy–galaxy power with the full k-dependence of bφ/T(k), summed over modes inside the survey
   volume (no multipole compression, both μ-dependence and the k^-2 term retained). This should
   slightly exceed Part A at the same k range because nothing is compressed.
2. **Tier 2 (add tree-level bispectrum)**: add the Fisher information from the galaxy bispectrum at
   tree level with local-PNG primordial bispectrum + bφδ term + gravitational F2, Gaussian
   covariance with shot noise. Use standard Fisher-bispectrum code (e.g. adapt Karagiannis et al.
   2018 formulae). This approximates the extra information field-level inference captures beyond P(k).
3. Marginalize over (b1, b2, bK, s_n) per shell; run with bφ, bφδ fixed (UMF) and with a 40% Gaussian prior.
4. Report σ(f_NL): QSO only, LRG only, combined; k_max = 0.05 and 0.1; p=1.6 and p=1.0 for QSOs.

**Go/no-go**: combined σ(f_NL) ≲ 5 at k_max=0.1 with bias marginalized and bφ sampled with 40% prior.
If ≳ 7: reframe toward DESI Y5 and/or add CMB-lensing cross-correlation.

### 5.6 Outputs of Stage 1
- `results/fisher_pk.json`: σ(f_NL) per configuration vs DESI Tables 4–5.
- `results/fisher_field.json`: σ(f_NL) per shell/tracer/k_max/bφ-treatment.
- Plots: σ vs k_min, σ vs k_max, per-mode information density dF/dlnk (shows where the signal is;
  expect k≈0.005–0.02, nP≈0.3–0.5 for QSOs).
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
    fisher_field.py
    bispectrum_fisher.py
  configs/*.yaml           (tracer, z-range, b1, p, k-range, weights)
  scripts/run_fisher_pk.py, run_fisher_field.py
  results/
  notebooks/
```

### 5.8 Things to watch
- Units: CLASS returns Mpc, DESI quotes h/Mpc. T_Φ→δ normalization (Φ = 3/5 R convention) changes bφ f_NL by a constant factor if wrong.
- Sign convention of f_NL (LSS vs CMB) — DESI follows the Slosar et al. LSS convention.
- Shot noise at the QSO densities: n̄P ≈ 0.5 at the P(k) peak; results are sensitive to n̄(z) at z > 2.
- Σs is poorly constrained for QSOs (OQE fit returned 0); fix it or put a prior if it destabilizes the Fisher.
- Don't over-invest in the window matrix in Stage 1; k_min cut is enough for the go/no-go.