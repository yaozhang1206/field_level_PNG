# Schur-complement operator check on the linear model (periodic box)

res=80, periodic box, no window; voxels: LRG: 512000, QSO: 512000
- realisation 0: sigma(f_NL): QSO: Schur 18.16 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 0.98 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 6e-07]; QSO+LRG: Schur 0.92 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 7e-07]
- realisation 1: sigma(f_NL): QSO: Schur 20.57 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 1.19 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 8e-07]; QSO+LRG: Schur 1.11 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 1e-06]
- realisation 2: sigma(f_NL): QSO: Schur 17.63 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 0.92 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 8e-07]; QSO+LRG: Schur 0.87 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 1e-06]

Interpretation: in the periodic box, Schur must equal Tier1-Laplace (same modes, no window) up to the realisation
scatter of the few largest-scale modes; with the DR1 footprint the window removes the k ~ k_f modes that dominate the
k_min = k_f mode sum, so Schur is larger than the full-box closed form (compare with the k_min = 0.003 closed form).
The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is (P+N)/(2P) times the Gaussian-field Fisher.

Figure: `test/figures/fisher_field_linear_check_res80_box.png`
