# Schur-complement operator check on the linear model (DR1 footprint)

res=80, DR1 footprint; voxels: LRG: 7647, QSO: 46090
- realisation 0: sigma(f_NL): QSO: Schur 14.04 vs Tier1-Laplace 4.46 (exact Gaussian 7.86) [15s, CG 6e-07]; LRG: Schur 11.09 vs Tier1-Laplace 7.25 (exact Gaussian 6.45) [14s, CG 9e-07]; QSO+LRG: Schur 8.51 vs Tier1-Laplace 3.80 (exact Gaussian 4.99) [22s, CG 9e-07]
- realisation 1: sigma(f_NL): QSO: Schur 15.35 vs Tier1-Laplace 4.46 (exact Gaussian 7.86) [14s, CG 9e-07]; LRG: Schur 16.76 vs Tier1-Laplace 7.25 (exact Gaussian 6.45) [14s, CG 9e-07]; QSO+LRG: Schur 11.11 vs Tier1-Laplace 3.80 (exact Gaussian 4.99) [22s, CG 1e-06]
- realisation 2: sigma(f_NL): QSO: Schur 12.97 vs Tier1-Laplace 4.46 (exact Gaussian 7.86) [14s, CG 9e-07]; LRG: Schur 12.17 vs Tier1-Laplace 7.25 (exact Gaussian 6.45) [14s, CG 9e-07]; QSO+LRG: Schur 8.72 vs Tier1-Laplace 3.80 (exact Gaussian 4.99) [22s, CG 1e-06]

Interpretation: in the periodic box, Schur must equal Tier1-Laplace (same modes, no window) up to the realisation
scatter of the few largest-scale modes; with the DR1 footprint the window removes the k ~ k_f modes that dominate the
k_min = k_f mode sum, so Schur is larger than the full-box closed form (compare with the k_min = 0.003 closed form).
The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is (P+N)/(2P) times the Gaussian-field Fisher.

Figure: `test/figures/fisher_field_linear_check_res80.png`
