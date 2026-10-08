# Schur-complement operator check on the linear model

res=80, voxels in footprint: LRG: 7647, QSO: 46090
- realisation 0: 24s, max CG residual 9.2e-07; sigma(f_NL): QSO: Schur 8.07 vs Tier1-Laplace 5.25 (exact Gaussian 6.64); LRG: Schur 7.96 vs Tier1-Laplace 4.84 (exact Gaussian 6.28); QSO+LRG: Schur 8.51 vs Tier1-Laplace 5.37 (exact Gaussian 7.05)
- realisation 1: 23s, max CG residual 9.6e-07; sigma(f_NL): QSO: Schur 10.29 vs Tier1-Laplace 5.25 (exact Gaussian 6.64); LRG: Schur 9.96 vs Tier1-Laplace 4.84 (exact Gaussian 6.28); QSO+LRG: Schur 11.11 vs Tier1-Laplace 5.37 (exact Gaussian 7.05)

Interpretation: agreement of Schur with Tier1-Laplace validates jvp/vjp/CG and the footprint/noise bookkeeping.
The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is (P+N)/(2P) times the Gaussian-field Fisher.

Figure: `test/figures/fisher_field_linear_check_res80.png`
