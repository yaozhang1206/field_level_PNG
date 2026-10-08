# Schur-complement operator check on the linear model (periodic box)

res $=80$, periodic box, no window; voxels: LRG: 512000, QSO: 512000
Linear model: $\langle N\rangle = \bar nV\,[1 + (b_1 + b_\phi f_{\rm NL}/T(k) + f\mu_z^2)\,D\,\delta_L]$; Schur $= F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta}$; Tier 1 weights: exact $\partial P\partial P/(P+N)^2$, Laplace $\partial P\partial P/(2P(P+N))$ per mode.
- realisation 0: $\sigma(f_{\rm NL})$: QSO: Schur 11.02 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 0.59 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 6e-07]; QSO+LRG: Schur 0.56 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 7e-07]
- realisation 1: $\sigma(f_{\rm NL})$: QSO: Schur 12.48 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 0.72 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 8e-07]; QSO+LRG: Schur 0.67 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 1e-06]
- realisation 2: $\sigma(f_{\rm NL})$: QSO: Schur 10.70 vs Tier1-Laplace 11.33 (exact Gaussian 17.48) [4s, CG 9e-07]; LRG: Schur 0.56 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [4s, CG 8e-07]; QSO+LRG: Schur 0.53 vs Tier1-Laplace 0.62 (exact Gaussian 0.51) [5s, CG 1e-06]

Interpretation: in the periodic box, Schur must equal Tier1-Laplace (same modes, no window) up to the realisation
scatter of the few largest-scale modes; with the DR1 footprint the window removes the $k \sim k_f$ modes that dominate the
$k_{\min} = k_f$ mode sum, so Schur is larger than the full-box closed form (compare with the $k_{\min} = 0.003$ closed form).
The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is $(P+N)/(2P)$ times the Gaussian-field Fisher.

Figure: `test/figures/fisher_field_linear_check_res80_box.png`
