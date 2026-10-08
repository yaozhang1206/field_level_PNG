# Schur-complement operator check on the linear model (DR1 footprint)

res $=80$, DR1 footprint; voxels: LRG: 7647, QSO: 46090
Linear model: $\langle N\rangle = \bar nV\,[1 + (b_1 + b_\phi f_{\rm NL}/T(k) + f\mu_z^2)\,D\,\delta_L]$; Schur $= F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta}$; Tier 1 weights: exact $\partial P\partial P/(P+N)^2$, Laplace $\partial P\partial P/(2P(P+N))$ per mode.
- realisation 0: $\sigma(f_{\rm NL})$: QSO: Schur 5.34 vs Tier1-Laplace 4.46 (exact Gaussian 7.86) [15s, CG 6e-07]; LRG: Schur 7.21 vs Tier1-Laplace 7.25 (exact Gaussian 6.45) [14s, CG 9e-07]; QSO+LRG: Schur 4.23 vs Tier1-Laplace 3.80 (exact Gaussian 4.99) [22s, CG 9e-07]
- realisation 1: $\sigma(f_{\rm NL})$: QSO: Schur 5.80 vs Tier1-Laplace 4.46 (exact Gaussian 7.86) [15s, CG 9e-07]; LRG: Schur 10.87 vs Tier1-Laplace 7.25 (exact Gaussian 6.45) [14s, CG 9e-07]; QSO+LRG: Schur 5.07 vs Tier1-Laplace 3.80 (exact Gaussian 4.99) [22s, CG 1e-06]

Interpretation: in the periodic box, Schur must equal Tier1-Laplace (same modes, no window) up to the realisation
scatter of the few largest-scale modes; with the DR1 footprint the window removes the $k \sim k_f$ modes that dominate the
$k_{\min} = k_f$ mode sum, so Schur is larger than the full-box closed form (compare with the $k_{\min} = 0.003$ closed form).
The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is $(P+N)/(2P)$ times the Gaussian-field Fisher.

Figure: `test/figures/fisher_field_linear_check_res80.png`
