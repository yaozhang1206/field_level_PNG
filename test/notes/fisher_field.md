# Part B: field-level Fisher results (Schur) vs closed forms

Field-level Fisher forecasts (DISCO-DJ 2LPT + PNG + Lagrangian bias + RSD, DR1 footprint, Gaussian noise).
Two estimators: Schur complement of the joint $(\varepsilon,\theta)$ Fisher, $F_{\rm marg} = F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta}$ (profile curvature; biased by $(P+N)/(2P)$ per mode, see
schur_dense16.md) and the exact Gaussian-marginal Fisher $F_{ij} = \tfrac12{\rm Tr}[C^{-1}\partial_iC\,C^{-1}\partial_jC]$ of the model linearised in $\varepsilon$ (Hutchinson trace estimator).
Marginalised over per-shell $(b_1, b_2, b_K)$ [+ $b_\phi$, $b_{\phi\delta}$ with 40% prior when sampled]; fiducial $f_{\rm NL} = 0$ (5 when $b_\phi$ sampled).
Closed forms from fisher_closed.md (same shells/volumes). 'unmarg.' = $1/\sqrt{F_{ff}}$ with all bias fixed.

| run | tracers | $n_{\rm real}$ | $\sigma(f_{\rm NL})$ Schur: mean $\pm$ scatter | $\sigma$ from mean $F$ | unmarg. $f_{\rm NL}$ | closed Tier 1 ($k_f$ / 0.003) | Tier 1+2 |
|---|---|---|---|---|---|---|---|
| res80_p1.6_bphi_fixed_fnl0 | QSO+LRG | 4 | 3.93 $\pm$ 0.44 | 3.84 | 3.42 | 5.4 / 8.9 | 5.4 |
| res80_p1.6_bphi_fixed_fnl0 | QSO | 4 | 4.74 $\pm$ 0.56 | 4.62 | 4.07 | 8.4 / 11.5 | 8.4 |
| res80_p1.6_bphi_fixed_fnl0 | LRG | 4 | 7.33 $\pm$ 0.80 | 7.19 | 6.53 | 7.1 / 14.1 | 7.1 |

| exact-marginal run | tracers | $n_{\rm real}$ | $\sigma(f_{\rm NL})$ marginal (per realisation) | from mean $F$ | Schur, same runs | closed Tier 1 exact ($k_f$ / 0.003) |
|---|---|---|---|---|---|---|
| marginal_res80_p1.0 | QSO+LRG | 1 | 4.36 $\pm$ 0.23 (16 probes) | 4.36 | 2.47 | 5.4 / 8.9 |
| marginal_res80_p1.6 | QSO+LRG | 2 | 5.40 $\pm$ 0.31 (24 probes), 6.00 $\pm$ 0.34 (12 probes) | 5.67 | 3.18, 4.05 | 5.4 / 8.9 |
| marginal_res80_p1.6_single | QSO | 1 | 8.99 $\pm$ 0.12 (24 probes) | 8.99 | 3.77 | 8.4 / 11.5 |
| marginal_res80_p1.6_single | LRG | 1 | 9.42 $\pm$ 0.41 (24 probes) | 9.42 | 6.20 | 7.1 / 14.1 |

Realisation scatter reflects the few largest-scale modes that dominate the PNG signal; use the mean-F value.
Figure: fisher_field_summary.png

Figure: `test/figures/fisher_field.png`
