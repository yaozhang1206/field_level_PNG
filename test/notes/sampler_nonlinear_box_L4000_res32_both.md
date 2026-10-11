# Stage 2.1: posterior width, linear vs full 2LPT model (small box, production cell size)

box $L=4000$, $32^3$ (cell 125 Mpc/h, $k_{\rm Ny}=0.0251$), tracers ['QSO', 'LRG'], $\bar n$ {'QSO': 2.5e-05, 'LRG': 0.0003}, Gaussian noise, $f_{\rm NL,true}=0$
- linear model: NUTS $\sigma(f_{\rm NL})$ = **3.014** (mean 1.263, $\tau$ 6.3, ESS 473, 40s, leapfrog 31/sample) | Fisher: Tier 1 exact 2.823, Hutchinson(linearised) 2.796, Schur 3.273
- full model: NUTS $\sigma(f_{\rm NL})$ = **2.350** (mean -0.150, $\tau$ 5.3, ESS 563, 2076s, leapfrog 255/sample) | Fisher: Tier 1 exact 2.823, Hutchinson(linearised) 2.015, Schur 2.126

If the full-model chain width tracks Tier 1 rather than the linearised Fisher, the quadratic-coupling information is not realised in the posterior.

Figure: `test/figures/sampler_nonlinear_box_L4000_res32_both.png`
