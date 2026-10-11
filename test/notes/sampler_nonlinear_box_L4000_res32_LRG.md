# Stage 2.1: posterior width, linear vs full 2LPT model (small box, production cell size)

box $L=4000$, $32^3$ (cell 125 Mpc/h, $k_{\rm Ny}=0.0251$), tracers ['LRG'], $\bar n$ {'LRG': 0.0003}, Gaussian noise, $f_{\rm NL,true}=0$
- linear model: NUTS $\sigma(f_{\rm NL})$ = **3.167** (mean 1.470, $\tau$ 8.4, ESS 357, 36s, leapfrog 31/sample) | Fisher: Tier 1 exact 2.825, Hutchinson(linearised) 2.841, Schur 3.617
- full model: NUTS $\sigma(f_{\rm NL})$ = **2.471** (mean -0.777, $\tau$ 7.6, ESS 397, 1961s, leapfrog 255/sample) | Fisher: Tier 1 exact 2.825, Hutchinson(linearised) 2.056, Schur 2.209

If the full-model chain width tracks Tier 1 rather than the linearised Fisher, the quadratic-coupling information is not realised in the posterior.

Figure: `test/figures/sampler_nonlinear_box_L4000_res32_LRG.png`
