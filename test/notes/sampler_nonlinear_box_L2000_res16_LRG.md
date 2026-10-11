# Stage 2.1: posterior width, linear vs full 2LPT model (small box, production cell size)

box $L=2000$, $16^3$ (cell 125 Mpc/h, $k_{\rm Ny}=0.0251$), tracers ['LRG'], $\bar n$ {'LRG': 0.0003}, Gaussian noise, $f_{\rm NL,true}=0$
- linear model: NUTS $\sigma(f_{\rm NL})$ = **13.301** (mean 20.741, $\tau$ 9.2, ESS 65, 6s, leapfrog 31/sample) | Fisher: Tier 1 exact 11.192, Hutchinson(linearised) 11.216, Schur 13.640
- full model: NUTS $\sigma(f_{\rm NL})$ = **9.339** (mean 11.438, $\tau$ 2.2, ESS 271, 575s, leapfrog 127/sample) | Fisher: Tier 1 exact 11.192, Hutchinson(linearised) 8.637, Schur 9.006

If the full-model chain width tracks Tier 1 rather than the linearised Fisher, the quadratic-coupling information is not realised in the posterior.

Figure: `test/figures/sampler_nonlinear_box_L2000_res16_LRG.png`
