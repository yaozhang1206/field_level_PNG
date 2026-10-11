# Stage 2.1: posterior width, linear vs full 2LPT model (small box, production cell size)

box $L=2000$, $16^3$ (cell 125 Mpc/h, $k_{\rm Ny}=0.0251$), tracers ['LRG'], $\bar n$ {'LRG': 0.0003}, Gaussian noise, $f_{\rm NL,true}=0$; spherical window radius 0.4L (27% of voxels)
- linear model: NUTS $\sigma(f_{\rm NL})$ = **37.356** (mean -6.708, $\tau$ 20.2, ESS 40, 8s, leapfrog 31/sample) | Fisher: Tier 1 exact 21.717, Hutchinson(linearised) 26.745, Schur 26.635
- full model: NUTS $\sigma(f_{\rm NL})$ = **23.922** (mean -16.333, $\tau$ 12.5, ESS 64, 452s, leapfrog 63/sample) | Fisher: Tier 1 exact 21.717, Hutchinson(linearised) 21.320, Schur 19.476

If the full-model chain width tracks Tier 1 rather than the linearised Fisher, the quadratic-coupling information is not realised in the posterior.

Figure: `test/figures/sampler_nonlinear_box_L2000_res16_LRG_mask0.4.png`
