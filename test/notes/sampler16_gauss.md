# Sampler mechanics (exact Gaussian target) and multi-mock $f_{\rm NL}$ width at $16^3$

- Gaussian target (eps only, 4096 dims), 3000 NUTS samples: mean error / posterior std: rms 0.013 (expect ~0.013 from ESS);
  variance ratio chain/exact: mean 1.000, 5-95% [0.935, 1.067]; random projections: var ratio [0.979, 0.976, 0.918, 0.971, 0.968], tau [0.5, 0.5, 0.5, 0.5, 0.5]; leapfrog/sample 31, 23s
- f_NL width over 6 mocks (LRG, b1 free): chain std per mock [0.685, 1.257, 0.656, 0.648, 0.752, 0.623], mean 0.770 ± 0.090; exact-marginal Fisher 0.638; posterior means [-0.09, -1.48, -0.61, 0.19, 0.94, -0.6] (truth 0; rms 0.800)

No figure (numerical table).

Figure: `test/figures/sampler16_gauss.png`
