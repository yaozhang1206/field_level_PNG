# Stage 2.1: NUTS on the $16^3$ linear box vs exact marginal

- LRG ($\bar n=3.0e-04$), 1500 samples after 300 warmup: $\sigma(f_{\rm NL})$ chain 0.804 (exact marginal 0.614), mean 1.094, $\tau_{\rm int}$ 3.7, ESS 404; mean leapfrog/sample 31, accept 0.85; 12s
- QSO ($\bar n=2.5e-05$), 1500 samples after 300 warmup: $\sigma(f_{\rm NL})$ chain 22.234 (exact marginal 26.730), mean 35.147, $\tau_{\rm int}$ 2.9, ESS 509; mean leapfrog/sample 31, accept 0.86; 12s

Posterior width must match the exact Gaussian marginal; ESS per gradient sets the cost baseline.

Figure: `test/figures/sampler16_linear.png`
