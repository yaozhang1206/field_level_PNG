# Stage 2.0: log-posterior gradient and Hessian checks ($16^3$)

- gaussian: mock counts mean 39673.03; grad check eps: AD 5.968213e-01 vs FD 5.968214e-01 (rel 1.5e-07); f_NL: AD -1.006678e+00 vs FD -1.006678e+00 (rel 1.8e-08)
- Gaussian, noise-free data: max |Hessian - F_tt| / max|F_tt| = 3.27e-15 (expect ~1e-8)
- whitened logdensity on flat state (n=4103): value -6124.886, grad norm 1.639e+02, 2.9s incl. compile
- poisson: mock counts mean 39675.78; grad check eps: AD 4.267116e+00 vs FD 4.270077e+00 (rel 6.9e-04); f_NL: AD 5.588013e-01 vs FD 5.578995e-01 (rel 1.6e-03)

No figure (numerical table).

Figure: `test/figures/posterior16.png`
