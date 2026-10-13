# Stage 4: exact template-deprojection Fisher loss vs dense brute force

Periodic box $L = 2000$, $8^3$, LRG-like, full 2LPT model linearised at a random field; 4 smooth random templates.

| deprojected | $\Delta F_{f f}$ (CG + Woodbury) | $\Delta F_{f f}$ (dense) | max rel. diff (all $ij$) | $\sigma(f_{\rm NL})$ dense |
|---|---|---|---|---|
| none | 0.0000e+00 | 0.0000e+00 | 0.0e+00 | 11.679 |
| T0 | 6.0125e-04 | 6.0125e-04 | 3.6e-13 | 12.084 |
| T01 | 1.0286e-03 | 1.0286e-03 | 1.9e-12 | 12.388 |
| T0123 | 1.8425e-03 | 1.8425e-03 | 1.7e-12 | 13.030 |

**PASS** (criterion: max relative difference < 1e-4).

Figure: `test/figures/deproj_dense.png`
