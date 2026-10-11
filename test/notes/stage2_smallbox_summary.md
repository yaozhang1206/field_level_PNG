# Stage 2.1 summary: is the quadratic-coupling information real?

Same mock ($f_{\rm NL,true}=0$, Gaussian noise, periodic box, production cell size 125 Mpc/$h$) sampled with NUTS under the
strictly linear model and under the full 2LPT model ($b_1,b_2,b_K$ free). Fisher: Tier 1 exact mode sum, exact marginal of the
linearised 2LPT model (Hutchinson), Schur. Chain widths carry $\approx 3$–5 % sampling error (ESS 270–560).

| box | linear chain | full chain | ratio | Tier 1 | linearised Fisher | ratio | Schur (full) | leapfrog/sample lin / full |
|---|---|---|---|---|---|---|---|---|
| $16^3, L=2000, LRG$ | 13.30 | 9.34 | 1.42 | 11.19 | 8.64 | 1.30 | 9.01 | 31 / 127 |
| $32^3, L=4000, QSO+LRG$ | 3.01 | 2.35 | 1.28 | 2.82 | 2.01 | 1.40 | 2.13 | 31 / 255 |
| $32^3, L=4000, LRG$ | 3.17 | 2.47 | 1.28 | 2.83 | 2.06 | 1.37 | 2.21 | 31 / 255 |

Conclusion: the posterior of the full model is narrower than that of the linear model by a factor 1.28–1.42 in $\sigma$, close to the
1.30–1.40 predicted by the linearised Fisher. The quadratic-coupling information is realised, at $\gtrsim 85$ % of the Fisher
prediction in $\sigma$. Scaled to DR1 ($80^3$, linear model with window 8.3): expect the full-model posterior near 6–6.5 at $80^3$;
the $80^3$ production chains (queued) measure this directly. Cost: the full model needs $\approx 8\times$ more leapfrog steps per NUTS sample.

Figure: `test/figures/stage2_smallbox_summary.png`
