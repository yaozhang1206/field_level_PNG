# Stage 2: DR1 $80^3$ posterior widths, linear vs full 2LPT model

DR1 configuration, $80^3$ ($k_{\rm Ny}=0.025$), QSO+LRG, 6 shells, Gaussian noise, $f_{\rm NL,true}=0$, NUTS started at the truth;
$b_1$ free (linear) or $b_1, b_2, b_K$ free per shell with flat priors (full); $b_\phi$, $b_{\phi\delta}$ fixed (UMF).
Fisher column: exact-marginal Fisher (Hutchinson) of Stage 1 for the corresponding model (same seed-0 field).

| chain | samples | $\sigma(f_{\rm NL})$ | mean | $\tau_{\rm int}$ | ESS | leapfrog/sample | Fisher |
|---|---|---|---|---|---|---|---|
| seed 0, linear | 3000 | 7.77 ± 0.33 | 2.3 | 11.1 | 269 | 63 | 8.28 |
| seed 0, full 2LPT (chain A) | 3000 | 6.97 ± 0.23 | 2.4 | 6.4 | 472 | 255 | 5.32 |
| seed 0, full 2LPT (chain B) | 2500 | 7.16 ± 0.21 | 2.1 | 4.1 | 608 | 255 | 5.32 |
| seed 1, linear | 3000 | 7.45 ± 0.28 | -1.9 | 8.4 | 358 | 63 | 8.28 |
| seed 1, full 2LPT | 2500 | 7.51 ± 0.20 | -6.0 | 3.4 | 730 | 255 | 5.32 |

Result: at DR1 scale the full 2LPT model gives $\sigma(f_{\rm NL}) = 7.0$–7.5 at $80^3$ and $5.6 \pm 0.3$ at $160^3$ (pooled two chains; linear 6.5), a gain of 10–14 %
posterior, versus 36 % predicted by the linearised Fisher and 23–30 % realised in periodic boxes. With the 40 % $b_\phi$ prior
propagated ($\times1.22$): $\approx 8.5$–9. Mixing: $\tau_{\rm int} = 3$–11 samples; 2 s per sample (255 leapfrog × 14 ms) on one GH200.

Figure: `test/figures/stage2_dr1_chains80.png`
