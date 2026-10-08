# DISCO-EB vs CLASS linear power $P_{\rm lin}(k)$

Both backends use the same $P_\Phi(k)$ and growth; only $T(k)$ differs (normalised to 1 at $k_{\min}$).
DISCO-EB: RECFAST, $\ell_{\max}=11$, $n_{q,\max}=3$, rtol $=10^{-5}$; solver precision/$\ell_{\max}$/$N_{\rm eff}$ changes move the result by < 0.1%.

- $m_\nu = 0.06$ eV: $\sigma_8$ DISCO-EB 0.8157 vs CLASS 0.8117; $\max|\Delta P/P|$ = 1.14% ($k\le0.1$), 1.45% ($k\le0.3$); time 2s / 0s
- $m_\nu = 0$ (massless): $\sigma_8$ DISCO-EB 0.8244 vs CLASS 0.8261; $\max|\Delta P/P|$ = 0.46% ($k\le0.1$), 0.72% ($k\le0.3$); time 0s / 0s

Result: FAIL against the 0.5% target (worst 1.14% at $k\le0.1$).
A $\sim1\%$ $P_{\rm lin}$ offset changes $\sigma(f_{\rm NL})$ by $\sim0.5\%$ and is irrelevant for the Stage-1 go/no-go.
Decision: Part A uses CLASS (fast, standard). DISCO-EB stays available for Stage 2 consistency with DISCO-DJ.

Figure: `test/figures/cosmology_discoeb_vs_class.png`
