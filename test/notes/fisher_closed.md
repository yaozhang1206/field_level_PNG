# Part B closed forms: Tier 1 ($P$) and Tier 1+2 ($P+B$) field-level Fisher

Shells (tracer, $z$ range: $b_1$, $b_2$, $b_K$, $b_\phi$, $b_{\phi\delta}$): QSO 0.8–1.3: 1.77, -0.43, -0.22, 0.56, -1.42; QSO 1.3–2.1: 2.50, 0.98, -0.43, 3.03, 2.23; QSO 2.1–3.1: 3.84, 6.35, -0.81, 7.56, 13.60; LRG 0.4–0.6: 1.89, -0.27, -0.25, 2.99, 1.07; LRG 0.6–0.8: 2.02, -0.06, -0.29, 3.44, 1.66; LRG 0.8–1.1: 2.21, 0.30, -0.35, 4.08, 2.60
Volumes: full DR1 footprint (9070 deg$^2$), shot noise from observed density $\bar c\,{\rm NX}$; $k_{\min} = 2\pi/L$ unless stated.
Tier 1 = Gaussian mode sum over the grid cube (Kaiser RSD; parameters $f_{\rm NL}$, $b_1$ per shell [+$b_\phi$]):
$$F_{ij} = \sum_{\rm shells}\int dV(z)\int\frac{d^3k}{(2\pi)^3}\,\frac{\partial_i P_g\,\partial_j P_g}{2\,[P_g(k,\mu) + 1/n_{\rm obs}(z)]^2},\qquad P_g = [b_1 + b_\phi f_{\rm NL}/T(k,z) + f\mu^2]^2 P_{\rm lin}(k,z).$$
Tier 2 = tree-level real-space bispectrum, $k < k_{\rm Ny}$ sphere, $\Delta k=0.002$, Gaussian covariance; adds $b_2$, $b_K$, $b_{\phi\delta}$:
$$B_g = c_1c_2c_3\,B_{\rm PNG} + \sum_{\rm cyc} c_1 c_2\left[2b_1F_2 + b_2 + 2b_K S_2 + b_{\phi\delta} f_{\rm NL}\left(\tfrac{1}{T_1}+\tfrac{1}{T_2}\right)\right]P_1P_2,\quad c_i = b_1 + \frac{b_\phi f_{\rm NL}}{T_i},\quad {\rm Var}[B] = \frac{s_B (2\pi)^6\,\tilde P_1\tilde P_2\tilde P_3}{8\pi^2 V k_1k_2k_3\Delta k^3}.$$
Marginalised over all bias parameters; '$b_\phi$ sampled' = 40% Gaussian prior on $b_\phi$ and $b_{\phi\delta}$ per shell.
NOTE: at fiducial $f_{\rm NL} = 0$ the Fisher is blind to the $b_\phi$ prior ($b_\phi$ enters only as $b_\phi f_{\rm NL}$), so 'sampled' = 'fixed' here;
propagating $f_{\rm NL} = A/b_\phi$ with a 40% prior on $b_\phi$ inflates $\sigma$ by $\approx\sqrt{1+3\cdot0.4^2} = 1.22$ (see memo).

| grid ($k_{\rm Ny}$) | tracers | $b_\phi$ | Tier 1 ($P$ only) | Tier 1 + Tier 2 ($P+B$) | Tier 1, $k_{\min}=0.003$ | Tier 2 only |
|---|---|---|---|---|---|---|
| $80^3$ (0.025) | QSO | fixed | 8.4 | 8.4 | 11.5 | 383.8 |
| $80^3$ (0.025) | LRG | fixed | 7.1 | 7.1 | 14.1 | 285.3 |
| $80^3$ (0.025) | QSO+LRG | fixed | 5.4 | 5.4 | 8.9 | 229.0 |
| $80^3$ (0.025) | QSO | 40% prior | 8.4 | 8.4 | 11.5 | 383.8 |
| $80^3$ (0.025) | LRG | 40% prior | 7.1 | 7.1 | 14.1 | 285.3 |
| $80^3$ (0.025) | QSO+LRG | 40% prior | 5.4 | 5.4 | 8.9 | 229.0 |
| $160^3$ (0.050) | QSO | fixed | 7.5 | 7.4 | 9.5 | 61.7 |
| $160^3$ (0.050) | LRG | fixed | 6.7 | 6.6 | 11.9 | 37.2 |
| $160^3$ (0.050) | QSO+LRG | fixed | 5.0 | 4.9 | 7.4 | 31.9 |
| $160^3$ (0.050) | QSO | 40% prior | 7.5 | 7.4 | 9.5 | 61.7 |
| $160^3$ (0.050) | LRG | 40% prior | 6.7 | 6.6 | 11.9 | 37.2 |
| $160^3$ (0.050) | QSO+LRG | 40% prior | 5.0 | 4.9 | 7.4 | 31.9 |

Caveats: Tier 2 ignores RSD and the $P$–$B$ cross-covariance; Tier 1 with $k_{\min} = k_f$ counts modes larger than a shell.
Figure: fisher_closed_kmax.png ($\sigma$ vs $k_{\max}$, QSO+LRG, $160^3$). Runtime 21s.

Figure: `test/figures/fisher_closed.png`
