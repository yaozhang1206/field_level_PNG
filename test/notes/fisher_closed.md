# Part B closed forms: Tier 1 (P) and Tier 1+2 (P+B) field-level Fisher

Shells (tracer, z range, b1, b2, bK, bphi, bphidelta): QSO 0.8-1.3: 1.77, -0.43, -0.22, 0.56, -1.42; QSO 1.3-2.1: 2.50, 0.98, -0.43, 3.03, 2.23; QSO 2.1-3.1: 3.84, 6.35, -0.81, 7.56, 13.60; LRG 0.4-0.6: 1.89, -0.27, -0.25, 2.99, 1.07; LRG 0.6-0.8: 2.02, -0.06, -0.29, 3.44, 1.66; LRG 0.8-1.1: 2.21, 0.30, -0.35, 4.08, 2.60
Volumes: full DR1 footprint (9070 deg^2), shot noise from observed density c_bar*NX; k_min = 2pi/L unless stated.
Tier 1 = Gaussian mode sum over the grid cube (Kaiser RSD, parameters f_NL, b1 per shell [+bphi]).
Tier 2 = tree-level real-space bispectrum, k < k_Ny sphere, Dk=0.002, Gaussian covariance; adds b2, bK, bphidelta.
Marginalised over all bias parameters; 'bphi sampled' = 40% Gaussian prior on bphi and bphidelta per shell.
NOTE: at fiducial f_NL = 0 the Fisher is blind to the bphi prior (bphi enters only as bphi*f_NL), so 'sampled' = 'fixed' here;
propagating f_NL = A/bphi with a 40% prior on bphi inflates sigma by ~sqrt(1+3*0.4^2) = 1.22 (see memo).

| grid (k_Ny) | tracers | bphi | Tier1 (P only) | Tier1+Tier2 (P+B) | Tier1 kmin=0.003 | Tier2-only |
|---|---|---|---|---|---|---|
| 80^3 (0.025) | QSO | fixed | 8.4 | 8.4 | 11.5 | 383.8 |
| 80^3 (0.025) | LRG | fixed | 7.1 | 7.1 | 14.1 | 285.3 |
| 80^3 (0.025) | QSO+LRG | fixed | 5.4 | 5.4 | 8.9 | 229.0 |
| 80^3 (0.025) | QSO | 40% prior | 8.4 | 8.4 | 11.5 | 383.8 |
| 80^3 (0.025) | LRG | 40% prior | 7.1 | 7.1 | 14.1 | 285.3 |
| 80^3 (0.025) | QSO+LRG | 40% prior | 5.4 | 5.4 | 8.9 | 229.0 |
| 160^3 (0.050) | QSO | fixed | 7.5 | 7.4 | 9.5 | 61.7 |
| 160^3 (0.050) | LRG | fixed | 6.7 | 6.6 | 11.9 | 37.2 |
| 160^3 (0.050) | QSO+LRG | fixed | 5.0 | 4.9 | 7.4 | 31.9 |
| 160^3 (0.050) | QSO | 40% prior | 7.5 | 7.4 | 9.5 | 61.7 |
| 160^3 (0.050) | LRG | 40% prior | 6.7 | 6.6 | 11.9 | 37.2 |
| 160^3 (0.050) | QSO+LRG | 40% prior | 5.0 | 4.9 | 7.4 | 31.9 |

Caveats: Tier 2 ignores RSD and the P-B cross-covariance; Tier 1 k_min = k_f counts modes larger than a shell.
Figure: fisher_closed_kmax.png (sigma vs k_max, QSO+LRG, 160^3). Runtime 25s.

Figure: `test/figures/fisher_closed.png`
