# Part A: DESI DR1 $P(k)$ Fisher vs Chaussidon et al. 2024

Gaussian Fisher, $P_0+P_2$, $\Delta k=0.001$, FKP-weighted covariance with the DR1 $n(z)$ (NX column); nz_kind=nx
('full': sample variance over the geometric DR1 footprint, shot noise from the observed density $\bar c\,{\rm NX}$;
 'nx': everything over the completeness-weighted effective area).
Covariance: ${\rm Cov}[P_\ell,P_{\ell'}](k) = (2\ell+1)(2\ell'+1)\,\frac{4\pi^2}{k^2\Delta k}\int_0^1 d\mu\, L_\ell L_{\ell'}\, S(k,\mu)$, $S = \int dV\, n^4 w^4 (P_g + 1/n_{\rm obs})^2 / (\int dV\, n^2 w^2)^2$, $w = 1/(1+nP_0)$.
Parameters $(f_{\rm NL}, b_1, s_{n0}, \Sigma_s)$; fiducial $f_{\rm NL}=0$, $s_{n0}=0$, $\Sigma_s=3$ Mpc/$h$. Cosmology Planck 2018 via CLASS.
Geometry: LRG: $V_{\rm eff}=6.20$ (Gpc/$h$)$^3$, $\bar n_{\rm eff}=3.04e-04$, $A_{\rm eff}=5417$ deg$^2$, $A_{\rm full}=9070$ deg$^2$, $\bar c=0.60$; QSO: $V_{\rm eff}=8.29$ (Gpc/$h$)$^3$, $\bar n_{\rm eff}=2.48e-05$, $A_{\rm eff}=7159$ deg$^2$, $A_{\rm full}=9070$ deg$^2$, $\bar c=0.79$

| config | $\sigma(f_{\rm NL})$ marg. | $b_1$ fixed | $\Sigma_s$ fixed | $\Sigma_s$ prior 2 | $\sigma(b_\phi f_{\rm NL})$ | DESI clean-mock | notes |
|---|---|---|---|---|---|---|---|
| QSO_FKP | 15.0 | 11.5 | 14.8 | 15.0 | 42.6 | DESI 13-18 (mean 15.5) | $z_{\rm eff}=1.651$, $b_1=2.44$, $b_\phi=2.83$ |
| QSO_OQE_singlez | 8.4 | 6.5 | 8.3 | 8.4 | 42.0 | DESI 10-13 (mean 11.5) | $z_{\rm eff}=2.082$, $b_1=3.08$, $b_\phi=4.99$ |
| LRG_FKP | 24.4 | 16.0 | 23.7 | 24.3 | 87.4 | DESI 11-18 (mean 14.5) | $z_{\rm eff}=0.733$, $b_1=2.06$, $b_\phi=3.57$ |
| LRG_FKP_kmin003 | 15.8 | 12.5 | 15.5 | 15.7 | 56.3 | DESI 11-18 (mean 14.5) | $z_{\rm eff}=0.733$, $b_1=2.06$, $b_\phi=3.57$ |
| QSO_shells | 9.2 | 7.1 | 9.2 | 9.2 | nan | DESI 10-13 (mean 11.5) | 10 shells, $b_1(z)$ template, $p=1.6$ |
| LRG_shells | 21.8 | 15.1 | 21.2 | 21.7 | nan | - | 10 shells, $b_1(z)$ template, $p=1.0$ |
| QSO+LRG_FKP | 12.8 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_FKP+LRG_FKP |
| QSO+LRG_OQE_singlez | 8.0 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_OQE_singlez+LRG_FKP |
| QSO+LRG_shells | 8.5 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_shells+LRG_shells |

DESI numbers are the asymmetric EZmock errors (Tables 4-5); the Fisher is symmetric, target is ~20% of the mean.
'OQE_singlez' evaluates the model at the OQE $z_{\rm eff}$/$b_1$ but keeps the FKP $n(z)$ weighting in the covariance;
'shells' sums 10 redshift shells with the $b_1(z)$ template and is the optimal-weighting (OQE) limit.
Figures: fisher_pk_kcuts_nx.png ($\sigma$ vs $k_{\max}$ / $k_{\min}$), fisher_pk_info_density_nx.png (where the signal is).

Figure: `test/figures/fisher_pk_nx.png`
