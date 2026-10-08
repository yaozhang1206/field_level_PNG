# Part A: DESI DR1 P(k) Fisher vs Chaussidon et al. 2024

Gaussian Fisher, P0+P2, Dk=0.001, FKP-weighted covariance with the DR1 n(z) (NX column); nz_kind=full
('full': sample variance over the geometric DR1 footprint, shot noise from the observed density c_bar*NX;
 'nx': everything over the completeness-weighted effective area).
Parameters (f_NL, b1, s_n0, Sigma_s); fiducial f_NL=0, s_n0=0, Sigma_s=3 Mpc/h. Cosmology Planck 2018 via CLASS.
Geometry: LRG: V_eff=6.20 (Gpc/h)^3, nbar_eff=3.04e-04, area_eff=5417 deg^2, area_full=9070 deg^2, c_bar=0.60; QSO: V_eff=8.29 (Gpc/h)^3, nbar_eff=2.48e-05, area_eff=7159 deg^2, area_full=9070 deg^2, c_bar=0.79

| config | sigma(f_NL) marg. | b1 fixed | Sigma_s fixed | Sigma_s prior 2 | sigma(b_phi f_NL) | DESI clean-mock | notes |
|---|---|---|---|---|---|---|---|
| QSO_FKP | 15.5 | 11.8 | 15.3 | 15.5 | 44.0 | DESI 13-18 (mean 15.5) | z_eff=1.651, b1=2.44, b_phi=2.83 |
| QSO_OQE_singlez | 8.6 | 6.6 | 8.5 | 8.6 | 43.1 | DESI 10-13 (mean 11.5) | z_eff=2.082, b1=3.08, b_phi=4.99 |
| LRG_FKP | 20.0 | 13.1 | 19.4 | 20.0 | 71.6 | DESI 11-18 (mean 14.5) | z_eff=0.733, b1=2.06, b_phi=3.57 |
| LRG_FKP_kmin003 | 13.1 | 10.4 | 12.9 | 13.1 | 46.8 | DESI 11-18 (mean 14.5) | z_eff=0.733, b1=2.06, b_phi=3.57 |
| QSO_shells | 9.6 | 7.4 | 9.6 | 9.6 | nan | DESI 10-13 (mean 11.5) | 10 shells, b1(z) template, p=1.6 |
| LRG_shells | 18.0 | 12.5 | 17.5 | 18.0 | nan | - | 10 shells, b1(z) template, p=1.0 |
| QSO+LRG_FKP | 12.3 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_FKP+LRG_FKP |
| QSO+LRG_OQE_singlez | 7.9 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_OQE_singlez+LRG_FKP |
| QSO+LRG_shells | 8.5 | - | - | - | - | DESI 7.9-10.5 (mean 9.2) | QSO_shells+LRG_shells |

DESI numbers are the asymmetric EZmock errors (Tables 4-5); the Fisher is symmetric, target is ~20% of the mean.
'OQE_singlez' evaluates the model at the OQE z_eff/b1 but keeps the FKP n(z) weighting in the covariance;
'shells' sums 10 redshift shells with the b1(z) template and is the optimal-weighting (OQE) limit.
Figures: fisher_pk_kcuts_full.png (sigma vs k_max / k_min), fisher_pk_info_density_full.png (where the signal is).

Figure: `test/figures/fisher_pk_full.png`
