# Stage 3: f_NL recovery on Quijote halo ensembles (model-mismatch test)

Quijote halo ensembles, DR1 forward model (2LPT, Lagrangian bias + $b_\nabla$, Kaiser RSD, $b_\phi$ from UMF with the DR1 $p$),
$16^3$ grid (62.5 Mpc/h), 4× CIC-deconvolved Fourier-truncated assignment for data and model, Poisson-level Gaussian noise, cold start.
Pull = (posterior mean − truth)/posterior σ per box; a calibrated posterior has pull mean 0 and s.d. 1.

| run | N | truth | $\langle f_{\rm NL}\rangle$ | scatter | $\langle\sigma\rangle$ | pull mean | pull s.d. | $\langle b_1\rangle$ | $\chi^2$/dof |
|---|---|---|---|---|---|---|---|---|---|
| LC_m_z0.5_M2e13_res16_p1m4d_lpt2_rsd_lap_p1_M6e+13 | 20 | -100 | -94.4 ± 13.1 | 59 | 65 | +0.04 | 0.92 | 3.17 | nan |
| LC_p_z0.5_M2e13_res16_p1m4d_lpt2_rsd_lap_p1_M6e+13 | 20 | +100 | +119.7 ± 16.7 | 75 | 70 | +0.19 | 1.06 | 2.97 | nan |
| fiducial_HR_z0.5_LRG_res16_p1m4d_lpt2_rsd_lap_p1 | 26 | +0 | -5.1 ± 16.2 | 83 | 62 | -0.24 | 1.45 | 1.92 | nan |
| fiducial_HR_z1_QSO_b2.44_res16_p1m4d_lpt2_rsd_lap_p1.6 | 20 | +0 | +33.6 ± 21.5 | 96 | 82 | +0.30 | 1.10 | 2.41 | nan |
| fiducial_HR_z1_QSO_b2.9_res16_p1m4d_lpt2_rsd_lap_p1.6 | 20 | +0 | +34.9 ± 19.6 | 87 | 68 | +0.37 | 1.17 | 2.82 | nan |
| fiducial_z0.5_M2e13_res16_p1m4d_lpt2_rsd_lap_p1_M6e+13 | 20 | +0 | +19.7 ± 13.9 | 62 | 63 | +0.25 | 0.94 | 3.07 | nan |

Pooled $f_{\rm NL} = 0$ halo samples (2LPT, $16^3$): $\langle f_{\rm NL}\rangle = +17.9 \pm 8.5$.

Figure: `test/figures/stage3_ensembles.png`
