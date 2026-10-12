# Stage 3 data: Quijote Rockstar halo samples for the model-mismatch tests

Date 2026-10-12. Code: `src/quijote.py`, `scripts/stage3_cut_quijote.py`, `scripts/stage3_fetch_quijote.sh`.
Figure: `test/figures/stage3_quijote_cuts.png`. Cuts: `/work/hdd/bdne/yzhang116/quijote_cuts/<set>_z<z>_<tag>/{i}.npz` + `summary.json`.

## Procedure

Globus transfer (Quijote_simulations2 → NCSA Delta) of `Halos/Rockstar/<set>/<i>/out_<snap>_pid.list` in batches of
50, host halos only (PID = −1), mass cut, compact `.npz` (pos, vel, mass), raw file deleted. Raw never exceeds ≈ 10 GB
on disk; the cuts total 1.6 GB. The linear bias of each cut is measured from the halo auto-spectrum,
$b_1^2 = [P_{hh}(k) - 1/\bar n]/P_{\rm lin}(k,z)$ over $0.02 < k < 0.06\,h/{\rm Mpc}$ (CIC $128^3$ mesh, deconvolved), with
$P_{\rm lin}$ from CLASS at the Quijote cosmology ($\Omega_m = 0.3175$, $h = 0.6711$, $n_s = 0.9624$, $\sigma_8 = 0.834$).
Mass thresholds for a target $b_1$ are found by bisection on the first three realisations and then held fixed.

The catalogues already on disk (`halo_catalogs_quijote_all/CV`, ASCII) are the **fiducial_HR** Rockstar hosts at $z = 0.5$:
positions match the Globus `fiducial_HR/0/out_3_pid.list` one to one; their mass column is 0.89 × Mvir (a different
overdensity definition), which shifts the equivalent threshold by 10 % and nothing else.

## Samples

| directory | set | z | selection | N real. | $b_1$ | $\bar n$ [(Mpc/h)$^{-3}$] | role |
|---|---|---|---|---|---|---|---|
| fiducial_HR_z0.5_LRG | HR, $f_{\rm NL}=0$ | 0.5 | $M > 8.9\times10^{12}$ | 100 | 1.87 ± 0.02 | 3.6e-4 | LRG-like, bias at $f_{\rm NL} = 0$ |
| fiducial_HR_z1_QSO_b2.44 | HR | 1 | $M > 7.0\times10^{12}$ | 50 | 2.41 ± 0.03 | 3.6e-4 | QSO-like ($b_1$ of DESI QSO at $z_{\rm eff}$) |
| fiducial_HR_z1_QSO_b2.9 | HR | 1 | $M > 1.4\times10^{13}$ | 50 | 2.86 ± 0.04 | 1.5e-4 | QSO-like ($b_1$ of the $z \approx 2$ shell) |
| fiducial_HR_z2_all | HR | 2 | all hosts | 50 | 4.26 ± 0.07 | 1.7e-4 | high-bias stress sample |
| LC_p_z0.5_M2e13 | std, $f_{\rm NL}=+100$ | 0.5 | $M > 2\times10^{13}$ | 50 | 3.02 ± 0.05 | 4.9e-5 | response test |
| LC_m_z0.5_M2e13 | std, $f_{\rm NL}=-100$ | 0.5 | $M > 2\times10^{13}$ | 50 | 2.92 ± 0.05 | 4.8e-5 | response test |
| fiducial_z0.5_M2e13 | std, $f_{\rm NL}=0$ | 0.5 | $M > 2\times10^{13}$ | 59 | 2.97 ± 0.05 | 4.9e-5 | control for the response test |

(The LC_p / LC_m "$b_1$" includes the scale-dependent bias at $k < 0.06$; the ±0.05 split about the fiducial is the
$f_{\rm NL} = \pm100$ signal.)

## Findings

1. **At $z = 2$ the DESI QSO bias (2.9) is not reachable**: the smallest resolved HR halos ($\approx 4\times10^{12}$) already
   have $b_1 \approx 4.3$. The QSO-like samples are therefore taken at $z = 1$, matching $b_1$ and $\bar n$ rather than
   redshift; the full $z = 2$ sample is kept as a high-bias stress test.
2. **The standard-resolution Rockstar catalogues (used for the PNG sets) are incomplete below $\approx 6\times10^{13}$**
   (right panel): realisation 5 has 49k hosts vs 693k in HR, and $\bar n(M > 2\times10^{13})$ is 3× lower, while masses of
   matched halos agree to 1 % above $3\times10^{13}$. The response test must use $M \gtrsim 6\times10^{13}$
   ($\bar n \approx 3\times10^{-5}$, $b_1 \approx 3.2$), re-cut from the saved files; the matched standard-resolution
   fiducial control is processed identically. If a denser PNG sample is needed, the FoF catalogues (complete to 20
   particles, $1.3\times10^{13}$) are the alternative.
3. Realisation 0 of the standard fiducial set is empty on the server; the fetch script skips empty files.
4. Transfer rate 85–155 MB/s; a batch of 50 (0.4–10 GB) takes 1–2 min plus ≈ 3 s per catalogue to cut.
