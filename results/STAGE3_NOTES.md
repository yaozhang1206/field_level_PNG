# Stage 3 notes: model mismatch on Quijote halos

Written 2026-10-13 (in progress). Code: `src/quijote.py`, `scripts/stage3_cut_quijote.py`, `scripts/stage3_fetch_quijote.sh`,
`scripts/run_stage3_halos.py`, `scripts/stage3_diag_window.py`, `scripts/stage3_summary.py`. Test notes:
`test/notes/stage3_quijote_cuts.md`, `stage3_window_*.md`, `stage3_gridding.md`, `stage3_ensembles.md`.
Per-realisation chain summaries: `results/stage3/*.json`.

---

## 1. Purpose and scope

Stage 2 tested the posterior on mocks drawn from the forward model itself. Stage 3 asks whether the **DR1 forward
model** recovers $f_{\rm NL}$ from tracers it did not generate: Quijote N-body halos. Quijote is only a model-mismatch
test bench, so the DR1 model is kept as it is (Lagrangian bias $b_1, b_2, b_K$, local PNG with $b_\phi$, $b_{\phi\delta}$
from the universality relation with the DR1 $p$ values, 2LPT, Kaiser RSD, Gaussian likelihood, 62.5 Mpc/h cells).
Changes were adopted only when they would also go into the DR1 analysis.

The brief's criterion is $|\Delta f_{\rm NL}| < 0.5\,\sigma$ at the DR1 resolution. A model-mismatch offset does not
shrink with volume, so the reference $\sigma$ is the DR1 width ($\approx 6$), not the per-box width ($\approx 65$ for a
$1\,(h^{-1}{\rm Gpc})^3$ box). An offset that comes from the shape of the posterior (skewness, priors) scales with
$\sigma$ and would be harmless; mocks drawn from the model separate the two cases (§6).

## 2. Data

Quijote Rockstar host halos, $L = 1\,h^{-1}$Gpc, Quijote cosmology ($\Omega_m = 0.3175$, $h = 0.6711$, $n_s = 0.9624$,
$\sigma_8 = 0.834$, CLASS linear power). Catalogues were downloaded with Globus in batches of 50, mass-cut, saved as
compact `.npz` files and the raw files deleted (scratch is 96 % full; the cuts total 1.6 GB).

| sample | set | $z$ | selection | boxes | $b_1$ | $\bar n$ [$(h^{-1}{\rm Mpc})^{-3}$] | role |
|---|---|---|---|---|---|---|---|
| LRG-like | fiducial_HR | 0.5 | $M > 8.9\times10^{12}$ | 100 | 1.87 | $3.6\times10^{-4}$ | $f_{\rm NL} = 0$ offset |
| QSO-like | fiducial_HR | 1 | $M > 7.0\times10^{12}$ | 50 | 2.41 | $3.6\times10^{-4}$ | $f_{\rm NL} = 0$ offset |
| QSO-like, high bias | fiducial_HR | 1 | $M > 1.4\times10^{13}$ | 50 | 2.86 | $1.5\times10^{-4}$ | $f_{\rm NL} = 0$ offset |
| PNG $\pm100$ | LC_p / LC_m | 0.5 | $M > 6\times10^{13}$ | 50 each | 3.0 | $3\times10^{-5}$ | response |
| control | fiducial (std res.) | 0.5 | $M > 6\times10^{13}$ | 59 | 3.0 | $3\times10^{-5}$ | control for the response |

Findings from building the samples (`test/notes/stage3_quijote_cuts.md`):
- At $z = 2$ even the smallest resolved high-resolution halos have $b_1 \approx 4.3$, so the DESI QSO bias is matched at $z = 1$.
- The standard-resolution Rockstar catalogues (the only resolution with PNG runs) are incomplete below $\approx 6\times10^{13}\,M_\odot/h$;
  the response test uses that threshold.
- The ASCII catalogues already on disk are the fiducial_HR set at $z = 0.5$ (mass column $= 0.89\,M_{\rm vir}$).

## 3. What was tested, in order

| step | setup | result | diagnosis |
|---|---|---|---|
| 1 | Stage 2 model unchanged; halos CIC-gridded on $16^3$, real space | $f_{\rm NL} \approx +700$, $b_1 \approx 1.05$; a mock drawn from the model was recovered | Model and halo grid power have different shapes (§4) |
| 2 | particles on a 4× finer lattice, block-averaged to the grid; halos NGP-gridded; $b_\nabla$ free; noise amplitude free; RSD | mocks recovered; halos: $f_{\rm NL} = -191$, $b_\nabla = 266$, noise $\times 5.8$ | Halo power above the grid Nyquist aliases into the cells; the model has no modes for it |
| 3 | Fourier truncation of a 4× CIC mesh (data and model) | $f_{\rm NL} = +8300$ | Bug: data in counts per fine cell (factor 64); fixed and an assertion on the data mean added |
| 4 | same with the CIC window deconvolved; noise amplitude free | noise amplitude $\ln A \to -27$, posterior width $\to 0$ | $\sigma$ collapse: the white-noise field has more degrees of freedom than the kept data modes |
| 5 | noise fixed at the Poisson level | pilot: $-91 \pm 68$, $+50 \pm 65$, $b_1 \approx 1.95$ | adopted as the Stage 3 baseline |
| 6 | ensembles of 20–26 boxes per sample | §5 | |

## 4. Gridding: the main technical lesson

The Stage 1–2 model scatters one particle per analysis cell directly onto the analysis grid. Its grid power falls toward
the Nyquist frequency much faster than that of a real tracer, because a regular lattice has no sub-cell structure.
Real tracers have structure below the cell size, which aliases into the grid. Mocks drawn from the model share the
model's window, so Stage 2 could not see this.

The adopted scheme, applied identically to data and model (`ForwardModel(..., mesh_oversample=4)`, `quijote.fourier_grid`):
1. CIC-assign the particles (data: tracers) to a mesh $m\times$ finer than the analysis grid;
2. keep only the Fourier modes of the analysis grid, $|n_i| < r/2$, and divide them by the fine-mesh CIC window;
3. sample the result at the analysis-grid nodes.

This is an orthogonal projection: power above the grid Nyquist does not alias in, and white noise stays white, so
the Gaussian likelihood remains valid. Measured on halos (`test/notes/stage3_gridding.md`):
- data power is independent of $m$ (2, 4, 8) to 0.2 %;
- the model converges at $m = 4$ ($m = 2$ is 5 % high at the Nyquist frequency);
- a finer **particle** lattice changes nothing (0.3 %). For DR1 this means $160^3$ particles plus one $640^3$ assignment
  mesh per shell (2 GB), not $640^3$ particles.

The Stage 1–2 results used the old scheme on self-consistent mocks; they are unaffected as Fisher/posterior widths, but
the DR1 analysis must use the new scheme.

## 5. Ensemble results (2LPT, $16^3$, $b_\nabla$ free, Poisson noise, RSD, cold start)

Pull $= (\bar f_{\rm NL} - f_{\rm NL}^{\rm true})/\sigma$ per box; a calibrated posterior has pull mean 0 and s.d. 1.

| sample | boxes | truth | $\langle f_{\rm NL}\rangle$ | box scatter | $\langle\sigma\rangle$ | pull s.d. | $\langle b_1\rangle$ |
|---|---|---|---|---|---|---|---|
| LRG-like, HR, $z = 0.5$, $p = 1.0$ | 26 | 0 | $-5 \pm 16$ | 83 | 62 | 1.45 | 1.92 |
| QSO-like, HR, $z = 1$, $b_1 = 2.4$, $p = 1.6$ | 20 | 0 | $+34 \pm 22$ | 96 | 82 | 1.10 | 2.41 |
| QSO-like, HR, $z = 1$, $b_1 = 2.9$, $p = 1.6$ | 20 | 0 | $+35 \pm 20$ | 88 | 68 | 1.17 | 2.82 |
| control, std res., $M > 6\times10^{13}$, $p = 1.0$ | 20 | 0 | $+20 \pm 14$ | 62 | 63 | 0.94 | 3.07 |
| PNG $+100$, same cut | 20 | +100 | $+120 \pm 17$ | 75 | 71 | 1.06 | 2.97 |
| PNG $-100$, same cut | 20 | −100 | $-94 \pm 13$ | 59 | 65 | 0.92 | 3.17 |

- **Response.** The $\pm100$ pair differs by $214 \pm 21$ (200 expected). Relative to the matched control the shifts are
  $+100$ and $-114$. The universality relation with $p = 1.0$ describes these halos; no tuning of $p$ is needed.
- **Offset at $f_{\rm NL} = 0$.** Pooled over the four $f_{\rm NL} = 0$ samples, $\langle f_{\rm NL}\rangle = +18 \pm 9$ (2σ).
  Per box this is $0.3\,\sigma$, but if it is model mismatch it would be $3\,\sigma$ for DR1. Not yet decisive.
- **Width calibration.** The standard-resolution samples are calibrated (pull s.d. 0.92–1.06). The high-resolution
  samples are too narrow by 10–45 %, most for the densest, lowest-mass LRG-like sample, pointing to unmodelled
  small-scale noise.

## 6. Lessons learned

1. **Self-consistent mocks hide the grid window.** Any DR1 pipeline needs one test against tracers from a different
   simulator before its results are trusted. Stage 2's sampler and Fisher validation did not exercise the gridding.
2. **Data and model must be gridded by the same operator, and that operator must not alias.** Fourier truncation of a
   4× CIC mesh with the window divided out is cheap and DR1-compatible; cell averaging or direct CIC on the analysis
   grid are not adequate.
3. **The Laplacian bias $b_\nabla$ is needed.** It is a physical term for any tracer and absorbs residual scale
   dependence near the grid scale; fitted values are $25$–$70\,(h^{-1}{\rm Mpc})^2$.
4. **Do not sample a free noise amplitude together with the field.** With more field degrees of freedom than data
   modes the noise collapses to zero. Fix it (Poisson level) or calibrate it on mocks.
5. **Cold starts work.** A $0.1\times$ random field, $f_{\rm NL} = +50$ and $b_1$ from the power spectrum, with a phased
   warm-up (field, then bias, then $f_{\rm NL}$), converges in all runs; Stage 2 had only started at the truth.
6. **Check normalisations explicitly.** The factor-64 data bug gave $f_{\rm NL} = 8300$; the run script now asserts that
   the gridded data mean equals $\bar n V_{\rm cell}$.
7. **Judge offsets against the DR1 width, not the per-box width,** unless mocks show that the offset scales with $\sigma$.
8. **"$k_{\max}$" convention.** Our $62.5\,h^{-1}$Mpc grid has Nyquist $k = 0.05\,h/$Mpc. Stage 1–2 called this
   "$k_{\max} = 0.1$" ($2\pi/\Delta L$, the brief's convention); Andrews et al. 2026 use $k_{\max} \approx 0.1$ for
   $31.25\,h^{-1}$Mpc cells (Nyquist). Their numbers and ours are not on the same footing.
9. **Comparison with Andrews et al. 2026 (arXiv:2603.20855).** Their bias is Eulerian on the evolved field with
   $b_\nabla$, $p = 1.3$ tuned to FoF halos, and a noise level fitted with the true initial conditions. We adopted only
   what transfers to DESI ($b_\nabla$, consistent mass assignment, phased warm-up, redshift space) and kept the DR1 $p$ values.

## 7. Running and open

Running or queued (2026-10-13):
- **Mock ensembles** (interactive sessions, chained by a driver script): 30 model-drawn mocks for the LRG-like and the
  high-bias QSO-like setups, recording $\chi^2$/dof. These decide whether the $+18$ offset is a posterior-shape effect
  (also present in mocks, scales with $\sigma$) or model mismatch (absent in mocks), and calibrate the widths.
- **Batch jobs (queued):** 30 more boxes for each $f_{\rm NL} = 0$ sample (offset error $9 \to \approx 5$); $b_\phi$,
  $b_{\phi\delta}$ sampled with the 40 % prior; Zel'dovich (1LPT) vs 2LPT; $32^3$ grid.

Open:
- If the offset is model mismatch: identify the source (bias expansion order, RSD treatment, 2LPT accuracy) and test
  remedies on the same boxes.
- If the high-resolution widths stay too narrow: a calibrated noise level (from mocks or the $\chi^2$ of halos vs
  mocks) for DR1.
- Light-cone and survey window are not part of Stage 3; Stage 4 (imaging systematics) is untouched.
