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
| 1 | Stage 2 model unchanged; halos CIC-gridded on $16^3$, real space | $f_{\rm NL} \approx +700$, $b_1 \approx 1.05$; a mock drawn from the model was recovered | CIC smooths the halo signal and noise; the lattice model is unsmoothed and assumes Poisson noise (§4.3–4.4) |
| 2 | particles on a 4× finer lattice, block-averaged to the grid; halos NGP-gridded; $b_\nabla$ free; noise amplitude free; RSD | mocks recovered; halos: $f_{\rm NL} = -191$, $b_\nabla = 266$, noise $\times 5.8$ | Halo power above the grid Nyquist aliases into the cells; the model has no modes for it |
| 3 | Fourier truncation of a 4× CIC mesh (data and model) | $f_{\rm NL} = +8300$ | Bug: data in counts per fine cell (factor 64); fixed and an assertion on the data mean added |
| 4 | same with the CIC window deconvolved; noise amplitude free | noise amplitude $\ln A \to -27$, posterior width $\to 0$ | $\sigma$ collapse: the white-noise field has more degrees of freedom than the kept data modes |
| 5 | noise fixed at the Poisson level | pilot: $-91 \pm 68$, $+50 \pm 65$, $b_1 \approx 1.95$ | adopted as the Stage 3 baseline |
| 6 | ensembles of 20–26 boxes per sample | §5 | |

## 4. Gridding: the main technical lesson

### 4.1 How a grid sees a field

Let the analysis grid have spacing $H = L/r$ ($H = 62.5\,h^{-1}$Mpc, $r = 16$), Nyquist frequency $k_N = \pi/H$
($0.05\,h/$Mpc), and let a continuous density $\rho(\mathbf x)$ be assigned to it with a kernel $W$ (NGP: top-hat of width
$H$; CIC: its self-convolution). The value at a node is a sample of $W * \rho$, and sampling folds every wavevector
onto the grid's Fourier cube:
$$\tilde\delta_{\rm grid}(\mathbf k) = \sum_{\mathbf n \in \mathbb Z^3} \tilde W(\mathbf k + 2k_N\mathbf n)\,\tilde\delta(\mathbf k + 2k_N\mathbf n),
\qquad |k_i| \le k_N,$$
$$\tilde W_{\rm NGP}(\mathbf k) = \prod_i {\rm sinc}\!\left(\frac{k_i H}{2\pi}\right),\qquad \tilde W_{\rm CIC} = \tilde W_{\rm NGP}^2 .$$
The expected grid power of a tracer with power $P(k)$ and number density $\bar n$ is then (Jing 2005)
$$P_{\rm grid}(\mathbf k) = \sum_{\mathbf n} |\tilde W(\mathbf k + 2k_N\mathbf n)|^2\, P(|\mathbf k + 2k_N\mathbf n|) \;+\; \frac{1}{\bar n}\sum_{\mathbf n}|\tilde W(\mathbf k + 2k_N\mathbf n)|^2 .$$
Three things set the grid power near $k_N$: the window $|\tilde W|^2$ (suppression), the aliased images
$\mathbf n \neq 0$ (power from $|\mathbf k| > k_N$ folded back in), and the shot-noise term. For NGP the shot-noise sum is
exactly $1/\bar n$. For CIC it is $\frac{1}{\bar n}\prod_i[1 - \tfrac23\sin^2(k_iH/2)]$, which falls to $(1/3)/\bar n$ per axis at
$k_N$; in real space the CIC noise variance per cell is $(2/3)^3 \bar n V_{\rm cell} \approx 0.30\,\bar n V_{\rm cell}$, which is why
Andrews et al. find a fitted noise level well below Poisson.

### 4.2 Real tracers

Halos are points scattered at arbitrary positions inside cells, and their power does not stop at $k_N$: for the LRG-like
sample $b_1^2 P_{\rm lin}$ at $k = 2k_N = 0.1\,h/$Mpc is still 45 % of its value at $k_N$, and nonlinear clustering
adds more. So gridded halos feel the full window and receive aliased power from $k > k_N$, exactly as in §4.1.

### 4.3 The Stage 1–2 model

The model is not a set of random points. Its particles start on a regular lattice with one particle per analysis cell,
at the grid nodes, and are moved by displacements $\boldsymbol\Psi(\mathbf q)$ built from a white-noise field with modes
only up to $k_N$. Each particle carries the Lagrangian bias weight $w(\mathbf q) = 1 + b_1^L\delta_L + \dots$ and is CIC-assigned
to the analysis grid. Two consequences:

1. **The window is not that of a point set.** A particle displaced by much less than a cell lands almost on its own node,
   so CIC gives it almost entirely to that node. Expanding to first order in $\boldsymbol\Psi$,
   $$\tilde\delta_{\rm grid}(\mathbf k) \simeq \tilde\delta_w(\mathbf k) + \sum_i \frac{\sin(k_i H)}{k_i H}\,\frac{k_i^2}{k^2}\,\tilde\delta_L(\mathbf k),$$
   where the first term is the Lagrangian weight field sampled at the nodes (no window at all) and the second is the
   matter displacement term, whose CIC finite-difference response is $\sin(k_iH)/(k_iH)$ instead of 1 (zero at $k_i = k_N$).
   The bias part is therefore unsmoothed while the displacement part is suppressed, and neither matches the
   $|\tilde W|^2$ that real points experience. The mismatch is scale dependent and differs between the $b_1^L$ and the
   matter terms, so no single bias parameter can absorb it.
2. **Nothing to alias.** The model field has no modes above $k_N$ (the initial field is truncated there, and the
   discreteness of a regular lattice only appears at the reciprocal-lattice vectors $2k_N\mathbf n$, which fold onto
   $\mathbf k = 0$). Its grid power has no aliased contribution, while the halo grid power does.

In addition the likelihood assumes noise variance $\bar n V_{\rm cell}$, the Poisson value for counts in cells, which holds
for NGP-gridded data but overestimates CIC-gridded data by a factor $\approx 3$.

### 4.4 What this did to the fits

- **Step 1 (halos CIC-gridded, lattice model).** The halo grid is smoothed in both signal (by $\tilde W_{\rm CIC}^2$) and
  noise (variance $\approx 0.3\,\bar n V_{\rm cell}$); the model is not smoothed and assumes the full Poisson noise. The model
  therefore predicts too much small-scale variance. Measured on one box (shot noise subtracted with $1/\bar n$):
  $P_{\rm data}/b_1^2P_{\rm lin} = 0.31$ and $0.20$ at $k = 0.040$ and $0.047$, against 0.51 and 0.42 for the model. The
  posterior lowered $b_1$ to $1.05$ to match the small scales and restored the large-scale power with the
  $b_\phi f_{\rm NL}/k^2$ term, $f_{\rm NL} \approx +700$.
- **Step 2 (halos NGP-gridded, 4× particle lattice block-averaged to the grid).** Now both sides have the NGP window
  and NGP noise is exactly Poisson, but the halos carry aliased power and the model has none. Averaged over 4 boxes
  (real space, $b_1^2 = 3.53$), $P_{\rm grid}/P_{\rm lin}$ at $k = 0.052, 0.061, 0.071$ is $2.43, 2.16, 2.00$ for the halos and
  $1.22, 0.68, 0.25$ for the model. The posterior absorbed the missing power as extra noise ($\times 5.8$) and a large
  Laplacian bias ($b_\nabla = 266$), with $f_{\rm NL} = -191$.

Mocks drawn from the model have the model's window by construction, so the Stage 2 validation could not see either effect.

### 4.5 The fix

The adopted scheme, applied identically to data and model (`ForwardModel(..., mesh_oversample=4)`, `quijote.fourier_grid`):
1. CIC-assign the particles (data: tracers) to a mesh $m\times$ finer than the analysis grid;
2. keep only the Fourier modes of the analysis grid, $|n_i| < r/2$, and divide them by the fine-mesh CIC window;
3. sample the result at the analysis-grid nodes.

Why each problem in §4.3–4.4 goes away:
- **Aliasing.** The fine mesh has Nyquist $4k_N$. Power between $k_N$ and $4k_N$ is resolved there and then discarded by
  the truncation instead of being folded onto the kept modes; what aliases from above $4k_N$ is suppressed by
  $|\tilde W_{\rm CIC}|^2 \lesssim 0.4^2$ per axis and by the falling $P(k)$, a sub-percent effect.
- **Window.** On the fine mesh a model particle is spread over fine cells of width $H/4$, so for the kept modes
  ($|k_i| < k_N$) its assignment is that of a point: both the weight term and the displacement term see the same
  fine-mesh window, which is divided out. The lattice discreteness sits at $2k_N\mathbf n$, outside the kept modes.
  Data and model then both estimate the true $\tilde\delta(\mathbf k)$ for $|\mathbf k| < k_N$.
- **Noise.** Truncating to the coarse modes and sampling at the coarse nodes is an orthogonal projection (an exact
  change of basis on the kept modes), so white noise stays white and uncorrelated between cells. After the window
  division its variance is $\bar n V_{\rm cell}$ times a factor within a few per cent of 1 at $m = 4$, so the Poisson-level
  Gaussian likelihood applies.

Measured on halos (`test/notes/stage3_gridding.md`):
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
