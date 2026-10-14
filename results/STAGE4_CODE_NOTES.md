# Stage 4 code guide: imaging systematics in the field-level model

Written 2026-10-14 for code review. State of the code: commit `d216d29` on `main`. The physics, run plan and results
are in `results/STAGE4_NOTES.md`. This note explains which code does what and why, and lists the points worth
checking.

---

## 1. What Stage 4 does

The data are DR1-like counts on the analysis grid. Imaging systematics change the expected counts multiplicatively:
$$\langle N_t\rangle(\mathbf x) = \underbrace{\bar n_t(z)\,V_{\rm cell}\,W_{\rm ang}(\hat n)\,[1+\delta_t(\mathbf x)]}_{\text{Stage 1–2 forward model}}\;\times\;\Big[1 + c_t(\mathbf x)\Big],\qquad
c_t(\mathbf x) = \sum_{k} a_k\,\tau_k(\mathbf x).$$
- **Template $\tau_k$.** A standardised imaging map, such as stellar density, restricted to one tracer, one redshift
  shell and one photometric region, and averaged over each voxel's solid angle. Optional additions are a constant
  (**intercept**) per shell × region and, for QSOs, **quadratic** products of the maps.
- **Amplitude $a_k$.** A free parameter with a flat prior, sampled jointly with the white-noise field, $f_{\rm NL}$ and
  the bias parameters. The $f_{\rm NL}$ posterior is therefore marginalised over the systematics.

Tests use DR1 mocks from the Stage 2 forward model. A mock is either clean or **contaminated** by DESI's own imaging
weights, with $1/w_{\rm sys}$ mapped on the sky per redshift bin. Two numbers matter:
- the bias of $f_{\rm NL}$ on contaminated mocks;
- the growth of $\sigma(f_{\rm NL})$ on clean mocks.

A Fisher calculation (exact deprojection) cross-checks the growth of $\sigma$.

Data flow:
```
DESI DR1 hpmaps (nside 256)  ─┐
DESI dust maps (E(B-V) diff) ─┼─ desi_feature_maps ─→ standardised maps per (z-bin, region)
DR9 footprint (regions)      ─┘                              │
                                                             ▼
                                         grid_template_compact (voxel-averaged; + intercepts / quadratic)
                                                             │  labels + compact arrays F, G, g, f, k
                       ┌─────────────────────────────────────┼─────────────────────────────────┐
                       ▼                                     ▼                                 ▼
     ParamLayout(templates=labels)        ForwardModel.set_templates_compact      dense_from_compact → data_space_from_grid
     (amplitudes appended to theta)       → _contaminate multiplies the counts     → FieldFisher.deproj_loss (Fisher)
                       └──────────── FieldPosterior → run_sampler.py (MCLMC) ──────────┘
DR1 clustering catalogue (WEIGHT_SYS) → wsys_contamination_desi → contaminated mock (run_sampler.py --contam wsys)
```

## 2. Inputs on disk

| path | content | used by |
|---|---|---|
| `/work/hdd/bdne/yzhang116/desi_dr1/hpmaps/{LRG,QSO}_mapprops_healpix_nested_nside256_{N,S}.fits` | DR1 LSS v1.5 imaging property maps (NESTED, nside 256), separate N (BASS/MzLS) and S (DECaLS+DES) photometry | `desi_feature_maps` |
| `/work/hdd/bdne/yzhang116/desi_dr1/dust/desi_dust_{gr,rz}_256.fits` | DESI stellar-reddening maps (RING; columns `EBV_GR` / `EBV_RZ`, `EBV_SFD`) | `_ebv_diff` |
| `/work/hdd/bdne/yzhang116/desi_dr1/regressis_data/Legacy_Imaging_DR9_footprint_256.fits` | regressis DR9 footprint flags `ISNORTH`, `ISSOUTH` (contains DES), `ISDES`, `ISNGC`, `ISSGC` (NESTED) | `desi_region_masks` |
| `/work/hdd/bdne/yzhang116/desi_dr1/{LRG,QSO}_{NGC,SGC}_clustering.dat.fits` | DR1 clustering catalogues (`RA`, `DEC`, `Z`, `WEIGHT_SYS`) | `wsys_contamination_desi` |

The paths can be overridden with `DESI_HPMAPS_DIR`, `DESI_DUST_DIR`, `DESI_DR9_FOOTPRINT` and `DESI_DR1_DIR`.

## 3. `src/templates.py`: template and contamination construction

### 3.1 DESI scheme definitions (current setup)

- **`DESI_FEATURES`** (line ~178). The features DESI regresses on, per tracer and redshift bin, from Chaussidon et al.
  2024 §5.1.1. The redshift bins equal our shells.
  - **LRG:** 6 features per bin, different in each bin.
  - **QSO:** the same 12 features in all bins.
  - **No plain `EBV`.** DESI leaves it out deliberately, because the SFD map traces the cosmic infrared background, i.e. real structure. QSOs use the two extinction differences instead.
- **`DESI_REGIONS`.** The regions in which DESI fits independently.
  - **LRG:** North, South_NGC, SGC. SGC includes DES.
  - **QSO:** North, South without DES, DES.
- **`desi_region_masks(tracer)`.** Boolean NESTED nside-256 masks of these regions, from the DR9 footprint flags.
  `ISSOUTH` includes DES, so the QSO "South" is `S & ~D`.
- **`_ebv_diff()`.** Computes $E(B-V)_{\rm SFD} - E(B-V)_{\rm DESI\ stars}$ from the g−r and r−z reddening maps, converted from
  RING to NESTED ordering. The sign convention does not matter for a linear fit.
- **`desi_feature_maps(tracer, coverage_nest)`.** For every (z-bin, region) it returns `(names, T, mask)`, with `T` of
  shape `(n_feat, npix)`.
  - **Source maps.** North uses the N maps, everything else the S maps.
  - **Units.** Features stay in raw units, as in regressis, with no magnitude conversion.
  - **Standardisation.** Each feature is **standardised within the region** (coverage-weighted mean 0, variance 1) and clipped at ±5σ.
  - **Missing values.** A NaN, or a 0 in a property-map column, is set to 0, the mean.

  Standardising per region makes the amplitudes comparable across regions. It is also why a region's mean density
  offset cannot be absorbed without an intercept.

### 3.2 Projection onto the analysis grid

- **`grid_directions(fm)`.** The NESTED nside-256 pixel of each voxel centre, as seen from the observer. Only the legacy
  functions and the path without voxel averaging use it.
- **`voxel_subpixels(fm, nsub)` / `voxel_average(fm, pixmap, nsub)`.** Each voxel is sampled at $n_{\rm sub}^3$ sub-points:
  4³ for res ≤ 96 and 3³ above. An angular map is then averaged over the sub-points' pixels, ignoring NaN.
  - **Why.** A voxel's counts integrate the selection over its solid angle. A 125 $h^{-1}$Mpc voxel at the far QSO shells spans about 2°, roughly 60 nside-256 pixels.
  - **The problem it fixes.** Sampling at the voxel centre turned pixel-level scatter in the maps, and in DESI's weights, into spurious voxel-scale structure.
  - **Caching.** The sub-pixel table is cached per (model, nsub).
- **`grid_template_compact(fm, tracers, intercepts=False, voxel_avg=True, quadratic=())`** (line ~310). Builds the
  templates actually used.

  Every footprint voxel belongs to exactly one group, a shell × region. So instead of a dense `(K, res³)` array, it
  returns compact arrays per tracer. The dense array would be 162 × 160³ × 8 bytes ≈ 5 GB at 160³, which ran out of
  memory. The compact arrays:
  - **`F`** `(n_feat, res³)` float32: the value of each feature column at each voxel, from the voxel's own region's
    standardised map, voxel-averaged. `tk = np.where(m, T[k], nan)` restricts the average to sub-points inside the region.
  - **`G`** `(res³,)` int32: the group id of each voxel; 0 means no template.
  - **`g`, `f`, `k`:** for every amplitude, its group, its feature column, and its index in `labels`.
  - **`labels`:** `(tracer, "z<lo>-<hi>:<region>:<feature>")`, in the order the amplitudes appear in θ.

  Options:
  - **`intercepts=True`** adds a constant `INTERCEPT` column (= 1) per group.
  - **`quadratic=("QSO",)`** appends all squares and pairwise products of that tracer's linear features as extra
    columns. Each (group, product) gets one amplitude, but only for features used in that bin.

  Counts at 80³:

  | set | amplitudes |
  |---|---|
  | linear | 162 (QSO 108, LRG 54) |
  | + 18 intercepts | 180 |
  | + QSO quadratic | 882 |
- **`dense_from_compact(fm, labels, comp)`.** Expands the compact form to `(K, res³)`. Used only for the Fisher and the diagnostics.
- **`data_space_from_grid(fm, tracers, idx, maps, labels)`.** The Fisher data-space vectors
  $T_k(\mathbf x) = \langle N_t\rangle(\mathbf x)\,\tau_k(\mathbf x)$ on the footprint voxels, in `FieldFisher`'s ordering. These
  are the derivatives of the counts with respect to $a_k$ at $a = 0$.

### 3.3 Contamination for mocks

- **`wsys_contamination_desi(fm, tracer, voxel_avg=True)`** (line ~266). The "realistic" contamination. For each shell it:
  1. takes the DR1 objects in that redshift bin;
  2. computes $1/\langle w_{\rm sys}\rangle$ per nside-256 pixel, using only pixels that contain objects;
  3. averages that over each voxel's sub-points, setting voxels with no information to 1;
  4. normalises to unit mean over the shell's voxels.

  It returns a `(res, res, res)` factor $g(\mathbf x)$, and the mock counts become $g\,\langle N\rangle$.
  - **No per-region normalisation.** The normalisation is over the whole shell, not per region. DESI's region-to-region density differences are therefore part of the contamination: 5–7% steps for QSOs. That is what the intercepts must absorb.
  - **Size.** At 80³ the voxel-averaged factor has rms 2.0% for LRG and 4.6% for QSO.

### 3.4 Legacy functions (first Stage 4 setup, kept for the earlier runs)

- **`SETS`, `_read`, `template_maps`, `grid_template_maps`.** Nine maps for all shells, N/S regions only, depths in magnitudes, plain `EBV`.
- **`wsys_contamination`.** One map per tracer from all redshifts.
- **`data_space_templates`.** Imaging and radial Legendre modes. The Fisher script still uses it for the radial modes.
- **`coverage_nest256`.** The DR1 coverage map evaluated on nside-256 pixels. Still used.

## 4. `src/forward_model.py`: how the amplitudes enter the model

- **`ParamLayout(shells, sample_bphi, lap, templates=labels)`.** Appends one parameter per template after the shell
  parameters. `self.t0` is the index of the first amplitude. `fiducial()` sets all amplitudes to 0.
- **`ForwardModel.set_templates_compact(layout, comp)`.** Stores the compact arrays on the device.
- **`ForwardModel._contaminate(out, theta, layout)`** (line ~245). Called at the end of both `__call__` (the 2LPT model)
  and `linear_galaxy_field`. For each tracer it:
  1. scatters the amplitudes into a small matrix `A[group, feature]`;
  2. forms $c(\mathbf x) = \sum_f A[G(\mathbf x), f]\,F_f(\mathbf x)$;
  3. multiplies the tracer's counts by $1 + c$.

  It is linear in the amplitudes and differentiable. The cost is one gather per feature column.
- **`set_templates(layout, maps)`.** The legacy dense-map path, from the first setup (`--templates imaging9`).

## 5. `scripts/run_sampler.py`: the Stage 4 chains

Relevant options:

| option | effect |
|---|---|
| `--templates desi` | DESI scheme via `grid_template_compact`; amplitudes are sampled (added to `free`) |
| `--intercepts` | add the per-group constant |
| `--quadratic QSO` | add QSO quadratic products |
| `--contam wsys` | contaminate the mock with `wsys_contamination_desi` |
| `--contam linear` | contaminate with random amplitudes on the model's own templates (exactly representable) |
| `--sampler mclmc --mclmc_step_factor 0.5` | production sampler (half the tuned step) |
| `--resume` | continue from `results/chain_<tag>.npz` (position, L, step size, inverse mass) |

Steps in the script:
1. **Build the model.** The DR1 model (`stage1_config.build`, 10 $h^{-1}$Gpc box, DR1 angular mask), the template labels and
   compact arrays, and the `ParamLayout` with amplitudes.
2. **Draw the mock.** From the model at the truth: $f_{\rm NL} = 0$, fiducial bias, all amplitudes 0. The noise is Gaussian
   with variance $\bar n V_{\rm cell}$.
3. **Contaminate (`--contam wsys`).** `d = mu * g + (d - mu)`: the noise realisation is kept and the mean is multiplied by the
   contamination factor.
4. **Scale the parameters.** The sampler's parameter scaling comes from the diagonal of the Stage 1 Fisher block
   (`FieldFisher.fisher`) at the truth. The amplitudes are included in `free` and start at 0.
5. **Run MCLMC.** It tunes $L$ and the step size on 300 steps, skipped on resume, then samples. It checkpoints every
   2000 steps to `results/chain_<tag>.npz`, plus a `.json` summary with `"partial": true` until finished.

## 6. Fisher cross-check: `src/fisher_field.py` and `scripts/run_fisher_syst.py`

- **`FieldFisher.fisher_marginal(...)`** (existing, from Stage 1). A Hutchinson estimate of the exact Gaussian-marginal
  Fisher, $F = \tfrac12{\rm Tr}[C^{-1}\partial_iC\,C^{-1}\partial_jC]$ with $C = J_\varepsilon J_\varepsilon^\top + N$.
- **`FieldFisher.deproj_loss(T, configs, checkpoint=...)`** (line ~237). The exact information lost when the template
  modes $T$ are marginalised with flat priors ($C\to C + \Lambda T^\top T$, $\Lambda\to\infty$), for several nested template
  sets at once:
  $$\Delta F_{ij} = \tfrac12\big[{\rm Tr}(Q\,\partial_iC\,C^{-1}\partial_jC) + (i\leftrightarrow j)\big] - \tfrac12{\rm Tr}(Q\,\partial_iC\,Q\,\partial_jC),\quad
  Q = C^{-1}T^\top(TC^{-1}T^\top)^{-1}TC^{-1}.$$
  It needs $k(1+n)$ conjugate-gradient solves for $k$ templates and $n$ parameters. It is deterministic and
  checkpoints after every parameter. Then $F_c = F - \Delta F_c$.
- **`fisher_marginal_deproj(...)`.** An earlier probe-based version. It is noisier and is not used for results.
- **`scripts/run_fisher_syst.py --res 80 --probes 32`.** Builds the DESI templates (`grid_template_compact` →
  `dense_from_compact` → `data_space_from_grid`) and per-shell radial Legendre modes.
  - **Output.** $\sigma(f_{\rm NL})$ for `none`, `desi`, `radial_P0`, `radial_P2`, `desi+radial_P0`, `desi+radial_P2`, `desi_LRGonly` and `desi_QSOonly`, written to `results/fisher_syst_res80.json`.
  - **Limitation.** The Fisher uses the DESI linear templates **without intercepts**.

## 7. Diagnostics and infrastructure

- **`scripts/stage4_fit_contam.py [res]`.** A weighted least-squares fit of the contamination $g - 1$ on candidate
  template sets: linear, + intercepts, + squares, + per-region radial modes. The weights are the expected counts. It
  prints the residual rms per tracer. It runs on CPU in under a minute at 80³ and is used to rank template models
  before running chains.
- **`scripts/interactive_queue_driver.sh`.** Runs the tasks in `logs/interactive_queue.txt` one after another in
  2-hour interactive GPU sessions. It skips a task after three sessions shorter than 10 minutes. Log:
  `logs/interactive_queue.log`.
- **`scripts/check_done.py chain <json>`.** The done-check for a chain: it has a summary and is not partial.
- **`src/sampler.py: run_mclmc(..., resume=, params_out=, checkpoint_every=)`.** MCLMC with resume.

## 8. Tests and validation

| check | result | where |
|---|---|---|
| `deproj_loss` vs dense brute force (8³ box, 4 smooth templates, 2LPT linearised) | agreement to $10^{-12}$ | `test/test_deproj_dense.py`, `test/notes/deproj_dense.md` |
| compact vs dense templates in the model (32³, random amplitudes) | max relative difference $10^{-8}$ | ad-hoc check (not a saved test) |
| MCLMC resume after a kill | chain completes to the requested length with the saved L, step, mass | ad-hoc 16³ check |
| contamination-fit diagnostic | residual rms: LRG 42% → 15% with intercepts; QSO 86% → 57% (→ 54% with squares) | `stage4_fit_contam.py` |

## 9. Points worth checking in the code

1. **Voxel-averaged features are not re-standardised.** `desi_feature_maps` standardises at the pixel level. After
   `voxel_average`, the per-group mean is only approximately 0 and the variance is below 1. With intercepts this does not
   affect the fit. Without them, a small mean offset remains in each template.
2. **Quadratic products are not mean-subtracted** (`grid_template_compact`, quadratic block). The intercept absorbs the
   mean, so `--quadratic` should always be used together with `--intercepts`.
3. **Contaminated-mock noise** (`run_sampler.py`, step 3). Only the mean is multiplied by $g$; the noise realisation is
   unchanged. For Poisson-like noise the variance should scale with $g$ too. That is a 5% effect on the noise in the most
   contaminated voxels. It does not bias $f_{\rm NL}$, but strictly the mock noise is slightly mis-specified.
4. **Shell normalisation of the contamination** (`wsys_contamination_desi`). The mean is set to 1 over each shell, not
   weighted by $\bar n$. The region offsets are kept on purpose, because DESI's weights include them.
5. **Out-of-date docstrings.**
   - `wsys_contamination_desi` still says empty pixels use the nside-64 parent average. That is only true for `voxel_avg=False`.
   - `stage4_fit_contam.py` claims to report large-scale angular power. It reports only the residual rms. The $k$-binned residual power quoted in the conversation came from a separate inline script.
6. **Voxels with no information.** Voxels with no DESI objects in any sub-pixel get contamination 1 and template value 0
   (`np.nan_to_num`). At 80³ there are few such voxels. At 160³ there will be more, and they should be counted.
7. **LRG voxels without a template.** In the first DESI-scheme version, 12% of LRG footprint voxels had none, because centre
   sampling fell outside the region masks. With voxel averaging, every footprint voxel of both tracers falls in some
   region (checked at 80³: "outside regions 0.00"). This should be re-checked at 160³.
8. **The Fisher has no intercepts or quadratic terms.** `run_fisher_syst.py` does not include them yet. Its $\sigma$
   inflation (×1.14 for imaging, ×1.22 with per-shell normalisation) therefore does not match the current chain setup exactly.
9. **DESI feature list transcription.** `DESI_FEATURES` was typed from the paper's text (§5.1.1). The LRG bins use galaxy
   depths and the QSO bins PSF depths. It should be checked against the paper and against regressis' defaults.

## 10. Current results (80³, $f_{\rm NL,true} = 0$, voxel-averaged)

| run | templates | mock | $f_{\rm NL}$ |
|---|---|---|---|
| s4_80_none (reference) | none | clean | $+2.0 \pm 7.1$ |
| s4v_80_Ti | DESI + intercepts (180) | clean | $-1.1 \pm 8.1$ |
| s4v_80_wsys_noT | none | contaminated | $+94.3 \pm 7.2$ |
| s4v_80_T_wsys | DESI, no intercepts (162) | contaminated | $+80.5 \pm 7.5$ |
| s4v_80_Ti_wsys | DESI + intercepts (180) | contaminated | $+22.1 \pm 7.8$ |

**Next in the queue.** The same runs with QSO quadratic terms (882 amplitudes), on the contaminated and the clean mock.
The 160³ runs are paused until the 80³ bias meets the brief's $|\Delta f_{\rm NL}| < 2$.
