# Stage 4 memo: imaging systematics in the field-level model

Written 2026-10-13. Status: design, code and validation done; first posterior chains queued. Results will be added
to §6 as they arrive. Code: `src/templates.py`, template amplitudes in `src/forward_model.py` (`ParamLayout(templates=...)`,
`ForwardModel.set_templates`), `scripts/run_sampler.py` (`--templates`, `--contam`), Fisher cross-check in
`src/fisher_field.py` (`deproj_loss`) and `scripts/run_fisher_syst.py`. Test: `test/notes/deproj_dense.md`.

---

## 1. Question

The brief's Stage 4 asks whether imaging systematics can be handled inside the forward model, by sampling template
amplitudes jointly with the field, $f_{\rm NL}$ and bias, instead of pre-weighting the catalogue (decision 5). The
targets are $|\Delta f_{\rm NL}| < 2$ on contaminated mocks and a $\sigma(f_{\rm NL})$ degradation below 30 %. The
immediate question is how much $\sigma(f_{\rm NL})$ grows once the systematics are marginalised, compared with the
Stage 2 values (160³: $5.6 \pm 0.3$ with $b_\phi$ fixed and $6.0 \pm 0.3$ with $b_\phi$ sampled, mock 0).

## 2. Contamination model

For tracer $t$ at position $\mathbf x$ with sky direction $\hat n$:
$$\langle N_t\rangle(\mathbf x) \;\to\; \langle N_t\rangle(\mathbf x)\Big[1 + \sum_{r\in\{N,S\}}\sum_{k} a_{t,r,k}\,\tau_k(\hat n)\,\mathbb 1_r(\hat n)\Big].$$
- **Templates** $\tau_k$: the DESI DR1 LSS healpix property maps (`LSScats/v1.5/hpmaps`, nside 256), separately for the
  LRG and QSO selections and the two photometric regions. The default set ("imaging9") is $E(B-V)$, stellar density,
  PSF depth in $g, r, z, W1$, and PSF size in $g, r, z$, the inputs of the DESI DR1 regressions. "imaging3" is
  $E(B-V)$, stellar density and $z$ depth. Depths are converted to 5σ magnitudes, $m = 22.5 - 2.5\log_{10}(5/\sqrt{\rm ivar})$.
  Each map is standardised (zero mean, unit variance, coverage-weighted) within each region and clipped at 5σ.
- **Regions**: North (BASS/MzLS, Dec > 32.375° in the NGC) and South (DECaLS + DES). Each covers roughly one fifth and four
  fifths of the footprint, as in DESI.
- **Amplitudes** $a_{t,r,k}$: free per tracer and region, shared by all redshift shells of a tracer. Imaging-9 gives
  $9 \times 2 \times 2 = 36$ parameters, with flat priors, sampled jointly with the white-noise field, $f_{\rm NL}$ and the bias
  parameters. They start at 0 and are scaled for the sampler by their Fisher diagonal.
- **Radial modes** (Fisher only, so far): an unknown $n(z)$ shape per shell, $\langle N_s\rangle \to \langle N_s\rangle[1 + \sum_l c_{s,l}P_l(u)]$,
  with $u$ the normalised comoving distance across the shell. $l = 0$ is the shell normalisation that DESI's radial
  integral constraint removes.

The multiplicative form acts on the expected counts, so the contamination is modulated by the density field itself, as
in real data. To first order in $a$, marginalising flat-prior amplitudes is equivalent to deprojecting the data-space
vectors $T_{t,r,k}(\mathbf x) = \langle N_t\rangle(\mathbf x)\,\tau_k(\hat n)\mathbb 1_r$ (Kalus et al. 2016; Elsner et al. 2016).

Two consequences of sampling rather than regressing:
- The angular integral constraint does not arise. The amplitudes are fitted together with the large-scale modes, so
  the true large-scale clustering is not partly absorbed into the weights before the analysis.
- The cost of the templates appears directly as a wider $f_{\rm NL}$ posterior, instead of a bias that has to be
  calibrated on mocks.

## 3. Contaminated mocks

- **"wsys" (realistic).** The observed DR1 imaging modulation for each tracer: $1/\langle w_{\rm sys}\rangle$ per nside-256
  pixel, where $w_{\rm sys}$ is DESI's per-object imaging weight in the DR1 clustering catalogue. Pixels with fewer than
  3 objects use the nside-64 parent. The map is normalised to unit mean over the footprint. On the 16³ grid it has
  4.3 % rms, range 0.84–1.16. DESI's weights come from linear regression (LRG) and random forests (QSO) on these
  maps, so the 9 linear templates can describe this contamination only partly. It is therefore both a
  realistic-amplitude test and a model-mismatch test.
- **"linear" (exactly representable).** Random amplitudes $a \sim \mathcal N(0, 0.02^2)$ on the 36 templates. With the
  templates sampled the analysis must be unbiased; with them ignored it shows the size of the bias.

Mocks are drawn from the Stage 2 forward model at the same fiducial field and seed, so the widths compare directly
with Stage 2.

## 4. Run plan

MCLMC at half the auto-tuned step (the Stage 2 production setting), 60 000 steps, $f_{\rm NL,true} = 0$, DR1 window,
QSO + LRG, six shells. Chains checkpoint every 2000 steps and resume across 2-hour interactive sessions.

| tag | grid | templates sampled | mock contamination | measures | where |
|---|---|---|---|---|---|
| s4_80_none | 80³ | — | none | reference σ (same settings) | interactive |
| s4_80_T9 | 80³ | imaging9 | none | σ inflation from 36 amplitudes | interactive |
| s4_80_T9_wsys | 80³ | imaging9 | wsys | $\Delta f_{\rm NL}$ and σ, realistic case | interactive |
| s4_80_wsys_noT | 80³ | — | wsys | bias with no mitigation | interactive |
| s4_160_T9 | 160³ | imaging9 | none | σ inflation at the go/no-go resolution | interactive |
| s4_160_T9_wsys | 160³ | imaging9 | wsys | $\Delta f_{\rm NL}$ and σ at 160³ | interactive |
| s4_160_wsys_noT | 160³ | — | wsys | bias with no mitigation at 160³ | batch |
| s4_160_T9_bphi | 160³ | imaging9, $b_\phi$ 40 % prior | none | the brief's go/no-go configuration with systematics | batch |

Reference widths from Stage 2 (mock 0): 80³ full model 7.0–7.2; 160³ 5.6–5.75 ($b_\phi$ fixed), 6.0 ($b_\phi$ sampled).

**Fisher cross-check (queued after the chains).** The exact Gaussian-marginal Fisher with template modes deprojected,
$C \to C + \Lambda T^{\top}T$ with $\Lambda \to \infty$, for nested template sets (none, imaging3, imaging9, per-tracer
imaging9, radial $P_0$, radial $P_{0..2}$, imaging9 + radial). The information lost is computed exactly:
$$\Delta F_{ij} = \tfrac12\big[{\rm Tr}(Q\,\partial_iC\,C^{-1}\partial_jC) + (i\leftrightarrow j)\big] - \tfrac12{\rm Tr}(Q\,\partial_iC\,Q\,\partial_jC),\qquad Q = C^{-1}T^{\top}(TC^{-1}T^{\top})^{-1}TC^{-1}.$$
It costs $k(1+n)$ conjugate-gradient solves for $k$ templates and $n$ parameters and has no probe noise. It agrees with
a dense brute-force calculation to $10^{-12}$ (`test/notes/deproj_dense.md`). The baseline $F$ still comes from the
Hutchinson estimator and needs at least 32 probes: with 4 probes at 16³ the noisy baseline minus the exact loss was not
positive definite. The Fisher is the cheap way to rank template sets (e.g. adding radial modes or per-shell amplitudes);
the chains give the answer.

## 5. What to expect (literature)

From the Stage 4 part of the literature survey (`reports/PNG field level literature.md`):
- In P(k) analyses, regressing on more templates widens $f_{\rm NL}$ posteriors substantially: nine maps more than double
  σ relative to three for DESI LRGs (Rezaie et al. 2024), and fitting more than a few templates per region overfits
  (Krolewski et al. 2024). Linear deprojection of a few smooth templates is usually much cheaper than nonlinear weights.
- Ignoring integral constraints biased $f_{\rm NL}$ by $\sim 100$ in eBOSS-like mocks (Riquelme et al. 2023). In the
  forward model the angular one does not arise (§2). The radial one is equivalent to deprojecting the per-shell
  normalisation, which the Fisher cross-check includes.
- No published analysis samples imaging templates jointly with $f_{\rm NL}$ at the field level. BORG has block-sampled
  foreground template coefficients for SDSS (Jasche & Lavaux 2017; Lavaux et al. 2019) and has a likelihood that is
  robust to unknown foregrounds (Porqueres et al. 2019).
- Magnification and relativistic projection effects shift the effective $f_{\rm NL}$ by $\sim 0.5$–1 and are not part of this stage.

## 6. Results

*(to be filled in as the chains finish)*

| tag | $\langle f_{\rm NL}\rangle$ | $\sigma(f_{\rm NL})$ | ESS | ratio to reference |
|---|---|---|---|---|
| s4_80_none | | | | 1 |
| s4_80_T9 | | | | |
| s4_80_T9_wsys | | | | |
| s4_80_wsys_noT | | | | |
| s4_160_T9 | | | | |
| s4_160_T9_wsys | | | | |
| s4_160_wsys_noT | | | | |
| s4_160_T9_bphi | | | | |

## 7. Validation so far

- Dense check of the deprojection Fisher loss: agreement to $10^{-12}$ for nested sets of 1, 2 and 4 smooth templates in a $8^3$ box.
- 16³ DR1 smoke runs: chains with 36 template amplitudes run at 10 ms per gradient. The contamination maps load and apply
  as intended (wsys rms 4.3 %).
- MCLMC checkpoint and resume: a chain killed after 2000 steps resumed from the saved position, $L$, step size and
  inverse mass, and completed to the requested length.

## 8. Open items

- Results of the eight chains (§6), then the decision against the brief's criteria ($|\Delta f_{\rm NL}| < 2$; σ degraded < 30 %).
- If the realistic contamination leaves a bias: add the templates' squares and cross-terms, or per-shell amplitudes for
  the QSOs (whose selection systematics vary with redshift), and compare with the Fisher ranking.
- Radial $n(z)$ modes in the sampled model (currently Fisher only).
- Contamination entering through the selection mask rather than the counts (e.g. depth-dependent completeness at the
  survey edges), and redshift-dependent QSO systematics.
- Combine with the Stage 3 model updates (Fourier-truncated gridding, Laplacian bias) before any DR1 application. The
  Stage 4 chains use the Stage 2 model so that their widths compare with Stage 2.
