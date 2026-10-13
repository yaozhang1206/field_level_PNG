# Field-level $f_{\rm NL}$ must now beat 7.4, not 9

No one has yet measured $f_{\rm NL}$ with field-level inference on real galaxy or quasar data. That makes a DESI DR1 field-level $f_{\rm NL}$ a first, but it also raises the bar the project must clear. The only field-level $f_{\rm NL}$ test on N-body halos (BORG on Quijote-PNG) gains about **1.3× over P+B at $k_{\max}\approx0.1$** ([Andrews et al. 2026](https://arxiv.org/abs/2603.20855)). The strongest DESI DR1 result is no longer the P(k) number of about 9. It is the EFT one-loop P+B reanalysis at **$f_{\rm NL}=-0.1\pm7.4$** ([Chudaykin et al. 2025](https://arxiv.org/abs/2512.04266)), and a one-loop P+B forecast for DESI reaches **$\sigma=3.3$** ([Bragança et al. 2023](https://arxiv.org/abs/2307.04992)). A field-level DR1 result of $\sigma\lesssim5$ therefore needs roughly 1.5× over P+B, which puts it at the edge of what the literature supports. Every problem the project has hit is documented elsewhere:
- **Sigma collapse** is a named failure ([Nguyen et al. 2021](https://arxiv.org/abs/2011.06587)), and the noise-field likelihood of [Rubira & Schmidt](https://arxiv.org/abs/2511.05484) is the first principled cure.
- **Incomplete window cancellation** between discrete halos and a one-particle-per-cell model is stated outright in the LEFTfield numerics paper ([Stadler et al. Part I](https://arxiv.org/abs/2409.10937)).
- **The universal-mass-function (UMF) $b_{\phi\delta}$** overpredicts halo values by up to about 3 ([Barreira 2022a](https://arxiv.org/abs/2107.06887)).
- **Andrews et al.'s $p=1.3$** is an empirical tuning of their forward model, not a measurement.

The most urgent configuration fix is the LRG sample. DESI's baseline uses **0.6<z<1.1 (1,631,716 LRGs)**, not the 0.4–1.1 sample of 2,130,621 in the project brief ([Chaussidon et al.](https://arxiv.org/abs/2411.17623)). This report deduplicates about 295 papers (293 arXiv IDs plus two books or articles without one) from five research-notes files into 20 categories and draws out the implications for Stages 1–4.

## Sources were checked against arXiv, INSPIRE and Crossref

Each arXiv ID below was checked on 2026-10-13 against the arXiv abstract page or API, INSPIRE-HEP, or Crossref by at least one of the five notes files. Journal references follow INSPIRE or the arXiv journal-ref field where the notes disagree. Markers:
- **[UNVERIFIED]** carries over items the notes took from memory or could not confirm.
- **[not found]** marks references named in the brief that do not exist as written.

Each paper appears once, under its most relevant category (letters A–T). "See X" points to related entries in other categories.

## Eight implications that change the project plan

### The DR1 LRG baseline is 0.6–1.1, and the real bar is P+B at 7.4

Chaussidon et al.'s baseline LRG sample is **0.6<z<1.1 with 1,631,716 objects**. They dropped the 0.4–0.6 sub-sample because its posterior was bimodal and not robust to $k_{\min}$, probably because of residual imaging systematics ([arXiv:2411.17623](https://arxiv.org/abs/2411.17623); [HTML](https://arxiv.org/html/2411.17623)). The 2,130,621 figure in the brief appears in the full text as the parent 0.4–1.1 sample. One notes file read it as the analysis sample, but the abstract and the robustness discussion settle the question.

Part A must use 0.6–1.1 to reproduce the LRG $\sigma$. The Part B 0.4–0.6 LRG shell is a forecast choice outside the DESI baseline, and it should be reported separately, since the data showed systematics there.

On baselines, the field-level result has to be compared against the full set of DR1 analyses, not only the P(k) paper:

| DR1 analysis | Method | $f_{\rm NL}$ | Source |
|---|---|---|---|
| Chaussidon et al. 2025 | $P_0+P_2$, QSO $p=1.6$ | $-3.6^{+9.0}_{-9.1}$ | [2411.17623](https://arxiv.org/abs/2411.17623) |
| Chaussidon et al. 2025 | same, QSO $p=1.0$ | $3.5^{+10.7}_{-7.4}$ | [2411.17623](https://arxiv.org/abs/2411.17623) |
| Fondi et al. 2026 | TNG $p$ prior, $\bar p\simeq1.4$ | $-3.3\pm9.2$ | [2602.12357](https://arxiv.org/abs/2602.12357) |
| Rosado-Marín et al. 2026 | LRG×QSO cross spectra | $2.1^{+8.8}_{-8.3}$ | [2604.05213](https://arxiv.org/abs/2604.05213) |
| **Chudaykin, Ivanov & Philcox 2025** | **EFT one-loop P+B** | **$-0.1\pm7.4$** (with CMB: $-0.0\pm4.1$) | [2512.04266](https://arxiv.org/abs/2512.04266) |
| Brown et al. 2026 | 2pcf, simulation-based | $-3\pm12$ | [2606.24651](https://arxiv.org/abs/2606.24651) |
| Chiarenza et al. 2026 | DR1 QSO × Planck PR4 $\kappa$ | $2^{+28}_{-34}$ | [2512.17865](https://arxiv.org/abs/2512.17865) |
| Clean EZmocks (Chaussidon) | DR1 combined OQE / Y5 OQE | $1.1^{+10.5}_{-7.9}$ / $-0.1^{+6.8}_{-6.2}$ | [HTML](https://arxiv.org/html/2411.17623) |
| Bragança et al. 2023 (forecast) | one-loop P+B, DESI | $\sigma=3.3$ | [2307.04992](https://arxiv.org/abs/2307.04992) |
| Planck 2018 IX | CMB | $-0.9\pm5.1$ | [1905.05697](https://arxiv.org/abs/1905.05697) |

The arithmetic is unforgiving. Suppose the Quijote field-level gain of about 1.3× over P+B carries over to DR1. Then the expected field-level $\sigma\approx7.4/1.3\approx5.7$, just above the ≲5 go/no-go threshold (this is an inference, not a published number). The P+B forecast of 3.3 for DESI is below the threshold. That forecast appears to be for the full survey, and the notes do not record its $b_\phi$ treatment. Chudaykin et al.'s abstract also does not state the $b_\phi$/$p$ treatment behind 7.4, so check it before quoting a ratio.

Two further results argue for modest expectations. On real data the bispectrum has added only **15–20%** to P ([Cabass et al. 2022](https://arxiv.org/abs/2204.01781); [Cagliari et al. 2025](https://arxiv.org/abs/2502.14758)), against idealised >5× forecasts. CMB-lensing cross-correlation on DR1 QSOs gives $\sigma\sim30$, about 3× weaker than the auto-power. The brief's "add CMB lensing if $\sigma\gtrsim7$" fallback is therefore a robustness check, not a precision lever.

### $b_{\phi\delta}$, not only $p$, is probably mis-specified

The brief fixes $b_{\phi\delta}$ by the UMF formula, $b_{\phi\delta}=b_\phi-b_1+1+\delta_c[b_2-\tfrac{8}{21}(b_1-1)]$. The first direct simulation measurement shows that this formula **overpredicts halo $b_{\phi\delta}(b_1)$ by up to $\Delta b_{\phi\delta}\sim3$** for $1\lesssim b_1\lesssim3$. The breakdown is not simply inherited from $b_\phi$. The measurement gives a fit for halos with $b_1<4$ at all $z$: $b_{\phi\delta}=3.85-9.49\,b_1+3.44\,b_1^2$ ([Barreira 2022a](https://arxiv.org/abs/2107.06887)).

At $b_1=2$ that fit gives about −1.4, against roughly +1.6 from UMF. The UMF value uses a Lazeyras et al. 2016 $b_2(b_1)$ fit whose coefficients were quoted from memory [UNVERIFIED], so recheck this arithmetic against the $b_2$ used in the code.

Fixing both $b_\phi$ and $b_{\phi\delta}$ to UMF in a halo P(1-loop)+B(tree) likelihood produced a **significant systematic shift in $f_{\rm NL}$** ([Moradinezhad Dizgah et al. 2021](https://arxiv.org/abs/2010.14523)). Field-level runs at $k_{\max}=0.1$ draw heavily on bispectrum-like information, so they inherit this risk. Two other results point the same way:
- Marginalising $b_\phi$ and $b_{\phi\delta}$ at the field level degrades $f_{\rm NL}$ by 2–3× and leaves it biased on halos ([Sullivan & Chen 2024](https://arxiv.org/abs/2410.18039)).
- Wide $b_\phi$ priors cause projection-effect biases in posterior means ([Barreira 2022b](https://arxiv.org/abs/2205.05673)).

The 40% prior should therefore be tested for mean shifts in Stage 2 MCMC, not only for $\sigma$.

On $p$, the three values the project has met measure different things:
- **1.6** is the extended Press–Schechter (EPS) upper bound for objects observed at formation ([Slosar et al. 2008](https://arxiv.org/abs/0805.3580)). TNG merger trees revise it to **$\bar p\simeq1.4$ for DESI-like QSOs** ([Fondi et al. 2026](https://arxiv.org/abs/2602.12357)).
- **1.3** is Andrews et al.'s tuning to remove a $b_1$ offset across $f_{\rm NL}=\pm100$ runs ([2603.20855](https://arxiv.org/abs/2603.20855), App. A).
- **1.0** from the project's ±100 response measurement is consistent with PNG-UNIT ($p=0.955\pm0.013$; [Gutiérrez Adame et al. 2024](https://arxiv.org/abs/2312.12405)) and with the 10–20% high-mass suppression ($p_{\rm eff}\approx1.1$–1.3) reported by [Biagetti et al. 2017](https://arxiv.org/abs/1611.04901) and [AbacusPNG](https://arxiv.org/abs/2402.10881).

A cheap Stage 3 test follows: swap in the Barreira fit, or sample $b_{\phi\delta}$, and see whether the need for $p>1$ disappears. [Hadzhiyska & Ferraro 2025](https://arxiv.org/abs/2501.14873) offer a route to a tighter $b_\phi$ prior from quantities a field-level model already fits. They cut the $b_\phi$ uncertainty by >70% by regressing $b_\phi$ on Lagrangian biases, mainly $b_\nabla$ as a concentration proxy.

### Sigma collapse is a known failure with a principled cure

"Sigma collapse" was named by [Nguyen et al. 2021](https://arxiv.org/abs/2011.06587). They found that a free Gaussian variance "collapses to a very small value" when the Poisson variance per cell is small, and they fixed $\sigma^2$ in all runs. Other groups have handled it in different ways:
- [Nguyen et al. 2024](https://arxiv.org/abs/2403.03220) imposed an informative lower limit.
- Andrews et al. fix $\sigma$ from a fixed-IC calibration run, at the *upper* end of its posterior, "to provide additional robustness to the mismatch of the model". They report an inferred noise about **1/4 of Poisson, "primarily due to the use of the CIC kernel"**. A linear bias model collapses, while their full bias model stabilises it ([2603.20855](https://arxiv.org/abs/2603.20855)).
- Rubira & Schmidt add non-Gaussian, density-dependent noise terms. With those terms "'sigma collapse' does not occur" and $\sigma$ stays near Poisson ([2511.05484](https://arxiv.org/abs/2511.05484)).

[Akitsu et al. 2025](https://arxiv.org/abs/2509.09673) show that a Gaussian field-level likelihood fit to non-Gaussian-noise data produces "significant biases". The project's posteriors are 10–45% too narrow on dense samples, which matches overconfidence from a mis-specified noise model. That is most likely model mismatch, not a sampler bug. Two tests separate the causes:
- Simulation-based calibration (SBC; [Talts et al.](https://arxiv.org/abs/1804.06788)) on self-consistent mocks.
- The fixed-parameter frequentist test of [Alokda et al. 2026](https://arxiv.org/abs/2605.00980), who warn that average coverage can pass while per-realisation widths fail. The Quijote ±100 ensembles are exactly that test.

No paper reports width miscalibration as a function of tracer density, so the project's 10–45% result appears new.

### LEFTfield gridding practice extends the project's supersample fix

[Stadler et al. Part I](https://arxiv.org/abs/2409.10937) state the project's problem almost verbatim. If the same kernel is used in data and model it cancels "to some extent", but "data and forward model contain different particle numbers … rendering the cancellation incomplete". NUFFT assignment on a supersampled grid converges "much faster" than CIC. [Nguyen et al. 2021](https://arxiv.org/abs/2011.06587) add that CIC "does not completely remove modes with $k>k_{\rm Ny}$", letting nonlinear modes slip in.

The project's 4× supersample, CIC deconvolution and Fourier truncation is close to LEFTfield practice. The literature suggests four further steps, in order of expected value:
1. Interlacing on the 4× mesh, which removes the dominant odd alias images ([Sefusatti et al. 2016](https://arxiv.org/abs/1512.07295)).
2. A separate sharp-$k$ cutoff $\Lambda$ on $\varepsilon$ ([Schmidt et al. 2020](https://arxiv.org/abs/2004.06707)).
3. Sizing the mesh for $\phi^2$, $\delta^2$, $K^2$ and $\delta\phi$ by the generalised Orszag rule ([Schmidt 2021](https://arxiv.org/abs/2012.09837)).
4. A spherical $|k|\le k_{\max}$ likelihood mask rather than the coarse-grid cube.

There is also a useful diagnostic. At $\Delta=62.5\,h^{-1}$Mpc the CIC window's $k^2$ coefficient is $\Delta^2/12\approx325\,(h^{-1}{\rm Mpc})^2$, about 15× the physical $R_L^2\approx19$ for $3\times10^{13}\,h^{-1}M_\odot$ halos. That physical value follows [Lazeyras & Schmidt 2019](https://arxiv.org/abs/1904.11294), who find $b_{\nabla^2\delta}\approx-R_L^2$. A fitted $b_{\nabla^2}$ much larger than $R_L^2$ therefore flags residual window mismatch, not halo physics. These are the notes' own estimates.

### AbacusPNG and PNG-UNIT are better Stage 3 targets than Quijote Rockstar

[AbacusPNG](https://arxiv.org/abs/2402.10881) has four advantages: 2 Gpc/h boxes, $f_{\rm NL}\in\{-100,-30,0,30,100\}$ with two phases each, CompaSO halos matched to DESI HODs, and light cones. $f_{\rm NL}=\pm30$ is near DR1 sensitivity, and matched phases allow differencing. The abstract says 9 boxes, while the text describes 10 plus two "pngbase" runs.

[PNG-UNIT](https://arxiv.org/abs/2312.12405) ($4096^3$ particles in 1 Gpc/h) resolves QSO-host masses around $5\times10^{12}$, where the standard-resolution Quijote Rockstar catalogues are incomplete.

[Gutiérrez Adame et al. 2025](https://arxiv.org/abs/2506.06200) find that PNG-term aliasing in initial conditions is ≤3% at the start and below 0.1% by $z=0$. Quijote's 2LPTPNG initial conditions are therefore not a source of mismatch with a de-aliased DISCO-DJ model. One more caution for comparisons with Andrews et al.: report $p$ together with halo finder, mass definition and weighting. SO and FoF halos differ in sign at low mass ([Biagetti et al. 2017](https://arxiv.org/abs/1611.04901)).

### Euclid's headline changed between v1 and A&A

The project brief quotes the arXiv v1 numbers: $\sigma(f_{\rm NL})=6.0/4.0/2.3$ at 250/125/62.5 $h^{-1}$Mpc, 2.6 with fixed bias at 125, 6.3 with $p=1$, and 5.0 with $b_\phi$ and $b_{\phi\delta}$ sampled. The published A&A 711, A94 abstract instead headlines **$\sigma=2.6$ at $\Delta L=31.25\,h^{-1}$Mpc with a cutoff $k=0.1\,h$/Mpc** ([arXiv:2412.11945](https://arxiv.org/abs/2412.11945); [published abstract](https://uwcscholar.uwc.ac.za/items/d874fd68-c38e-41ed-a2df-bba46f645ea1)). One notes file reports the 6.0/4.0/2.3 sequence from the A&A full text ([A&A](https://www.aanda.org/articles/aa/full_html/2026/07/aa53802-25/aa53802-25.html)), so the body table may survive alongside the new headline.

Quote the published value with its setup. Also check that the 2.6 headline is not being confused with the fixed-bias 2.6 at 125. All Euclid runs are self-consistent mocks, so they are an upper bound. The Quijote mismatch result is the appropriate calibration.

### Joint gradient samplers are untested on $f_{\rm NL}$

The two Andrews papers disagree on mixing. The 2023 paper reports $f_{\rm NL}$ decorrelating after about $10^3$ samples. The Euclid paper reports about **$10^4$ for every run with sampled bias**, yet "little to no correlation" between $f_{\rm NL}$ and bias ([2203.08838](https://arxiv.org/abs/2203.08838); [2412.11945](https://arxiv.org/abs/2412.11945)). This points to the Gibbs blocking itself as the cause. Slice moves on $f_{\rm NL}$ at fixed $\varepsilon$ are short, because $\phi^2$ couples the two strongly. That is the notes' inference, not a published diagnosis.

The key benchmark ([Simon, Lanusse & de Mattia 2025](https://arxiv.org/abs/2504.20130)) found unadjusted MCLMC at about **1005 model evaluations per effective sample** for cosmology, roughly 10× better than adjusted samplers. Fourier-space "Kaiser" preconditioning supplies most of that gain: removing it raises the cost to 3112, and leaving Fourier space raises it to 6714.

The project's about 250 gradients/ESS at $160^3$ is in the expected regime. Two checks would make it publishable:
- Report the energy-error variance per dimension (EEVPD) at both step sizes. Robnik et al. bound the bias of unadjusted samplers through this quantity ([2412.08876](https://arxiv.org/abs/2412.08876)).
- Cross-check against MAMS ([2503.01707](https://arxiv.org/abs/2503.01707)) on an $80^3$ problem. The bias bound averages over $\sim10^6$ field dimensions and can hide a bias in $f_{\rm NL}$ alone.

No paper benchmarks samplers for $f_{\rm NL}$ specifically. A Simon et al. PNG/DESI follow-up exists only as a [conference abstract](https://indico.ict.inaf.it/event/3125/contributions/22844).

### Stage 4 template sampling is mode deprojection in disguise

Marginalising linear template amplitudes with flat priors is mathematically equivalent to mode deprojection ([Kalus et al. 2016](https://arxiv.org/abs/1607.02417); [Elsner et al. 2016](https://arxiv.org/abs/1509.08933)). Its cost in $\sigma(f_{\rm NL})$ can therefore be forecast now, by adding template amplitudes to $\theta$ in the Stage 1 Schur-Fisher code.

Over-regression is the documented failure:
- Nine maps instead of three "more than doubles" $\sigma$ for DESI photometric LRGs ([Rezaie et al. 2024](https://arxiv.org/abs/2307.01753)).
- More than about 7 templates overfit DESI QSO targets ([Krolewski et al. 2024](https://arxiv.org/abs/2305.07650)).

BORG has two precedents for handling this inside the forward model:
- Explicit template block sampling ([Jasche & Lavaux 2017](https://arxiv.org/abs/1706.08971)).
- A "robust likelihood" that marginalises per-patch amplitudes ([Porqueres et al. 2019](https://arxiv.org/abs/1812.05113)).

No paper samples templates jointly with $f_{\rm NL}$, so this element is novel.

Two Stage 4 inputs are encouraging, and one adds work:
- Spectroscopic DR1 QSOs show **no excess large-scale power** after linear weights ([Chiarenza et al. 2026](https://arxiv.org/abs/2512.17865)).
- Ignoring the integral constraint biased DES-like $f_{\rm NL}$ by about 23 (1.8σ) ([Riquelme et al. 2023](https://arxiv.org/abs/2209.07187)). A field-level model avoids this if $n(z)$ is a model input, which removes the RIC, and if templates are sampled, which removes the AIC.
- Lensing magnification is forecast to bias DESI $f_{\rm NL}$ by >1σ if neglected ([Viljoen et al. 2021](https://arxiv.org/abs/2108.05746)), so it belongs in the Stage 4 budget.

## Bibliography part 1: field-level inference of PNG

### A. BORG and forward-model field-level $f_{\rm NL}$ inference

| Paper | arXiv | Relevance |
|---|---|---|
| Andrews, Jasche, Lavaux & Schmidt 2023, MNRAS 520, 5746 | [2203.08838](https://arxiv.org/abs/2203.08838) | First BORG $f_{\rm NL}$. $\sigma=8.78$ on SDSS-III-like mocks, 5.70 for Stage IV, 4.43 with fixed bias. Gibbs (HMC on $\varepsilon$, slice on bias and $f_{\rm NL}$), decorrelation in about $10^3$ samples. |
| Euclid Collab.: Andrews et al. 2024/2026, A&A 711, A94 | [2412.11945](https://arxiv.org/abs/2412.11945) | The brief's template (2LPT, 8 Gpc/h, $p=0.55$). $f_{\rm NL}$ correlation length about $10^4$. v1 vs published headline differ (see implications). Releases PNG mocks and curvature maps. |
| Andrews, Jasche, Lavaux, Coulton, Villaescusa-Navarro et al. 2026 | [2603.20855](https://arxiv.org/abs/2603.20855) | Only field-level $f_{\rm NL}$ test on N-body halos (Quijote-PNG FoF $M>10^{13.5}$, $z=0.5$, $32^3$). $\sigma\approx23$ vs P 43 and P+B 27–29, i.e. 1.3× over P+B. $p=1.3$ is tuned, $b_{\phi\delta}$ comes from UMF, $\sigma$ is fixed from a fixed-IC run, ESS is 1175–2731, and the truth lies in 1σ in about 68% of 30 runs. Bias frame (checked in the v1 PDF, Eq. 6 and text): $b_1, b_2, b_K, b_\nabla$ act on the evolved (redshift-space) $\delta_m$, while $\phi_g$ is evaluated at Lagrangian positions, so the bias is Eulerian apart from the $\phi$ terms. |
| Andrews, Loureiro, Jasche, McAlpine et al. 2026 | [2602.02363](https://arxiv.org/abs/2602.02363) | BORG LPT light cone on Quaia in $(10\,h^{-1}{\rm Gpc})^3$ at 39.1 Mpc/h, with conservative sky cuts. About 4σ cross-correlation with Planck lensing, no $f_{\rm NL}$. Template for the DR1 QSO geometry. |
| Euclid Collab.: Finelli, Akrami, Andrews et al. 2025 | [2507.15819](https://arxiv.org/abs/2507.15819) | Euclid P+B Fisher for local, equilateral and orthogonal shapes, plus a review section on field-level inference of local $f_{\rm NL}$. |
| Seljak, Aslanyan, Feng & Modi 2017, JCAP 12, 009 | [1706.06645](https://arxiv.org/abs/1706.06645) | MAP plus Laplace expansion over linear modes, including PNG parameters. The conceptual ancestor of the Stage 1 Schur-complement Fisher. |
| Hoyland, Saadeh, Koyama & Desmond 2026 | [2607.03514](https://arxiv.org/abs/2607.03514) | Field-level inference separating modified gravity from bias. An adjacent example of bias-vs-new-physics degeneracy breaking at the field level. |

### B. Field-level information bounds and reconstruction for $f_{\rm NL}$

| Paper | arXiv | Relevance |
|---|---|---|
| Chen, Green & Lee 2026 | [2603.22415](https://arxiv.org/abs/2603.22415) | Field-level Cramér–Rao bound for PNG. Multi-tracer scale-dependent bias beats conservative higher-point analyses but falls short of the optimum. A sanity anchor for the Stage 1 field Fisher. |
| Baumann & Green 2021 | [2112.14645](https://arxiv.org/abs/2112.14645) | Locality argument: primordial information lives in long-range correlations that gravity cannot create, which motivates map-level analysis. |
| McQuinn 2020 | [2008.12312](https://arxiv.org/abs/2008.12312) | Reconstruction fidelity is set by galaxy number, not by where shot noise exceeds signal. Bounds field-level gains for sparse QSOs. |
| Darwish, Foreman, Abidi, Baldauf, Sherwin et al. 2020 | [2007.08472](https://arxiv.org/abs/2007.08472) | Quadratic-estimator reconstruction of large-scale modes with PNG, giving tens-of-percent gains through an effective single-tracer multi-tracer. A mechanism the Schur Fisher should capture. |
| Shirasaki, Sugiyama, Takahashi & Kitaura 2021, PRD 103, 023506 | [2010.04567](https://arxiv.org/abs/2010.04567) | Post-reconstruction bispectrum has near-Gaussian covariance. BOSS-like $\Delta f_{\rm NL}\sim13$, 1.3–3.2× over pre-reconstruction. |
| Laguë, Madhavacheril, Smith, Ferraro & Schaan 2024 | [2411.08240](https://arxiv.org/abs/2411.08240) | kSZ 3D velocity reconstruction (ACT×SDSS) gives $f_{\rm NL}=-90^{+210}_{-350}$. A sample-variance-cancellation pathfinder. |
| Kokron 2025 | [2504.20025](https://arxiv.org/abs/2504.20025) | Zero-bias 21 cm. New noise-like terms need field-level reconstruction. Adjacent tracer. |
| Vislosky, Brown, Demina & Chaussidon 2025, A&A 704, A297 | [2508.13381](https://arxiv.org/abs/2508.13381) | DESI-adjacent density-moment alternative to two-point statistics. |
| Sharma, Sullivan, Akitsu & Ivanov 2025 | [2511.14677](https://arxiv.org/abs/2511.14677) | Field-level EFT measurement of equilateral $b_\psi$ with a $b_\psi(b_1)$ fit. The equilateral analogue of Sullivan & Chen (see M). |

### C. Neural, SBI and summary-hybrid "field-level" $f_{\rm NL}$

| Paper | arXiv | Relevance |
|---|---|---|
| Giri, Münchmeyer & Smith 2022 | [2205.12964](https://arxiv.org/abs/2205.12964) | A neural network estimates local $\sigma_8$ and correlates it with the large-scale field, giving 3.5× over the halo approach on matter. |
| Giri, Münchmeyer & Smith 2023/2025, PRD 112, 123544 | [2305.03070](https://arxiv.org/abs/2305.03070) | Large-scale modulation of small-scale statistics. Strong on matter, only marginal on Quijote halos. |
| Kvasiuk, Münchmeyer & Smith 2024 | [2410.01007](https://arxiv.org/abs/2410.01007) | On AbacusPNG, mass binning is already optimal. Concentration gives "a factor of a few" gain. ML does not lower shot noise. |
| Flöss & Meerburg 2023 | [2305.07018](https://arxiv.org/abs/2305.07018) | U-Net reconstruction of matter plus P+B gives 3.65× on local $f_{\rm NL}$ (matter field, $k_{\max}=0.52$). |
| Bottema, Flöss & Meerburg 2025, JCAP 08, 030 | [2502.11846](https://arxiv.org/abs/2502.11846) | Same approach on redshift-space halos: only 1.33× for local. Halo gains are far smaller than matter gains. |
| Bairagi & Wandelt 2025 | [2509.03165](https://arxiv.org/abs/2509.03165) | PatchNet hierarchical CNN summaries on Quijote. |
| Cagliari, Bairagi & Wandelt 2026, A&A 710, A12 | [2510.12715](https://arxiv.org/abs/2510.12715) | PatchNet plus P+B gains 30–45% on $f_{\rm NL}$ at $k_{\max}\sim0.1$ on Quijote-PNG halos. Comparable to BORG's 1.3×. |
| Peron, Jung, Liguori & Pietroni 2024 | [2403.17657](https://arxiv.org/abs/2403.17657) | Wavelet scattering beats P by about 27% on local $f_{\rm NL}$ but is about 15% weaker than P+B. |
| Wang & He 2024, PRD 111, L041302 | [2408.13876](https://arxiv.org/abs/2408.13876) | Nonlinear-transform (peak/valley) summaries on Quijote-PNG. |
| Zhou, Wang, Zhang, Peron, Deng et al. 2026 | [2610.04486](https://arxiv.org/abs/2610.04486) | Zel'dovich-inspired transform. Large gains for equilateral and orthogonal over $P_h$. |
| Rouhiainen & Münchmeyer 2022 | [2211.15161](https://arxiv.org/abs/2211.15161) | Normalising-flow denoising of non-Gaussian fields. |
| Melsen, Flöss & Meerburg 2024/2025, MNRAS | [2412.12377](https://arxiv.org/abs/2412.12377) | Spherical CNNs for CMB PNG. Adjacent only. |

## Bibliography part 2: constraining local $f_{\rm NL}$ with large-scale structure

### D. Theory foundations and reviews

| Paper | arXiv | Relevance |
|---|---|---|
| Dalal, Doré, Huterer & Shirokov 2008, PRD 77, 123514 | [0710.4560](https://arxiv.org/abs/0710.4560) | Discovery of $k^{-2}$ scale-dependent halo bias in N-body simulations. |
| Matarrese & Verde 2008, ApJL 677, L77 | [0801.4826](https://arxiv.org/abs/0801.4826) | Analytic scale-dependent halo bias. |
| Slosar, Hirata, Seljak, Ho & Padmanabhan 2008, JCAP 08, 031 | [0805.3580](https://arxiv.org/abs/0805.3580) | Peak-background split derivation that rests on UMF. EPS "recent mergers" is the origin of QSO $p=1.6$. $-29<f_{\rm NL}<70$ (95%). |
| Afshordi & Tolley 2008, PRD 78, 123507 | [0806.1046](https://arxiv.org/abs/0806.1046) | Independent derivation. NVSS×ISW hint $236\pm127$. |
| Desjacques, Seljak & Iliev 2009, MNRAS 396, 85 | [0811.2748](https://arxiv.org/abs/0811.2748) | Simulation test. Agreement at the lowest $k$ for $b>1.5$; the notes quote $k<0.01$ and $k<0.03$. Suppression up to 3× at $k=0.1$, and an excess for $b<1.5$. |
| Desjacques & Seljak 2010, CQG 27, 124011 | [1003.5020](https://arxiv.org/abs/1003.5020) | Review of LSS PNG searches. A second "Advances in Astronomy" DS review is unverified. |
| Desjacques, Jeong & Schmidt 2018, Phys. Rept. 733, 1 | [1611.09787](https://arxiv.org/abs/1611.09787) | Standard bias review: full basis, higher-derivative terms, stochasticity, and $b_\phi$/$b_{\phi\delta}$ (Sec. 7). |
| Biagetti 2019, Galaxies 7, 71 | [1906.12244](https://arxiv.org/abs/1906.12244) | Review linking inflationary correlators to halo statistics. |
| Meerburg et al. 2019, BAAS 51, 107 | [1903.04409](https://arxiv.org/abs/1903.04409) | Astro2020 PNG white paper. |
| Achúcarro et al. 2022 | [2203.08128](https://arxiv.org/abs/2203.08128) | Snowmass inflation white paper. The $\sigma(f_{\rm NL}^{\rm loc})\sim1$ target. |
| Planck Collab. (Akrami et al.) 2020, A&A 641, A9 | [1905.05697](https://arxiv.org/abs/1905.05697) | CMB reference $f_{\rm NL}^{\rm loc}=-0.9\pm5.1$. The go/no-go threshold of 5 equals Planck. |
| Hamaus, Seljak & Desjacques 2011, PRD 84, 083509 | [1104.2321](https://arxiv.org/abs/1104.2321) | Optimal weighting and multi-tracer baseline for $f_{\rm NL}$ from two-point statistics. |

### E. Spectroscopic measurements (SDSS, BOSS, eBOSS, DESI)

| Paper | arXiv | Relevance |
|---|---|---|
| Ross et al. 2013, MNRAS 428, 1116 | [1208.1491](https://arxiv.org/abs/1208.1491) | BOSS DR9. Stellar systematics alone fake $P(f_{\rm NL}>0)=99.5\%$; corrected result $-45<f_{\rm NL}<195$. A free systematic amplitude is "not fully degenerate" with $f_{\rm NL}$. |
| Castorina et al. 2019, JCAP 09, 010 | [1904.08859](https://arxiv.org/abs/1904.08859) | eBOSS DR14 QSO, first optimal redshift weights: $-51<f_{\rm NL}<21$ (95%). |
| Mueller, Percival & Ruggeri 2019, MNRAS 485, 4160 | [1702.05088](https://arxiv.org/abs/1702.05088) | Derives the optimal $f_{\rm NL}$ redshift weights (about 30% gain for QSOs). Background to OQE. |
| Mueller et al. 2022, MNRAS 514, 3396 | [2106.13725](https://arxiv.org/abs/2106.13725) | eBOSS DR16 QSO, $f_{\rm NL}=-12\pm21$. Neural-network cleaning plus weights give a 37% improvement. |
| Cagliari, Castorina, Bonici & Bianchi 2024, JCAP 08, 036 | [2309.15814](https://arxiv.org/abs/2309.15814) | Optimal DR16 QSO $P_0$: $-4<f_{\rm NL}<27$. Only about 10% gain vs Fisher, a hint of unknown systematics. |
| Cagliari, Barberi-Squarotti, Pardede, Castorina et al. 2025, JCAP 07, 043 | [2502.14758](https://arxiv.org/abs/2502.14758) | DR16 QSO P+B with RIC corrections (only the RIC mattered). 16% better than P. |
| D'Amico, Lewandowski, Senatore & Zhang 2022/2025, PRD 111, 063514 | [2201.11518](https://arxiv.org/abs/2201.11518) | BOSS EFT one-loop P+B: $f_{\rm NL}^{\rm loc}=52\pm34$. An earlier version may differ. |
| Cabass, Ivanov, Philcox, Simonović & Zaldarriaga 2022, PRD 106, 043506 (+ PRL 129, 021301) | [2204.01781](https://arxiv.org/abs/2204.01781), [2201.07238](https://arxiv.org/abs/2201.07238) | BOSS P+B: $-33\pm28$. The bispectrum adds 20% and breaks the non-Gaussian-bias degeneracy. Companion paper on single-field shapes. |
| Chaussidon, de Mattia, Yèche et al. 2025, JCAP 01, 135 | [2406.00191](https://arxiv.org/abs/2406.00191) | DR1 blinding with field-level weights mimicking scale-dependent bias. |
| **Chaussidon, Yèche, de Mattia, Payerne et al. 2025, JCAP 06, 029** | [2411.17623](https://arxiv.org/abs/2411.17623) | **Project baseline.** $P_0+P_2$ with RIC and AIC windows: $-3.6^{+9.0}_{-9.1}$. LRG baseline 0.6<z<1.1 (1,631,716). Clean-EZmock errors are the Stage 1 Part A targets. |
| **Chudaykin, Ivanov & Philcox 2025, PRD 113, 063552** | [2512.04266](https://arxiv.org/abs/2512.04266) | **Strongest DR1 LSS result**: EFT P+B $-0.1\pm7.4$; with CMB $-0.0\pm4.1$. $b_\phi$ treatment not in the abstract. |
| Rosado-Marín, Ross, Seo, Chaussidon et al. 2026 | [2604.05213](https://arxiv.org/abs/2604.05213) | DR1 LRG×QSO cross spectra: $2.1^{+8.8}_{-8.3}$ (about 9% gain). ELGs add nothing. |
| Brown, Levi, Randall, Chaussidon et al. 2026 | [2606.24651](https://arxiv.org/abs/2606.24651) | First DR1 configuration-space measurement, joint $-3\pm12$. Priors $p_{\rm LRG}=1.0\pm0.1$, $p_{\rm QSO}=1.6\pm0.1$. |

See also Fondi et al. 2026 (M) for the DR1 QSO $p$-prior result, and Chiarenza et al. 2026 (F) for DR1 QSO×lensing.

### F. Angular, Quaia, CMB-lensing, kSZ and multi-tracer

| Paper | arXiv | Relevance |
|---|---|---|
| Xia, Baccigalupi, Matarrese, Verde & Viel 2011, JCAP 08, 033 | [1104.5015](https://arxiv.org/abs/1104.5015) | $C_\ell$ multi-probe $48\pm20$, an apparent detection later attributed to systematics. |
| Giannantonio, Ross, Percival et al. 2014, PRD 89, 023511 | [1303.1349](https://arxiv.org/abs/1303.1349) | Multi-survey plus ISW: $5\pm21$. NVSS auto dropped for systematics. |
| Leistedt, Peiris & Roth 2014, PRL 113, 221301 | [1405.4315](https://arxiv.org/abs/1405.4315) | 800k photometric QSOs with blind mode projection: $-49<f_{\rm NL}<31$. LSST-like forecast about 5. |
| Ho et al. 2015, JCAP 05, 040 | [1311.2597](https://arxiv.org/abs/1311.2597) | 1.6M photometric QSOs plus LRGs: $-113\pm154$. |
| Agarwal, Ho & Shandera 2014, JCAP 02, 038 | [1311.2606](https://arxiv.org/abs/1311.2606) | Generalised $A_{\rm NL}/k^\alpha$ ansatz, $\alpha<2.0$. |
| Rezaie, Ross, Seo, Kong et al. 2024, MNRAS 532, 1902 | [2307.01753](https://arxiv.org/abs/2307.01753) | DESI photometric LRGs $34^{+24}_{-44}$. Nine imaging maps more than double $\sigma$ (Stage 4). |
| Storey-Fisher, Hogg, Rix et al. 2024, ApJ 964, 69 | [2306.17749](https://arxiv.org/abs/2306.17749) | Quaia catalogue (1.3M QSOs). No $f_{\rm NL}$. |
| Fabbian, Alonso, Storey-Fisher & Cornish 2026, JCAP 02, 056 | [2504.20992](https://arxiv.org/abs/2504.20992) | Quaia auto plus $\kappa$: $-20.5^{+19.0}_{-18.1}$ ($p=1$). |
| Bermejo-Climent, Hernández-Monteagudo et al. 2026, A&A 710, A360 | [2601.16948](https://arxiv.org/abs/2601.16948) | Quaia with angular redshift fluctuations plus lensing: $-3\pm14$, the best projected-statistics result. |
| Seljak 2009, PRL 102, 021302 | [0807.1770](https://arxiv.org/abs/0807.1770) | Multi-tracer cancellation of cosmic variance. |
| McDonald & Seljak 2009, JCAP 10, 007 | [0810.0323](https://arxiv.org/abs/0810.0323) | Multi-tracer extension including PNG. |
| Ferraro & Smith 2015, PRD 91, 043506 | [1408.3126](https://arxiv.org/abs/1408.3126) | Forecast $\sigma=6$, or 1.5 with mass-weighted halos. |
| Ferramacho, Santos, Jarvis & Camera 2014, MNRAS 442, 2511 | [1402.2290](https://arxiv.org/abs/1402.2290) | Radio multi-tracer $\sigma=2.2$–3.6. |
| Alonso & Ferreira 2015, PRD 92, 063525 | [1507.03550](https://arxiv.org/abs/1507.03550) | Optical plus radio multi-tracer forecasts. |
| Sullivan, Prijon & Seljak 2023, JCAP 08, 004 | [2303.08901](https://arxiv.org/abs/2303.08901) | Splitting samples by ML-predicted $b_\phi$: DESI LRG 3.7, ELG+LRG 1.5, ideal split <1. |
| Giannantonio & Percival 2014, MNRAS 441, L16 | [1312.5154](https://arxiv.org/abs/1312.5154) | Planck $\kappa$ × six catalogues: $12\pm21$. |
| Schmittfull & Seljak 2018, PRD 97, 123540 | [1710.09465](https://arxiv.org/abs/1710.09465) | CMB-S4 × LSST forecast. Losing sky overlap costs 1.5–2×. |
| McCarthy, Madhavacheril & Maniyar 2023, PRD 108, 083522 | [2210.01049](https://arxiv.org/abs/2210.01049) | CIB × $\kappa$, $\sigma\approx41$. |
| Krolewski, Percival, Ferraro, Chaussidon et al. 2024, JCAP 03, 021 | [2305.07650](https://arxiv.org/abs/2305.07650) | DESI QSO targets × Planck: $-26^{+45}_{-40}$. Excess autopower, 2.6% stellar contamination, >7 templates overfit. |
| Bermejo-Climent, Demina, Krolewski et al. 2025, A&A 698, A177 | [2412.10279](https://arxiv.org/abs/2412.10279) | DESI photometric LRG × Planck: $24^{+20}_{-21}$ with auto. Lensing is more stable against imaging systematics. |
| Chiarenza, Krolewski, Bonici, Chaussidon et al. 2026, JCAP 05, 098 | [2512.17865](https://arxiv.org/abs/2512.17865) | DR1 spectroscopic QSO × PR4: $2^{+28}_{-34}$. **No excess large-scale power in the DR1 QSO auto-correlation.** |
| Thornton, McCarthy, Embil Villagra & Sherwin 2026 | [2605.03783](https://arxiv.org/abs/2605.03783) | Dust-cleaned CIB × $\kappa$: $43\pm23$. |
| Payerne, d'Assignies, Yèche et al. 2026, A&A 709, A70 | [2511.22243](https://arxiv.org/abs/2511.22243) | UNIONS LBG × Planck forecast, $\sigma=20$–34. |
| Münchmeyer et al. 2019, PRD 100, 083508 | [1810.13424](https://arxiv.org/abs/1810.13424) | kSZ tomography forecast $\sigma\sim0.5$. |
| Krywonos, Hotinli & Johnson 2024 | [2408.05264](https://arxiv.org/abs/2408.05264) | Planck×unWISE kSZ: $-220\lesssim f_{\rm NL}\lesssim136$. |

### G. Power spectrum + bispectrum, Quijote-PNG summary studies and forecasts

| Paper | arXiv | Relevance |
|---|---|---|
| Moradinezhad Dizgah, Biagetti, Sefusatti, Desjacques & Noreña 2021, JCAP 05, 015 | [2010.14523](https://arxiv.org/abs/2010.14523) | Halo P+B gives >5× over P with loose priors. **UMF $b_\phi$/$b_{\phi\delta}$ priors shift $f_{\rm NL}$**, and Poisson noise also biases it. |
| Euclid Collab.: Linde, Moradinezhad Dizgah et al. 2026 | [2605.21436](https://arxiv.org/abs/2605.21436) | AbacusPNG HOD mocks: $B_\ell$ gives 29–46% over $P_\ell$. Proposes a physical $b_\phi$ prior. |
| Karagiannis, Lazanu, Liguori, Raccanelli et al. 2018, MNRAS 478, 1341 | [1801.09280](https://arxiv.org/abs/1801.09280) | RSD bispectrum Fisher, the brief's Tier 2 reference. |
| Coulton et al. 2023b, ApJ 943, 178 | [2206.15450](https://arxiv.org/abs/2206.15450) | Quijote-PNG halos: for local PNG, scale-dependent bias in P beats B, and small scales add little. |
| Jung, Karagiannis, Liguori, Baldi et al. 2022, ApJ 940, 71 | [2206.01624](https://arxiv.org/abs/2206.01624) | QML on matter: $\Delta f_{\rm NL}\sim16$ per $(h^{-1}{\rm Gpc})^3$. |
| Jung et al. 2023a, ApJ 948, 135 | [2211.07565](https://arxiv.org/abs/2211.07565) | QML on halos: 45 per Gpc³. Local information saturates at $k\sim0.2$. |
| Jung, Ravenni, Baldi, Coulton et al. 2023b, ApJ 957, 50 | [2305.10597](https://arxiv.org/abs/2305.10597) | Halo mass function breaks the $f_{\rm NL}$–$b_\phi$ degeneracy. Mentions a preliminary GNN field-level analysis. |
| Jung, Ravenni, Liguori, Baldi et al. 2024, ApJ 976, 109 | [2403.00490](https://arxiv.org/abs/2403.00490) | Marked and optimised summaries. Gains mostly for equilateral. |
| Giannantonio, Porciani, Carron, Amara & Pillepich 2012, MNRAS 422, 2854 | [1109.0958](https://arxiv.org/abs/1109.0958) | Euclid-like forecast about 3. |
| Doré et al. 2014 | [1412.4872](https://arxiv.org/abs/1412.4872) | SPHEREx science case. The $\sigma=0.5$ goal is quoted second-hand. |
| Heinrich, Doré & Krause 2024, PRD 109, 123511 | [2311.13082](https://arxiv.org/abs/2311.13082) | SPHEREx multi-tracer bispectrum, $\sigma=0.7$. |
| DESI Collaboration 2016 | [1611.00036](https://arxiv.org/abs/1611.00036) | DESI design. No $f_{\rm NL}$ number. |
| Schlegel et al. 2022 | [2209.04322](https://arxiv.org/abs/2209.04322) | MegaMapper Stage-5 concept. |
| **Bragança, Donath, Senatore & Zheng 2023** | [2307.04992](https://arxiv.org/abs/2307.04992) | **One-loop P+B forecast $\sigma=3.3$ for DESI** (0.26 for MegaMapper). The P+B bar the field-level method must beat. |

## Bibliography part 3: literature on the project's specific problems

### H. Differentiable forward-model engines and the BORG lineage

| Paper | arXiv | Relevance |
|---|---|---|
| Hahn, List & Porqueres 2024, JCAP 06, 063 | [2311.03291](https://arxiv.org/abs/2311.03291) | DISCO-EB: JAX Einstein–Boltzmann solver, per-mille agreement with CLASS and CAMB. |
| List, Hahn, Flöss & Winkler 2026, JCAP 06, 033 | [2510.05206](https://arxiv.org/abs/2510.05206) | DISCO-DJ PM: BullFrog, nuFFT forces, adjoint with step-independent memory, and a field-level $\sigma_8$ demo. **The de-aliasing and sheet options in CLAUDE.md were not independently verified** (full text not retrieved). |
| Rampf, List & Hahn 2025, JCAP 02, 020 | [2409.19049](https://arxiv.org/abs/2409.19049) | BullFrog integrator. |
| Hahn, Rampf & Uhlemann 2021, MNRAS 503, 426 | [2008.09124](https://arxiv.org/abs/2008.09124) | monofonIC nLPT with Orszag de-aliasing, the lineage of DISCO-DJ's nLPT. |
| Michaux, Hahn, Rampf & Angulo 2020, MNRAS 500, 663 | [2008.09588](https://arxiv.org/abs/2008.09588) | Minimising truncation and discreteness errors in initial conditions. |
| Abel, Hahn & Kaehler 2012, MNRAS 427, 61 | [1111.3944](https://arxiv.org/abs/1111.3944) | Sheet tessellation density estimator. |
| Hahn, Abel & Kaehler 2013, MNRAS 434, 1171 | [1210.6652](https://arxiv.org/abs/1210.6652) | Sheet-based collisionless simulation. |
| Hahn & Angulo 2016, MNRAS 455, 1115 | [1501.01959](https://arxiv.org/abs/1501.01959) | Adaptive phase-space element method. |
| Stücker, Hahn, Angulo & White 2020, MNRAS 495, 4943 | [1909.00008](https://arxiv.org/abs/1909.00008) | Sheet algorithms. |
| Li, Lu, Modi, Jamieson et al. 2022 | [2211.09958](https://arxiv.org/abs/2211.09958) | pmwd JAX PM. |
| Li, Modi, Jamieson, Zhang, Lu et al. 2022 | [2211.09815](https://arxiv.org/abs/2211.09815) | Adjoint method for differentiable simulation. |
| Modi, Lanusse & Seljak 2021, Astron. Comput. 37, 100505 | [2010.11847](https://arxiv.org/abs/2010.11847) | FlowPM. |
| Lanzieri, Lanusse & Starck 2022 (venue [UNVERIFIED]) | [2207.05509](https://arxiv.org/abs/2207.05509) | Hybrid neural PM corrections behind JaxPM pipelines. JaxPM itself has no arXiv paper. |
| Tassev, Zaldarriaga & Eisenstein 2013, JCAP 06, 036 | [1301.0322](https://arxiv.org/abs/1301.0322) | COLA, used as tCOLA in BORG. |
| Jasche & Wandelt 2013, MNRAS 432, 894 | [1203.3639](https://arxiv.org/abs/1203.3639) | Original BORG (2LPT plus HMC). |
| Jasche & Lavaux 2019, A&A 625, A64 | [1806.11117](https://arxiv.org/abs/1806.11117) | BORG-PM with CIC assignment and block sampling. |
| Stopyra, Peiris, Pontzen, Jasche & Lavaux 2024, MNRAS 527, 1244 | [2304.09193](https://arxiv.org/abs/2304.09193) | 20-step COLA in BORG: accurate at large scales, not converged at virial masses. |
| Doeser, Jamieson, Stopyra, Lavaux, Leclercq & Jasche 2024, MNRAS 535, 1258 | [2312.09271](https://arxiv.org/abs/2312.09271) | V-net emulator inside BORG HMC. |
| Doeser & Jasche 2026 | [2606.10023](https://arxiv.org/abs/2606.10023) | Reliability of generative-model posteriors vs HMC references. |
| Doeser, Ata & Jasche 2025, MNRAS 542, 1403 | [2502.13243](https://arxiv.org/abs/2502.13243) | LULO: learned optimiser for non-differentiable simulators. |
| McAlpine, Jasche, Ata, Lavaux et al. 2025, MNRAS 540, 716 | [2505.10682](https://arxiv.org/abs/2505.10682) | Manticore I. Source of the white-noise variance prior used by Andrews et al. 2026. |
| McAlpine, Jasche, Lavaux, Doeser & Loureiro 2026, MNRAS 550, stag1366 | [2606.10020](https://arxiv.org/abs/2606.10020) | Manticore-Deep tiled BORG, the high-resolution cost reference. |

### I. Grid window, aliasing and mass assignment

| Paper | arXiv | Relevance |
|---|---|---|
| Hockney & Eastwood 1981/1988, *Computer Simulation Using Particles* [UNVERIFIED bibliographic details] | — | Origin of the NGP/CIC/TSC windows and interlacing. Cited by Andrews et al. 2026 for CIC. |
| Jing 2005, ApJ 620, 559 | [astro-ph/0409240](https://arxiv.org/abs/astro-ph/0409240) | Alias-sum formula: window and alias images enter together. It predicts why discrete halos and a one-particle-per-cell model have different alias sums. |
| Cui, Liu, Yang, Wang, Feng & Springel 2008, ApJ 687, 738 | [0804.0070](https://arxiv.org/abs/0804.0070) | Near-top-hat-in-Fourier (wavelet) kernels as an alternative to CIC plus deconvolution. |
| Colombi, Jaffe, Novikov & Pichon 2009, MNRAS 393, 511 | [0811.0313](https://arxiv.org/abs/0811.0313) | Foldover and Taylor alias reduction. Guides the supersampling factor. |
| Sefusatti, Crocce, Scoccimarro & Couchman 2016, MNRAS 460, 3624 | [1512.07295](https://arxiv.org/abs/1512.07295) | **Interlacing plus fourth-order kernels give <0.01% to Nyquist.** The next refinement for the 4× mesh. |
| Hand, Feng, Beutler, Li et al. 2018, AJ 156, 160 | [1712.05834](https://arxiv.org/abs/1712.05834) | nbodykit reference implementation of compensated windows and interlacing, for checking the JAX deconvolution. |
| Schmidt 2021, JCAP 04, 033 | [2012.09837](https://arxiv.org/abs/2012.09837) | nth-order Lagrangian forward model. **"No mode left behind" and the generalised Orszag rule** for sizing grids for $\phi^2$, $\delta^2$, $K^2$ and $\delta\phi$. |
| **Stadler, Schmidt & Reinecke 2025, JCAP 04, 089 (Part I)** | [2409.10937](https://arxiv.org/abs/2409.10937) | **Most relevant to the gridding problem.** Identical kernels cancel only partially for halos vs model. NUFFT converges faster than CIC. Truncation vs aliasing errors are separated, and Lagrangian weighted displacements are covered (§3.5). |
| Stadler, Schmidt, Reinecke & Esposito 2025, JCAP 11, 055 (Part II) | [2411.04513](https://arxiv.org/abs/2411.04513) | Redshift-space numerics. $f$ from halos to <1% with third-order bias. |
| Gutiérrez Adame, Avila, Gonzalez-Perez, Hahn et al. 2025, A&A 705, A20 | [2506.06200](https://arxiv.org/abs/2506.06200) | PNG-term aliasing in initial conditions is ≤3% and drops below 0.1% by $z=0$. 3LPT from $z\simeq11.5$ is recommended. Quijote initial-condition aliasing is not a mismatch source. |

See also Nguyen et al. 2021 (K) on CIC leakage of super-Nyquist modes, and Schmidt et al. 2020 (J) on the $\Lambda$ cut on initial conditions.

### J. EFT likelihood (LEFTfield), field-level information content and model-mismatch validation

| Paper | arXiv | Relevance |
|---|---|---|
| Schmidt, Elsner, Jasche, Nguyen & Lavaux 2019, JCAP 01, 042 | [1808.02002](https://arxiv.org/abs/1808.02002) | EFT forward model with a sharp-$k$ cutoff. The brief's "cosmological parameters from the field level" title [not found] maps here. |
| Elsner, Schmidt, Jasche, Lavaux & Nguyen 2020, JCAP 01, 029 | [1906.07143](https://arxiv.org/abs/1906.07143) | First EFT-likelihood inference on halos. |
| Cabass & Schmidt 2020, JCAP 04, 042 | [1909.04022](https://arxiv.org/abs/1909.04022) | Functional derivation of the EFT likelihood, with comments "with a particular eye towards primordial non-Gaussianity". |
| Schmidt, Cabass, Jasche & Lavaux 2020, JCAP 11, 008 | [2004.06707](https://arxiv.org/abs/2004.06707) | **Sharp cut $\Lambda$ on initial conditions** reduces the $\sigma_8$ bias several-fold. The project lacks a separate $\Lambda$ on $\varepsilon$. |
| Cabass 2020 | [2007.14988](https://arxiv.org/abs/2007.14988) | EFT likelihood in redshift space. |
| Schmidt 2021, JCAP 04, 032 | [2009.14176](https://arxiv.org/abs/2009.14176) | Real-space EFT likelihood that allows survey windows. Unbiased $\sigma_8$ at 2%. |
| Kostić, Nguyen, Schmidt & Reinecke 2023, JCAP 07, 063 | [2212.07875](https://arxiv.org/abs/2212.07875) | HMC plus slice consistency tests. $\Lambda_0>\Lambda$ mismatch recovered without bias. Linear-case analytic posterior check, the template for a Tier-1 sampler test. |
| Stadler, Schmidt & Reinecke 2023, JCAP 10, 069 | [2303.09876](https://arxiv.org/abs/2303.09876) | LEFTfield RSD. Velocity-bias expansion needed. Unbiased $f$ for $0.08\le\Lambda\le0.20$. |
| Babić, Schmidt & Tucci 2022, JCAP 08, 007 | [2203.06177](https://arxiv.org/abs/2203.06177) | BAO with the EFT likelihood. |
| Babić, Schmidt & Tucci 2025, JCAP 11, 066 | [2407.01524](https://arxiv.org/abs/2407.01524) | Field-level BAO under deliberate cutoff mismatch, <2% bias. |
| Babić, Schmidt & Tucci 2026, PRD 113, L061304 | [2505.13588](https://arxiv.org/abs/2505.13588) | Halo BAO with sampled initial conditions, bias and noise ($\sigma_\epsilon$ plus $k^2$). 1.2–1.4× over reconstruction. |
| Tucci & Schmidt 2024, JCAP 05, 063 | [2310.03741](https://arxiv.org/abs/2310.03741) | SBI P+B with the LEFTfield model, the P+B arm of the PRL comparison. |
| Ke, Wang & He 2025, JCAP 04, 064; JCAP 09, 029 | [2502.16082](https://arxiv.org/abs/2502.16082), [2505.10915](https://arxiv.org/abs/2505.10915) | **The only EFT-likelihood work with PNG.** It finds irreducible field-dependent likelihood terms under non-Gaussian initial conditions. |
| Cabass, Simonović & Zaldarriaga 2023 | [2307.04706](https://arxiv.org/abs/2307.04706) | In the small-noise limit, perturbative field-level inference equals the n-point functions at the same order. |
| Nguyen, Schmidt, Tucci, Reinecke & Kostić 2024, PRL 133, 221006 | [2403.03220](https://arxiv.org/abs/2403.03220) | Field-level $\sigma_8$ 3.5–5.2× tighter than SBI P+B on halos. Needed an informative noise lower bound. SBC run only on the SBI arm. |
| Schmidt 2025, JCAP 09, 056 | [2504.15351](https://arxiv.org/abs/2504.15351) | An $m$th-order model captures $n\le m+1$-point information. Shows how to estimate the bias from an incomplete model. |
| Spezzati, Marinucci & Simonović 2026, PRD 113, 063546 | [2507.05378](https://arxiv.org/abs/2507.05378) | P+B+T matches field-level inference on large scales. The trispectrum adds only 20–30%. |
| Akitsu, Simonović, Chen, Cabass & Zaldarriaga 2025 | [2509.09673](https://arxiv.org/abs/2509.09673) | Field-level ≈ P+B. **A Gaussian likelihood on non-Gaussian noise biases parameters.** |
| Pietroni & Schmidt 2026, JCAP 09, 011 | [2604.25385](https://arxiv.org/abs/2604.25385) | General field-level Fisher decomposed into connected correlators. Closest formal analogue to the Stage 1 Schur Fisher. |
| Nikolac, Schmidt & Tucci 2026 | [2606.18227](https://arxiv.org/abs/2606.18227) | Summaries lose information as coupling grows and noise falls. Dense samples are where field-level gains and calibration problems both appear. |
| Stadler 2026 | [2609.16719](https://arxiv.org/abs/2609.16719) | DESI-LRG-like mocks: field-level about 2× over P+B on amplitude in ΛCDM. |
| Krause, Kobayashi, Salcedo, Ivanov et al. (Beyond-2pt) 2025, ApJ 990, 99 | [2405.02252](https://arxiv.org/abs/2405.02252) | Blind HOD challenge including LEFTfield ($\sigma_8$ only). A sigma-collapse instance per Rubira & Schmidt. |
| Leclercq & Heavens 2021, MNRAS 506, L85 | [2103.04158](https://arxiv.org/abs/2103.04158) | Accuracy and precision of correlation functions vs field-level inference. |
| Leclercq 2025 (lecture notes) | [2509.13435](https://arxiv.org/abs/2509.13435) | Review entry point for field-level inference. |
| D'Amico, Senatore, Zhang & Nishimichi (JCAP 2024) | [2110.00016](https://arxiv.org/abs/2110.00016) | Owns the title "Taming redshift-space distortion effects" that the brief attributes to "Nguyen et al. 2021" [not found as a field-level paper]. The intended paper is 2011.06587 (K). |

### K. Higher-derivative bias, stochasticity and sigma collapse

| Paper | arXiv | Relevance |
|---|---|---|
| **Nguyen, Schmidt, Lavaux & Jasche 2021, JCAP 03, 058** | [2011.06587](https://arxiv.org/abs/2011.06587) | **Coined "sigma collapse"** and fixed $\sigma^2$. CIC leaks super-Nyquist modes. The IC amplitude bias depends strongly on the bias model and likelihood. Its §4.2 is the closest BORG analogue to Stage 3. |
| **Rubira & Schmidt 2026, JCAP 08, 023** | [2511.05484](https://arxiv.org/abs/2511.05484) | **Noise-field likelihood with non-Gaussian, density-dependent stochasticity.** Sigma collapse disappears on halos. The principled Stage 3 fix. |
| Rubira & Schmidt 2024, JCAP 10, 092 | [2404.16929](https://arxiv.org/abs/2404.16929) | Noise parameters run with $\Lambda$, so the effective noise depends on the grid and should not be fixed at $1/\bar n$. |
| Rubira & Schmidt 2024, JCAP 01, 031 | [2307.15031](https://arxiv.org/abs/2307.15031) | Galaxy-bias renormalisation group. |
| Cabass & Schmidt 2020, JCAP 07, 051 | [2004.00617](https://arxiv.org/abs/2004.00617) | Stochasticity of bias coefficients at all orders, i.e. density-dependent noise. |
| Lazeyras & Schmidt 2019, JCAP 11, 041 | [1904.11294](https://arxiv.org/abs/1904.11294) | $b_{\nabla^2\delta}<0$, roughly $-R_L^2(M)$. A physical prior and diagnostic for $b_{\nabla^2}$. |
| Fujita, Mauerhofer, Senatore, Vlah & Angulo 2020, JCAP 01, 009 | [1609.00717](https://arxiv.org/abs/1609.00717) | Very massive (QSO-like) tracers need higher-derivative biases. |
| Abidi & Baldauf 2018, JCAP 07, 029 | [1802.07622](https://arxiv.org/abs/1802.07622) | Cubic halo bias in Eulerian and Lagrangian space. |
| Hamaus, Seljak, Desjacques, Smith & Baldauf 2010, PRD 82, 043515 | [1004.5377](https://arxiv.org/abs/1004.5377) | High-mass halos are sub-Poisson, and mass weighting minimises stochasticity. Explains why $\sigma$ "wants" to drop for dense samples. |
| Baldauf, Seljak, Smith, Hamaus & Desjacques 2013, PRD 88, 083507 | [1305.2917](https://arxiv.org/abs/1305.2917) | Exclusion makes stochasticity scale-dependent even at $k<0.1$. |
| Schmittfull, Simonović, Assassi & Zaldarriaga 2019, PRD 100, 043514 | [1811.10640](https://arxiv.org/abs/1811.10640) | Shifted operators give near-Poisson white residuals. Linear bias gives residuals several times Poisson. |
| Schmittfull, Simonović, Ivanov, Philcox & Zaldarriaga 2021, JCAP 05, 059 | [2012.03334](https://arxiv.org/abs/2012.03334) | Redshift-space residual noise has a $k^2\mu^2$ term. Needs an anisotropic noise term. |
| Obuljen, Simonović, Schneider & Feldmann 2023, PRD 108, 083528 | [2207.12398](https://arxiv.org/abs/2207.12398) | Field-level noise is well below the P(k)-inferred shot noise: they are different quantities. |
| Kokron, DeRose, Chen, White & Wechsler 2022, MNRAS 514, 2198 | [2112.00012](https://arxiv.org/abs/2112.00012) | HEFT priors on non-Poisson stochasticity. Host halos are sub-Poisson. |
| Ivanov, Cuesta-Lazaro, Mishra-Sharma, Obuljen & Toomey 2024, PRD 110, 063538 | [2402.13310](https://arxiv.org/abs/2402.13310) | HOD-informed priors on EFT bias and noise. A template for $b_{\nabla^2}$ and noise priors. |

### L. Lagrangian vs Eulerian bias at the field level

| Paper | arXiv | Relevance |
|---|---|---|
| Matsubara 2008, PRD 78, 083519 | [0807.1733](https://arxiv.org/abs/0807.1733) | Foundational Lagrangian-bias perturbation theory with RSD. |
| Modi, Castorina & Seljak 2017, MNRAS 472, 3959 | [1612.01621](https://arxiv.org/abs/1612.01621) | Lagrangian halo-bias estimators. |
| Modi, Chen & White 2020, MNRAS 492, 5754 | [1910.07097](https://arxiv.org/abs/1910.07097) | HEFT origin. Smallest gains for high-bias, high-$z$ tracers, the QSO regime. |
| Kokron, DeRose, Chen, White & Wechsler 2021, MNRAS 505, 1422 | [2101.11014](https://arxiv.org/abs/2101.11014) | Aemulus-ν HEFT emulator. |
| Zennaro, Angulo, Pellejero-Ibáñez et al. 2023, MNRAS 524, 2407 | [2101.12187](https://arxiv.org/abs/2101.12187) | BACCO hybrid Lagrangian bias with a Laplacian term. |
| Chen, Vlah, Castorina & White 2021, JCAP 03, 100 | [2012.04636](https://arxiv.org/abs/2012.04636) | RSD in Lagrangian perturbation theory. |
| Barreira, Lazeyras & Schmidt 2021, JCAP 08, 029 | [2105.02876](https://arxiv.org/abs/2105.02876) | Fixed-IC field-level bias fits on TNG. A cheap diagnostic to reuse on Quijote. |
| Lazeyras, Barreira & Schmidt 2021, JCAP 10, 063 | [2106.14713](https://arxiv.org/abs/2106.14713) | Field-level $b_2$, $b_K$ and assembly bias. Feeds the $b_2$ input to the $b_{\phi\delta}$ formula. |

Nguyen et al. 2024 (J) note that the two bases are equivalent at fixed order and that a Lagrangian basis removes the need for an evolved-field cut. Stadler et al. Part I (I) analyse weighted-displacement assignment. The inference for the project: one-particle-per-cell Lagrangian weights make the window mismatch hit harder than an Eulerian basis would.

### M. $b_\phi$ and $p$ calibration (halos, galaxies, QSOs)

| Paper | arXiv | Relevance |
|---|---|---|
| Grossi, Verde, Carbone, Dolag et al. 2009, MNRAS 398, 321 | [0902.2013](https://arxiv.org/abs/0902.2013) | $\delta_c\to q\delta_c$ with $q\approx0.75$ for FoF halos. |
| Pillepich, Porciani & Hahn 2010, MNRAS 402, 191 | [0811.4176](https://arxiv.org/abs/0811.4176) | Universal FoF mass function under PNG. PBS inaccurate, so a fitting formula is given. |
| Reid, Verde, Dolag, Matarrese & Moscardini 2010, JCAP 07, 013 | [1004.1637](https://arxiv.org/abs/1004.1637) | First non-Gaussian assembly bias: $b_\phi$ varies by up to 100% with formation history. |
| Wagner & Verde 2012, JCAP 03, 002 | [1102.3229](https://arxiv.org/abs/1102.3229) | Halo bias with generic PNG. Needs simulation calibration. |
| Scoccimarro, Hui, Manera & Chan 2012, PRD 85, 083002 | [1108.5512](https://arxiv.org/abs/1108.5512) | Non-universal PBS bias formula. Also the 2LPTPNG IC paper used by Quijote-PNG. |
| Baldauf, Seljak, Senatore & Zaldarriaga 2016, JCAP 09, 007 | [1511.01465](https://arxiv.org/abs/1511.01465) | Curvature simulations: halo response departs from UMF by 8–20%. |
| Biagetti, Lazeyras, Baldauf, Desjacques & Schmidt 2017, MNRAS 468, 3277 | [1611.04901](https://arxiv.org/abs/1611.04901) | $b_\phi$ equals the response to $A_s$. High-mass halos are 10–15% below UMF. SO and FoF differ at low mass. |
| Barreira, Cabass, Schmidt, Pillepich & Nelson 2020, JCAP 12, 013 | [2006.09368](https://arxiv.org/abs/2006.09368) | TNG stellar-mass selection gives $p\in[0.4,0.7]$. Origin of Euclid's $p=0.55$. |
| Barreira 2020, JCAP 12, 031 | [2009.06622](https://arxiv.org/abs/2009.06622) | The $f_{\rm NL}$–$p$ posterior is bimodal. A wrong $p$ gives 0.5–1σ bias. $\Delta p\approx0.5$ is sufficient. |
| **Barreira 2022b, JCAP 11, 013** | [2205.05673](https://arxiv.org/abs/2205.05673) | BOSS $\sigma$ varies about 10× with $b_\phi(b_1)$ (16±16 to 230±226). **Wide $b_\phi$ priors cause projection biases.** |
| Barreira & Krause 2023, JCAP 10, 044 | [2302.09066](https://arxiv.org/abs/2302.09066) | Multi-tracer depends on $b_1^Bb_\phi^A-b_1^Ab_\phi^B$. More robust to $b_\phi$ mis-specification. |
| Lazeyras, Barreira, Schmidt & Desjacques 2023, JCAP 01, 023 | [2209.07251](https://arxiv.org/abs/2209.07251) | Concentration terciles at $b_1\approx2$ give $b_\phi\approx-1$, 3 and 9. BOSS $\sigma=165$, 44, 19. |
| Lucie-Smith, Barreira & Schmidt 2023, MNRAS 524, 1746 | [2304.09880](https://arxiv.org/abs/2304.09880) | CNN halo-formation surrogate. Concentration strongly affects $b_\phi$. |
| Shiveshwarkar, LoVerde, Hirata & Jamieson 2025, PRD 113, 103523 | [2508.11798](https://arxiv.org/abs/2508.11798) | Separate-universe origin of non-universality. The modified-$\delta_c$ basis for Andrews' $p=1.3$ (not read in full). |
| **Sullivan & Chen 2024/2025, JCAP 03, 016** | [2410.18039](https://arxiv.org/abs/2410.18039) | **First field-level $b_\phi$, $b_{\phi\delta}$.** $0.8\times$UMF fits halos. Marginalising degrades $f_{\rm NL}$ 2–3× and **biases it on halos**. Quijote halos deviate more than FastPM ones. |
| Hadzhiyska & Ferraro 2025, PRD 111, 103521 | [2501.14873](https://arxiv.org/abs/2501.14873) | Regressing $b_\phi$ on HEFT biases (mainly $b_\nabla$) cuts the uncertainty by >70–80% for DESI-like LRG and QSO samples. |
| Sullivan & Seljak 2025, PRD 112, 083522 | [2503.21736](https://arxiv.org/abs/2503.21736) | $b_\phi$ from the time evolution of tracer density. |
| Dalal & Percival 2025/2026, JCAP 04, 037 | [2503.21024](https://arxiv.org/abs/2503.21024) | $b_\Phi$ from $d\ln\bar n/dz$. Worst discrepancy about 40% near $z\sim0$. |
| Nascimento & Dalal 2026 | [2607.18229](https://arxiv.org/abs/2607.18229) | Time-evolution estimator tested in CAMELS, extended to photometric selections. |
| Marinucci, Desjacques & Benson 2023, MNRAS 524, 325 | [2303.10337](https://arxiv.org/abs/2303.10337) | Galacticus: large assembly bias for colour selection, none for Hα. |
| Fondi, Verde, Villaescusa-Navarro, Baldi et al. 2024, JCAP 02, 048 | [2311.10088](https://arxiv.org/abs/2311.10088) | A photometric proxy for assembly bias gives a $p$ prior with mild $\sigma$ cost. |
| **Fondi, Verde, Chaussidon et al. 2026, JCAP 06, 062** | [2602.12357](https://arxiv.org/abs/2602.12357) | **DESI QSO $\bar p\simeq1.41$** from TNG300 plus EPS, with weak $z$-dependence. DR1 $-3.3\pm9.2$. Fig. 6 percentiles (about ±0.1–0.2) were read approximately. |
| Perez, Genel, Krause & Somerville 2026 | [2602.04987](https://arxiv.org/abs/2602.04987) | CAMELS-SAM separate universe. sSFR selections robust. Table of the $p$ assumed in past analyses. |
| Moore, Perez & Krause 2026 | [2604.21790](https://arxiv.org/abs/2604.21790) | ELG-like $\sigma_{b_\phi}$ cut from 0.69 to 0.02–0.08 by conditioning on SMF/SHMR. |
| Yu & Nguyen 2026 | [2607.01314](https://arxiv.org/abs/2607.01314) | $b_\phi$ priors from DESI EDR HOD posteriors give unbiased $f_{\rm NL}$ even with assembly bias. |
| Merino, Avila, Gutiérrez Adame et al. 2026, A&A 707, L15 | [2602.04698](https://arxiv.org/abs/2602.04698) | Bias-zero tracers have $b_\phi\approx0$. QSO and LRG HODs are roughly universal for $b_1>1$. |
| Voivodic & Barreira 2021, JCAP 05, 069 | [2012.04637](https://arxiv.org/abs/2012.04637) | HOD responses to long modes. Relevant to HOD-based priors. |

### N. $b_{\phi\delta}$ and second-order PNG bias

| Paper | arXiv | Relevance |
|---|---|---|
| Giannantonio & Porciani 2010, PRD 81, 063530 | [0911.0017](https://arxiv.org/abs/0911.0017) | Multivariate $(\delta,\phi)$ bias expansion. |
| Baldauf, Seljak & Senatore 2011, JCAP 04, 006 | [1011.1513](https://arxiv.org/abs/1011.1513) | Tree-level halo bispectrum with $b_{\phi\delta}$-type amplification. |
| Schmidt & Kamionkowski 2010, PRD 82, 103002 | [1008.0638](https://arxiv.org/abs/1008.0638) | Generalised $b_\Phi$ for arbitrary bispectra. |
| Assassi, Baumann & Schmidt 2015, JCAP 12, 043 | [1510.03723](https://arxiv.org/abs/1510.03723) | Complete renormalised PNG bias expansion. |
| Tellarini, Ross, Tasinato & Wands 2015, JCAP 07, 004; 2016, JCAP 06, 014 | [1504.00324](https://arxiv.org/abs/1504.00324), [1603.06814](https://arxiv.org/abs/1603.06814) | Tree-level PNG bispectrum with $b_{\phi\delta}$, tidal terms and RSD. Inputs for the Tier 2 cross-check. |
| **Barreira 2022a, JCAP 01, 033** | [2107.06887](https://arxiv.org/abs/2107.06887) | **UMF overpredicts halo $b_{\phi\delta}$ by up to about 3.** Fit $3.85-9.49b_1+3.44b_1^2$. $b_\phi\approx0.85\times$UMF. QSO-like $\dot M_{\rm BH}$ selection does not follow $p=1.6$. |
| Lazeyras, Wagner, Baldauf & Schmidt 2016 [UNVERIFIED here] | [1511.01096](https://arxiv.org/abs/1511.01096) | $b_2(b_1)$ fit used for the UMF $b_{\phi\delta}$ estimate. Coefficients quoted from memory. |

See Moradinezhad Dizgah et al. 2021 (G) for the $f_{\rm NL}$ shift under UMF $b_{\phi\delta}$, and Sullivan & Chen 2024 (M) for the field-level $b_{\phi\delta}$.

### O. PNG simulation suites and initial-condition codes (Stage 3 targets)

| Paper | arXiv | Relevance |
|---|---|---|
| Coulton, Villaescusa-Navarro, Jamieson, Baldi et al. 2023a, ApJ 943, 64 | [2206.01619](https://arxiv.org/abs/2206.01619) | Quijote-PNG: 1 Gpc/h, $512^3$, 500 realisations per shape and sign at ±100, 2LPTPNG at $z=127$. |
| **Gutiérrez Adame, Avila, Gonzalez-Perez, Yepes et al. 2024, A&A 689, A69** | [2312.12405](https://arxiv.org/abs/2312.12405) | **PNG-UNITsims**: $4096^3$ in 1 Gpc/h, $f_{\rm NL}=100$, fixed-and-matched. $p=0.955\pm0.013$. Wrong $p$ biases DESI-like $f_{\rm NL}$ by up to 2σ. Resolves QSO hosts. |
| **Hadzhiyska, Garrison, Eisenstein & Ferraro 2024, PRD 109, 103530** | [2402.10881](https://arxiv.org/abs/2402.10881) | **AbacusPNG**: 2 Gpc/h, $f_{\rm NL}\in\{0,\pm30,\pm100\}$ × 2 phases, CompaSO, light cones. UMF overpredicts at $b_1>1.5$ ($c\approx0.9$). The notes disagree on the $b_\phi$-uncertainty $\sigma$ values (21/6/22 vs 15/8/7 for LRG $z=0.5$, LRG $z=0.8$, QSO $z=1.4$); check the paper. |
| Biagetti, Cole & Shiu 2021, JCAP 04, 061 | [2009.04819](https://arxiv.org/abs/2009.04819) | Uses the Eos suite (2 Gpc/h, $1536^3$, $f_{\rm NL}=10,\pm250$; specs from a search summary [UNVERIFIED]). No Eos release paper exists. |
| Wagner, Verde & Boubekeur 2010, JCAP 10, 022 | [1006.5793](https://arxiv.org/abs/1006.5793) | Generic-bispectrum IC method. |
| Fondi, Verde, Baldi, Coulton et al. 2025/2026, JCAP 08, 020 | [2508.01855](https://arxiv.org/abs/2508.01855) | GENGARS initial conditions for arbitrary bispectra, benchmarked against 2LPTPNG. |
| Crocce, Pueblas & Scoccimarro 2006, MNRAS 373, 369 | [astro-ph/0606505](https://arxiv.org/abs/astro-ph/0606505) | 2LPTic transients. |
| Baldi, Fondi, Karagiannis, Moscardini et al. 2024, JCAP 11, 053 | [2407.06641](https://arxiv.org/abs/2407.06641) | Scale-dependent $f_{\rm NL}(k)$ simulations. |
| Calles, Contardo, Noreña, Yip & Shiu 2025/2026, JCAP 08, 051 | [2512.09852](https://arxiv.org/abs/2512.09852) | PNG-pmwd: >20,000 fast halo catalogues. Transfer to Quijote degrades at small scales. The only PNG use of a differentiable PM code. |
| Euclid: Breton, Fosalba, Avila, Baldi et al. 2026 | [2603.13148](https://arxiv.org/abs/2603.13148) | Public HOD catalogues for non-standard cosmologies including PNG. |

No PNG EZmock, Pinocchio, GLAM-PNG or FastPM-PNG release paper was found.

### P. Posterior calibration and coverage

| Paper | arXiv | Relevance |
|---|---|---|
| Cook, Gelman & Rubin 2006, J. Comput. Graph. Stat. 15, 675 [UNVERIFIED bibliographic details] | — | Precursor of SBC. |
| Talts, Betancourt, Simpson, Vehtari & Gelman 2018 | [1804.06788](https://arxiv.org/abs/1804.06788) | SBC rank uniformity. A ∪-shaped histogram means overconfidence. |
| Modrák, Moon, Kim, Bürkner et al. 2023 (journal [UNVERIFIED]) | [2211.02383](https://arxiv.org/abs/2211.02383) | Use the joint log-likelihood ($\chi^2$, already recorded) as an SBC test quantity. |
| Säilynoja, Bürkner & Vehtari 2022 (journal [UNVERIFIED]) | [2103.10522](https://arxiv.org/abs/2103.10522) | ECDF confidence bands for small numbers of realisations. |
| Lemos, Coogan, Hezaveh & Perreault-Levasseur 2023 (venue [UNVERIFIED]) | [2302.03026](https://arxiv.org/abs/2302.03026) | TARP: a sample-based coverage test, necessary and sufficient. |
| Hermans, Delaunoy, Rozet, Wehenkel, Begy & Louppe 2022 (venue [UNVERIFIED]) | [2110.06581](https://arxiv.org/abs/2110.06581) | Approximate posteriors are commonly overconfident. |
| **Alokda, Porciani & Eggemeier 2026, A&A 714, A66** | [2605.00980](https://arxiv.org/abs/2605.00980) | **"Coverage is not enough"**: fixed-parameter frequentist tests for $f_{\rm NL}$. The Quijote ±100 ensembles are this test. |
| Hahn, Eickenberg, Ho, Hou et al. 2023, PNAS 120, e2218810120 | [2211.00723](https://arxiv.org/abs/2211.00723) | SimBIG: coverage validation in a cosmology forward-model analysis. |

### Q. Samplers for joint field + parameter inference

| Paper | arXiv | Relevance |
|---|---|---|
| Jasche, Kitaura, Wandelt & Enßlin 2010, MNRAS 406, 60 | [0911.2493](https://arxiv.org/abs/0911.2493) | ARES Gibbs sampling of field and P(k). The block template BORG inherits. |
| Taylor, Ashdown & Hobson 2008, MNRAS 389, 1284 | [0708.2989](https://arxiv.org/abs/0708.2989) | HMC for $C_\ell$. Origin of the prior/noise mass matrix in cosmology. |
| Jasche & Lavaux 2015, MNRAS 447, 1204 [volume/page UNVERIFIED] | [1402.1763](https://arxiv.org/abs/1402.1763) | Matrix-free messenger-field Gibbs sampler. |
| Jasche, Leclercq & Wandelt 2015, JCAP 01, 036 | [1409.6308](https://arxiv.org/abs/1409.6308) | First BORG data application with automatic noise calibration. |
| Kodi Ramanah, Lavaux, Jasche & Wandelt 2019, A&A 621, A69 | [1808.07496](https://arxiv.org/abs/1808.07496) | Cosmology sampled as a separate Gibbs block. |
| Hoffman & Gelman 2014, JMLR 15, 1593 [UNVERIFIED] | [1111.4246](https://arxiv.org/abs/1111.4246) | NUTS. |
| Ver Steeg & Galstyan 2021, NeurIPS [UNVERIFIED] | [2111.02434](https://arxiv.org/abs/2111.02434) | Energy-sampling Hamiltonian dynamics, the precursor of MCHMC. |
| Robnik, De Luca, Silverstein & Seljak 2023, JMLR 24 [UNVERIFIED] | [2212.08549](https://arxiv.org/abs/2212.08549) | MCHMC. Often >10× over NUTS. |
| Robnik & Seljak 2024, PMLR 253, 111 | [2303.18221](https://arxiv.org/abs/2303.18221) | MCLMC SDE. Ergodic with correct marginal. |
| **Robnik, Cohn-Gordon & Seljak 2024/2025** | [2412.08876](https://arxiv.org/abs/2412.08876) | **EEVPD bias bound** for unadjusted samplers (MCLMC target $5\times10^{-4}$). Half-step diagnostic. Unadjusted MCLMC needs 1344 vs 133,266 (NUTS) gradients on a 10⁶-dimensional lattice. |
| **Robnik, Cohn-Gordon & Seljak 2025, NeurIPS 38** | [2503.01707](https://arxiv.org/abs/2503.01707) | **MAMS**: Metropolis-adjusted microcanonical sampler, up to 7× over NUTS. The unbiased cross-check for MCLMC. |
| Sommer, Robnik, Nozadze, Seljak & Rügamer 2025, ICLR | [2502.06335](https://arxiv.org/abs/2502.06335) | Ensembles of MCLMC chains. |
| Cabezas, Corenflos, Lao, Louf et al. 2024 | [2402.10797](https://arxiv.org/abs/2402.10797) | BlackJAX. |
| Modi, Barnett & Carpenter 2024, Bayesian Anal. 19, 815 | [2110.00610](https://arxiv.org/abs/2110.00610) | Delayed-rejection HMC for multiscale targets (up to 5×). |
| **Simon, Lanusse & de Mattia 2025, JCAP 12, 039** | [2504.20130](https://arxiv.org/abs/2504.20130) | **Field-level sampler benchmark** (JaxPM). MCLMC at about 1005 evaluations/ESS, about 10× over adjusted samplers. Kaiser plus Fourier preconditioning dominate. Cost scales as about $d$ with resolution. EEVPD $10^{-6}$. Code is public. |
| Bayer, Seljak & Modi 2023 (ICML workshop [UNVERIFIED]) | [2307.09504](https://arxiv.org/abs/2307.09504) | MCLMC >10× over HMC at $2.6\times10^5$ dimensions (pmwd). $(P_L+N)/N$ preconditioning. |
| Crespi, Bonici, Loureiro et al. 2025 | [2510.26691](https://arxiv.org/abs/2510.26691) | Flinch: curved-sky field-level inference with MCLMC, orders of magnitude over HMC. |
| Chen, Chen & Dvorkin 2025, JCAP 11, 082 | [2508.13265](https://arxiv.org/abs/2508.13265) | Joint gradient sampling of initial conditions and bias with foreground-wedge modes removed. Systematics filtering inside a field likelihood. |
| Modi, Lanusse, Seljak, Spergel & Perreault-Levasseur 2021 | [2104.12864](https://arxiv.org/abs/2104.12864) | CosmicRIM learned MAP optimiser. |
| Modi, Li & Blei 2023, JCAP 03, 059 | [2206.15433](https://arxiv.org/abs/2206.15433) | Variational self-boosted sampling: a flow proposal learned from HMC. |
| Hoffman, Sountsov, Dillon et al. 2019 (venue [UNVERIFIED]) | [1903.03704](https://arxiv.org/abs/1903.03704) | NeuTra flow-warped HMC. |
| Gabrié, Rotskoff & Vanden-Eijnden 2022, PNAS 119, e2109420119 | [2105.12603](https://arxiv.org/abs/2105.12603) | Flow-augmented MCMC. |
| Legin, Ho, Lemos, Perreault-Levasseur et al. 2024, MNRAS Lett. 527, L173 | [2304.03788](https://arxiv.org/abs/2304.03788) | Score-based amortised IC posteriors. |
| Betancourt & Girolami 2015 | [1312.0906](https://arxiv.org/abs/1312.0906) | Funnels and non-centred parametrisation (the white-noise $\varepsilon$ is fully non-centred). |
| Gorinova, Moore & Hoffman 2020, ICML [UNVERIFIED] | [1906.03028](https://arxiv.org/abs/1906.03028) | Automatic partial centring, analogous to Kaiser preconditioning. |

### R. Imaging-systematics mitigation (Stage 4)

| Paper | arXiv | Relevance |
|---|---|---|
| Ross, Ho, Cuesta, Tojeiro, Percival et al. 2011, MNRAS 417, 1350 | [1105.2320](https://arxiv.org/abs/1105.2320) | Stellar-density weights. |
| Ross, Percival, Sánchez et al. 2012, MNRAS 424, 564 [UNVERIFIED] | [1203.6499](https://arxiv.org/abs/1203.6499) | BOSS systematics exceed statistical error at $s>120$. |
| Ross, Beutler, Chuang et al. 2017, MNRAS 464, 1168 [UNVERIFIED] | [1607.03145](https://arxiv.org/abs/1607.03145) | DR12 stellar-density and seeing weights. |
| Pullen & Hirata 2013, PASP 125, 705 | [1212.4500](https://arxiv.org/abs/1212.4500) | Photometric QSO cross-$z$ systematics at 9.2σ. Contamination must be <0.6% to reach $f_{\rm NL}<10$. |
| Leistedt, Peiris, Mortlock, Benoit-Lévy & Pontzen 2013, MNRAS 435, 1857 | [1306.0005](https://arxiv.org/abs/1306.0005) | Large-scale $C_\ell$ with systematics, SDSS QSOs. |
| Leistedt & Peiris 2014, MNRAS 444, 2 | [1404.6530](https://arxiv.org/abs/1404.6530) | Extended mode projection, QML. |
| Elsner, Leistedt & Peiris 2016, MNRAS 456, 2095 | [1509.08933](https://arxiv.org/abs/1509.08933) | Template subtraction is biased, basic mode projection is unbiased, and all methods add variance. |
| Elsner, Leistedt & Peiris 2017, MNRAS 465, 1847 [volume/page UNVERIFIED] | [1609.03577](https://arxiv.org/abs/1609.03577) | Pseudo-$C_\ell$ projection bias correction. |
| (Bahr-)Kalus, Percival, Bacon & Samushia 2016, MNRAS 463, 467 | [1607.02417](https://arxiv.org/abs/1607.02417) | **Best-fit mode subtraction ≡ deprojection**, so template marginalisation can be forecast. |
| Kalus, Percival, Bacon, Mueller, Samushia et al. 2019, MNRAS 482, 453 | [1806.02789](https://arxiv.org/abs/1806.02789) | Map-based mode subtraction on BOSS. A residual CMASS excess persists. |
| Weaverdyck & Huterer 2021, MNRAS 503, 5061 | [2007.14499](https://arxiv.org/abs/2007.14499) | Unified regression framework, extended to multiplicative contamination. |
| Rezaie, Seo, Ross & Bunescu 2020, MNRAS 495, 1613 | [1907.11355](https://arxiv.org/abs/1907.11355) | Neural-network imaging weights. |
| Rezaie, Ross, Seo, Mueller, Percival et al. 2021, MNRAS 506, 3439 | [2106.13724](https://arxiv.org/abs/2106.13724) | eBOSS QSO neural-network weights. The Gaia stellar template is the best. |
| Kitanidis, White, Feng, Schlegel et al. 2020, MNRAS 496, 2262 | [1911.05714](https://arxiv.org/abs/1911.05714) | Imaging systematics for DESI targets. |
| Chaussidon, Yèche, Palanque-Delabrouille et al. 2022, MNRAS 509, 3904 | [2108.03640](https://arxiv.org/abs/2108.03640) | Per-region regression for DESI QSO targets, with residual stellar excess. |
| Chaussidon et al. 2023, ApJ 944, 107 | [2208.08511](https://arxiv.org/abs/2208.08511) | DESI QSO selection with >99% purity in the automated catalogue. |
| Ross, Aguilar, Ahlen et al. 2024/2025 (JCAP [UNVERIFIED]) | [2405.16593](https://arxiv.org/abs/2405.16593) | DESI LSS catalogue construction (randoms, weights). |
| DESI Collaboration 2025, JCAP 07, 017 | [2411.12020](https://arxiv.org/abs/2411.12020) | DR1 samples and weights for the LSS catalogues. |
| Rosado-Marín, Ross, Seo, Rezaie et al. 2025, JCAP 07, 018 | [2411.12024](https://arxiv.org/abs/2411.12024) | Forward image simulations give a 3D selection function. The brief's "Rosell" is this paper. |
| Bruton, Cheshire, Doré et al. 2026 | [2605.10935](https://arxiv.org/abs/2605.10935) | Spherical Fourier-Bessel systematics discovery: residual eBOSS QSO stellar contamination. |

"Berlfein et al." on imaging systematics and $f_{\rm NL}$: [not found] in two searches.

### S. Template sampling in forward models, integral constraints and windows

| Paper | arXiv | Relevance |
|---|---|---|
| **Jasche & Lavaux 2017, A&A 606, A37** | [1706.08971](https://arxiv.org/abs/1706.08971) | **Block sampling of foreground-template coefficients** inside ARES. About 10% correlation with P(k). The Stage 4 precedent. |
| **Porqueres, Kodi Ramanah, Jasche & Lavaux 2019, A&A 624, A115** | [1812.05113](https://arxiv.org/abs/1812.05113) | **Robust likelihood** marginalising unknown per-patch contamination. The field-level analogue of deprojection. |
| Lavaux, Jasche & Leclercq 2019 | [1909.06396](https://arxiv.org/abs/1909.06396) | BORG on BOSS with known and unknown foregrounds. Unbiased at $k\simeq10^{-3}$–$10^{-2}$. |
| de Mattia & Ruhlmann-Kleider 2019, JCAP 08, 036 | [1904.08851](https://arxiv.org/abs/1904.08851) | Derives the RIC and AIC. The RIC is as large as the window effect on large scales. |
| de Mattia, Ruhlmann-Kleider, Raichoor et al. 2021, MNRAS 501, 5616 | [2007.09008](https://arxiv.org/abs/2007.09008) | First data application of the AIC and RIC (the brief's "Mattia et al. 2021"). |
| Wilson, Peacock, Taylor & de la Torre 2017, MNRAS 464, 3121 | [1511.07799](https://arxiv.org/abs/1511.07799) | Window-convolved multipoles with the integral constraint. |
| Riquelme, Avila, García-Bellido et al. 2023, MNRAS 523, 603 | [2209.07187](https://arxiv.org/abs/2209.07187) | Omitting the integral constraint gives $\Delta f_{\rm NL}\sim100$ in mocks and about 23 (1.8σ) for DES. |
| Nakano, Takada, Kurita & Terasawa 2026 | [2606.29735](https://arxiv.org/abs/2606.29735) | Window and integral-constraint modelling for PNG with light-cone mocks. |

### T. Relativistic projection, magnification and redshift errors

| Paper | arXiv | Relevance |
|---|---|---|
| Yoo, Fitzpatrick & Zaldarriaga 2009, PRD 80, 083514 | [0907.0707](https://arxiv.org/abs/0907.0707) | GR effects in galaxy clustering. |
| Yoo 2010, PRD 82, 083508 | [1009.3021](https://arxiv.org/abs/1009.3021) | GR terms can mimic a PNG detection. |
| Challinor & Lewis 2011, PRD 84, 043516 | [1105.5292](https://arxiv.org/abs/1105.5292) | Full linear number counts with magnification and evolution. |
| Bonvin & Durrer 2011, PRD 84, 063505 | [1105.5280](https://arxiv.org/abs/1105.5280) | What surveys measure. |
| Bruni, Crittenden, Koyama, Maartens, Pitrou & Wands 2012, PRD 85, 041301 | [1106.3999](https://arxiv.org/abs/1106.3999) | GR and gauge effects mimic $f_{\rm NL}\sim\mathcal{O}(1)$. |
| Jeong, Schmidt & Hirata 2012, PRD 85, 023504 | [1107.5427](https://arxiv.org/abs/1107.5427) | Residual GR acts as an effective $f_{\rm NL}<0.5$. |
| Raccanelli, Bertacca, Doré & Maartens 2014, JCAP 08, 022 | [1306.6646](https://arxiv.org/abs/1306.6646) | Wide-angle and Doppler terms are comparable to small-$f_{\rm NL}$ corrections. |
| Camera, Santos & Maartens 2015, MNRAS 448, 1035 | [1409.8286](https://arxiv.org/abs/1409.8286) | Relativistic SKA PNG forecast. |
| Wang, Beutler & Bacon 2020, MNRAS 499, 2598 | [2007.01802](https://arxiv.org/abs/2007.01802) | QSO relativistic terms comparable to $f_{\rm NL}\simeq1$ in the quadrupole at $z\gtrsim2.5$. Forward-model them from the luminosity function. |
| Castorina & Di Dio 2022, JCAP 01, 061 [UNVERIFIED] | [2106.08857](https://arxiv.org/abs/2106.08857) | GR terms are a <10% correction to the multipoles. |
| Foglieni, Pantiri, Di Dio & Castorina 2023, PRL 131, 111201 [UNVERIFIED] | [2303.03142](https://arxiv.org/abs/2303.03142) | The PNG-like GR subset is negligible for current surveys. |
| Namikawa, Okamura & Taruya 2011, PRD 83, 123514 | [1103.1118](https://arxiv.org/abs/1103.1118) | Ignoring magnification biases photometric $f_{\rm NL}$ beyond 3σ. |
| Lorenz, Alonso & Ferreira 2018, PRD 97, 023537 | [1710.02477](https://arxiv.org/abs/1710.02477) | Magnification alone can bias $f_{\rm NL}$ by about 1σ (search summary). |
| **Viljoen, Fonseca & Maartens 2021, JCAP 12, 004** | [2108.05746](https://arxiv.org/abs/2108.05746) | **Neglecting magnification biases DESI $f_{\rm NL}$ by >1σ.** Put it in the Stage 4 budget. |
| Smith, Burtin, Hou, Neveux, Ross et al. 2020, MNRAS 499, 269 | [2007.09003](https://arxiv.org/abs/2007.09003) | eBOSS QSO redshift-error mock challenge (RSD and BAO only, no $f_{\rm NL}$). |

## Conclusion

The literature shifts the project's centre of gravity. Model fidelity is a solved-in-principle problem: Stadler et al.'s numerics, Rubira & Schmidt's noise field, and Barreira's $b_{\phi\delta}$ fit each address a failure the project has already met. Two things remain genuinely open. One is whether field-level information beats P+B for $f_{\rm NL}$ at all on real tracers. Every independent group finds field-level ≈ P+B(+T) on the large scales where scale-dependent bias lives, and the one N-body $f_{\rm NL}$ test gives 1.3×. The other is how $b_\phi$/$b_{\phi\delta}$ freedom degrades that gain. The most defensible framing of a DR1 result is therefore not "beats DESI P(k)". It is "first field-level $f_{\rm NL}$ on survey data, validated by mismatch tests, compared with P+B at 7.4". The project's distinctive contributions would be the Schur-complement Fisher with 2LPT, joint gradient sampling of $(\varepsilon,f_{\rm NL},b)$, and template sampling inside the forward model, since the searches found no prior work on any of the three.

The cheapest high-value moves, in order:
1. Fix the LRG z-range in Part A.
2. Add template amplitudes and a sampled or Barreira-fit $b_{\phi\delta}$ to the Stage 1 Fisher, to see how much of the projected gain survives.
3. Move Stage 3 to AbacusPNG ($f_{\rm NL}=\pm30$) and PNG-UNIT.
4. Run SBC on self-consistent mocks before attributing the 10–45% width deficit to anything other than the noise model.
