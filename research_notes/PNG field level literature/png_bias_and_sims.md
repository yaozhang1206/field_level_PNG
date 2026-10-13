# PNG bias parameters ($b_\phi$, $b_{\phi\delta}$, the $p$ parameter), their simulation calibration, and PNG simulation suites

Scope: literature through 2026-10-13. All arXiv IDs below were checked against INSPIRE-HEP records (title, authors, date, journal) or the arXiv API during this research, unless marked **[unverified]**. Journal references come from INSPIRE. Quantitative statements come from the abstract or from the full text (downloaded PDF, grepped) as indicated.

Conventions used in these notes: $b_\phi = 2\delta_c(b_1 - p)$ with $\delta_c = 1.686$. "Universality" or UMF means $p = 1$. "Recent mergers" means $p = 1.6$. A multiplicative variant $b_\phi = 2c\,\delta_c(b_1 - p)$ with $c \approx 0.8$–$0.9$ is also common. Its equivalent effective $p$ is $p_{\rm eff} = b_1 - c(b_1 - 1)$.

---

## Q1. The universality relation and its departures for dark-matter halos (what is $p$ for halos?)

### Takeaway
For mass-selected N-body halos, $b_\phi(b_1)$ is close to universality at low bias ($b_1 \lesssim 1.5$) but falls 10–20% below it at high bias. The best simple description is $b_\phi \approx (0.8$–$0.9)\times 2\delta_c(b_1-1)$, which corresponds to an effective $p \approx 1.1$–$1.3$ for $b_1 \approx 2$–3. The exact value depends on halo finder, mass definition, redshift and estimator. Selecting on a secondary halo property, above all concentration, can move $b_\phi$ by much more: at fixed $b_1 \approx 2$, the 33% lowest- and highest-concentration terciles have $b_\phi \approx -1$ and $\approx 9$.

### Cited Findings

**Foundational derivations (universality and $p = 1.6$)**
- Dalal, Doré, Huterer & Shirokov 2008, "The imprints of primordial non-gaussianities on large-scale structure: scale dependent bias and abundance of virialized objects", arXiv:0710.4560, PRD 77, 123514. First N-body detection of the $k^{-2}$ scale-dependent halo bias. — [arXiv:0710.4560](https://arxiv.org/abs/0710.4560)
- Matarrese & Verde 2008, "The effect of primordial non-Gaussianity on halo bias", arXiv:0801.4826, ApJL 677, L77. Analytic derivation of scale-dependent halo bias. — [arXiv:0801.4826](https://arxiv.org/abs/0801.4826)
- Slosar, Hirata, Seljak, Ho & Padmanabhan 2008, "Constraints on local primordial non-Gaussianity from large scale structure", arXiv:0805.3580, JCAP 08 (2008) 031. Re-derives the effect with the peak-background split and shows it depends only on universality of the mass function. Using extended Press–Schechter, it argues that recent mergers have a suppressed effect, which is the origin of $b_\phi = 2\delta_c(b_1-1.6)$. In their data, modelling the SDSS photometric quasars as recent mergers weakens the 95% limit from $-29<f_{\rm NL}<70$ to $-31<f_{\rm NL}<70$. — [arXiv:0805.3580](https://arxiv.org/abs/0805.3580)
- Fondi et al. 2026 restate this in an EPS formula, $p = 1 - \frac{1}{\delta_c}\left[\frac{\omega_f e^{-\omega_f^2/2}}{\sqrt{2\pi}\,{\rm erfc}(\omega_f/\sqrt2)} - 1\right]$ with $\omega_f \propto z_f - z_o$. Halos observed exactly at formation ($\omega_f = 0$) give $p = 1.6$, which is a hard upper limit in this model. With no secondary dependence the formula returns universality, $p = 1$. — [arXiv:2602.12357 (full text, Eq. 2.4)](https://arxiv.org/abs/2602.12357)

**Early N-body calibrations (halo-finder and ellipsoidal-collapse corrections)**
- Grossi, Verde, Carbone, Dolag et al. 2009, "Large-scale non-Gaussian mass function and halo bias: tests on N-body simulations", arXiv:0902.2013, MNRAS 398, 321. Proposes $\delta_c \to q\,\delta_c$ in the non-Gaussian halo bias, with $q \approx 0.75$ (from the ellipsoidal-collapse correction to the mass function), for FoF halos. — [arXiv:0902.2013](https://arxiv.org/abs/0902.2013)
- Pillepich, Porciani & Hahn 2010, "Universal halo mass function and scale-dependent bias from N-body simulations with non-Gaussian initial conditions", arXiv:0811.4176, MNRAS 402, 191. FoF mass function is universal with PNG. PBS parametrizations "do not provide an accurate description" of the scale-dependent bias, so they give a fitting formula instead. — [arXiv:0811.4176](https://arxiv.org/abs/0811.4176)
- Desjacques, Seljak & Iliev 2009, "Scale-dependent bias induced by local non-Gaussianity: a comparison to N-body simulations", arXiv:0811.2748, MNRAS 396, 85. Theory and simulations agree at $k<0.01\,h$/Mpc for $b>1.5$. At $k = 0.1$ the effect is suppressed by up to a factor of 3 and they provide a fitting formula. For $b<1.5$ the measured PNG bias exceeds theory. — [arXiv:0811.2748](https://arxiv.org/abs/0811.2748)
- Reid, Verde, Dolag, Matarrese & Moscardini 2010, "Non-Gaussian halo assembly bias", arXiv:1004.1637, JCAP 07 (2010) 013. First N-body detection that $b_\phi$ depends on formation history, by up to 100%: suppressed for recently formed halos and enhanced for old ones. SAMs imply +23% (stellar-mass) and +48% (SFR-selected) enhancement at $z=1$. — [arXiv:1004.1637](https://arxiv.org/abs/1004.1637)
- Wagner & Verde 2012, "N-body simulations with generic non-Gaussian initial conditions II: Halo bias", arXiv:1102.3229, JCAP 03 (2012) 002. Analytic halo-bias predictions work once a calibration factor is fitted to simulations. Also introduces the improved generic-PNG IC scheme. — [arXiv:1102.3229](https://arxiv.org/abs/1102.3229)
- Scoccimarro, Hui, Manera & Chan 2012, "Large-scale bias and efficient generation of initial conditions for non-local primordial non-Gaussianity", arXiv:1108.5512, PRD 85, 083002. Gives a PBS bias formula with non-Markovian and non-universal corrections that fits simulations better. This is also the 2LPTPNG IC paper. — [arXiv:1108.5512](https://arxiv.org/abs/1108.5512)

**Separate-universe and response measurements**
- Baldauf, Seljak, Senatore & Zaldarriaga 2016, "Linear response to long wavelength fluctuations using curvature simulations", arXiv:1511.01465, JCAP 09 (2016) 007. The slope of halo-count response to variance versus density differs from the universal-mass-function value by about 8–20%. — [arXiv:1511.01465](https://arxiv.org/abs/1511.01465)
- Biagetti, Lazeyras, Baldauf, Desjacques & Schmidt 2017, "Verifying the consistency relation for the scale-dependent bias from local primordial non-Gaussianity", arXiv:1611.04901, MNRAS 468, 3277. The PNG-simulation $b_\phi$ equals the mass-function response to $A_s$ for three halo finders (SO and FoF). For all finders, high-mass halos have $b_\phi$ 10–15% below UMF. For SO halos the deviation changes sign at low mass. — [arXiv:1611.04901](https://arxiv.org/abs/1611.04901)
- Barreira 2022 (IllustrisTNG separate universe, total-mass halos): for $M_t$-selected objects, $b_\phi$ lies below universality for $b_1 \gtrsim 1.5$, and $b_\phi = 0.85\times2\delta_c(b_1-1)$ is "a more adequate description" of gravity-only results. — [arXiv:2107.06887 (full text, Sec. 3)](https://arxiv.org/abs/2107.06887)
- Hadzhiyska et al. 2024 (AbacusPNG): "the universality relation provides a good approximation for $b_1 \lesssim 1.5$, but overpredicts the numerical results for larger biases". They quote the common modification $b_\phi = 2c\,\delta_c(b_1-p)$ with $c\approx0.9$. Separate-universe halos agree with the modified relation. — [arXiv:2402.10881 (full text, Eqs. 16–17)](https://arxiv.org/abs/2402.10881)
- Sullivan & Chen 2024 (field-level Lagrangian bias, FastPM): $b_\phi = 0.8\times2\delta_c b_\delta$ fits halos better than UMF. Field-level, separate-universe and UMF agree at high bias but differ by several sigma at low bias. In Quijote-PNG the field-level vs SU/UMF disagreement is "much larger" than in FastPM. They suspect low mass resolution. — [arXiv:2410.18039 (full text, Sec. 3.1 and App. A)](https://arxiv.org/abs/2410.18039)

**Mass-dependence of $p$ in large PNG simulations**
- Gutiérrez Adame, Avila, Gonzalez-Perez, Yepes et al. 2024, "PNG-UNITsims: Halo clustering response to primordial non-Gaussianities as a function of mass", arXiv:2312.12405, A&A 689, A69. Uses $4096^3$ particles in 1 Gpc/h with $f_{\rm NL}=100$, fixed-and-matched ICs, Rockstar halos at $z\approx1$. Halos of $10^{12}$–$5\times10^{13}\,h^{-1}M_\odot$ follow $p=1$. Halos of $2\times10^{10}$–$10^{12}$ have $p<1$ at 1.5–3.1σ. Combined fit: $p = 0.955\pm0.013$, 3σ from universality ($p=0.950\pm0.025$ excluding extrapolated bins). The LR-UNIT gives $0.972\pm0.017$. Fixing $p=1$ when it is wrong can bias $f_{\rm NL}$ by up to 2σ in DESI-like forecasts. — [arXiv:2312.12405](https://arxiv.org/abs/2312.12405) (numbers also from full text Sec. 6)

**Assembly bias in $b_\phi$ for halos**
- Lazeyras, Barreira, Schmidt & Desjacques 2023, "Assembly bias in the local PNG halo bias and its implication for $f_{\rm NL}$ constraints", arXiv:2209.07251, JCAP 01 (2023) 023. Spin and sphericity move $b_1$ and $b_\phi$ together and roughly preserve $b_\phi(b_1)$. Concentration drives them in opposite directions. At $z=0.5$ and $b_1\approx2$: all halos $b_\phi\approx3$, lowest-33% concentration $b_\phi\approx-1$, highest-33% $b_\phi\approx9$. Applied to BOSS DR12 these give $\sigma(f_{\rm NL}) = 44$, 165 and 19. — [arXiv:2209.07251](https://arxiv.org/abs/2209.07251)
- Lucie-Smith, Barreira & Schmidt 2023, "Halo assembly bias from a deep learning model of halo formation", arXiv:2304.09880, MNRAS 524, 1746. A CNN surrogate trained on ICs to predict mass and concentration recovers $b_1$, $b_2$, $b_\phi$. At fixed mass, concentration mildly affects $b_1$ but strongly affects $b_\phi$. — [arXiv:2304.09880](https://arxiv.org/abs/2304.09880)
- Hadzhiyska, Garrison, Eisenstein & Ferraro 2024, "A modest set of simulations of local-type primordial non-Gaussianity in the DESI era" (AbacusPNG), arXiv:2402.10881, PRD 109, 103530. Finds a strong $b_\phi$ response to concentration, shear/environment and accretion rate. Allowed galaxy-halo models give $\sigma(f_{\rm NL})$ = 15, 8, 7 for LRG $z=0.5$, LRG $z=0.8$ and QSO $z=1.4$. — [arXiv:2402.10881](https://arxiv.org/abs/2402.10881)
- Shiveshwarkar, LoVerde, Hirata & Jamieson 2025, "Where does nonuniversality in assembly bias come from?", arXiv:2508.11798, PRD 113, 103523. A separate-universe explanation: universality holds for mass-selected halos because abundance depends on $\sigma_8$, whose responses to $\delta_L$ and to $\Phi$ match. It fails for properties whose responses differ, such as concentration. Concentration-selected halos become universal at the highest masses. They argue that non-universality in high-$z$ ($z\gtrsim3$), high-bias tracers is likely a selection-function artifact. — [arXiv:2508.11798](https://arxiv.org/abs/2508.11798)

**Field-level halo inference context (directly relevant to Stage 3)**
- Andrews, Jasche, Lavaux, Coulton et al. 2026, "Field-Level Inference of Primordial Non-Gaussianity with the Quijote Simulation Suite", arXiv:2603.20855 (MNRAS submitted). The setup:
  - Data: Quijote-PNG FoF ($b=0.2$) halos with $M > 10^{13.5}\,h^{-1}M_\odot$ at $z=0.5$, first 30 boxes, $f_{\rm NL} = 0, \pm100$, mass-weighted CIC, redshift space.
  - Bias: $b_\phi = 2\delta_c(b_1-p)$ with "$p$ is fixed to 1.3", and $b_{\phi\delta}$ from the UMF formula, both fixed.
  - Result: ~1.3× improvement over P+B.

  Appendix A explains the choice of $p$. With $p=1.0$, inferred $b_1$ differed systematically between the $f_{\rm NL}=-100, 0, +100$ runs on identical ICs. Following Shiveshwarkar et al. 2025 (who modify $\delta_c$), they raised $p$ from 1.0 to 1.3, which "distinctly alleviated" the offset. So **$p=1.3$ is an empirical tuning to the BORG forward model, not a measured $b_\phi$**. — [arXiv:2603.20855 (full text, Sec. 2, 3 and App. A)](https://arxiv.org/abs/2603.20855)

### Inferences
- Converting the halo relations to an effective $p$:
  - $0.85\times2\delta_c(b_1-1)$ gives $p_{\rm eff} = 0.15\,b_1 + 0.85$, so 1.15 at $b_1=2$, 1.23 at $b_1=2.5$ and 1.30 at $b_1=3$.
  - $0.8\times$ UMF gives $p_{\rm eff} = 0.2\,b_1 + 0.8$, so 1.20 at $b_1 = 2$ and 1.30 at $b_1 = 2.5$.
  - Lazeyras et al.'s "all halos" point ($b_1\approx2$, $b_\phi\approx3$) gives $p\approx1.11$.

  Andrews' $p=1.3$ for FoF $M>3\times10^{13}$ halos at $z=0.5$ is therefore at the upper edge of what halo calibrations predict. It may absorb other forward-model mismatches (Eulerian bias, mass-weighted assignment, fixed $b_{\phi\delta}$). This is my inference, not a statement in the paper.
- A Stage-3 result of $p\approx1.0$ from the $f_{\rm NL}=\pm100$ response does not contradict the literature, for three reasons:
  - Halo-finder and mass-definition effects are known to change the sign and size of the deviation (Biagetti et al. 2017: SO vs FoF).
  - The Rockstar-based samples (memory: Quijote PNG Rockstar, $M \ge 6\times10^{13}$ completeness) differ from Andrews' FoF selection.
  - A response measurement (the separate-universe-like $\pm f_{\rm NL}$ difference) measures $b_\phi$ itself. A BORG fit with fixed $b_{\phi\delta}$ measures an effective amplitude.

  The 10–20% suppression at high mass corresponds to $\Delta p\approx0.1$–0.3. With realistic Stage-3 error bars this may be hard to distinguish from $p=1$.
- Mass-weighted CIC (Andrews) produces a field whose effective $b_1$ and $b_\phi$ are mass-weighted averages over a mass function. Because $b_\phi(b_1)$ is not linear in $b_1$ through the 0.85 factor and assembly-bias terms, the effective $p$ of a mass-weighted field need not equal that of the number-weighted sample. This is an inference worth checking in Stage 3.

### Gaps
- No paper found that tabulates $p$ for Quijote-PNG **Rockstar** vs **FoF** halos at matched mass thresholds. Sullivan & Chen 2024 and Coulton et al. 2023b use FoF only.
- The Shiveshwarkar et al. 2025 "modified $\delta_c$" prescription that Andrews followed was not read in full here. The exact $\delta_c$ modification and its numerical value were not extracted.

---

## Q2. Separate-universe and alternative calibrations of $b_\phi$ (methods)

### Takeaway
$b_\phi$ equals the response of tracer abundance to the primordial amplitude, $b_\phi = \partial\ln n/\partial\ln\sigma_8$ (equivalently $\partial/\partial\delta A_s$). It can therefore be calibrated four ways:
- separate-universe runs with modified $A_s$ (gravity-only, hydro or SAM);
- direct PNG simulations with $\pm f_{\rm NL}$;
- field-level fits at fixed ICs;
- newer data-driven estimators that use the time evolution of tracer counts, which need no simulations of the specific galaxy population.

Separate-universe and PNG-simulation estimates agree for halos (Biagetti 2017, AbacusPNG, PNG-UNIT). Field-level estimates agree qualitatively but differ in detail.

### Cited Findings
- Biagetti et al. 2017 verified the consistency relation: $b_\phi$ from PNG simulations equals the mass-function response to $A_s$. — [arXiv:1611.04901](https://arxiv.org/abs/1611.04901)
- Barreira, Cabass, Schmidt, Pillepich & Nelson 2020, "Galaxy bias and primordial non-Gaussianity: insights from galaxy formation simulations with IllustrisTNG", arXiv:2006.09368, JCAP 12 (2020) 013. TNG separate-universe runs, with modified $\Omega_m$ for $b_1$ and modified $A_s$ for $b_\phi$. Stellar-mass-selected galaxies follow $b_\phi = 2\delta_c(b_1-p)$ with $p\in[0.4,0.7]$, best $p\in[0.5,0.6]$. This is the origin of the $p = 0.55$ default in the Euclid FLI paper. The cause is the different response of the stellar-to-halo-mass relation to $\delta$ and $\Phi$. Colour and $\dot M_{\rm BH}$ selections give non-trivial $b_\phi(b_1)$. UMF underpredicts $b_\phi(M_*)$, so analyses may underestimate the true constraining power. — [arXiv:2006.09368](https://arxiv.org/abs/2006.09368)
- Gutiérrez Adame et al. 2024: PNG-UNIT $b_\phi$ (from $f_{\rm NL}=100$ scale-dependent bias) "outperform[s]" separate-universe constraints for their setup. — [arXiv:2312.12405](https://arxiv.org/abs/2312.12405)
- Hadzhiyska et al. 2024: AbacusPNG $b_\phi$ agrees with separate-universe and one-loop PT. — [arXiv:2402.10881](https://arxiv.org/abs/2402.10881)
- Sullivan & Chen 2024, "Local primordial non-Gaussian bias at the field level", arXiv:2410.18039, JCAP 03 (2025) 016. First field-level measurement of $b_\phi$ and $b_{\delta\phi}$, using Lagrangian bias with N-body displacements. Results:
  - Qualitative agreement with UMF and SU, disagreement in detail.
  - Marginalizing over $b_\phi f_{\rm NL}$ even at the field level cannot precisely constrain $f_{\rm NL}$: a 2–3× degradation from linear to quadratic bias models on perturbative mocks.
  - On halos, pure-$f_{\rm NL}$ constraints "appear biased" when $b_\phi$ and $b_{\delta\phi}$ are marginalized. A $b_\phi$ prior is necessary.
  — [arXiv:2410.18039](https://arxiv.org/abs/2410.18039)
- Sullivan & Seljak 2025, "Local primordial non-Gaussian bias from time evolution", arXiv:2503.21736, PRD 112, 083522. $b_\phi$ is closely tied to the time evolution of tracer number density, both for universal tracers and for non-universal tracers in the SU picture. Validated in simulations, with a demonstration on simulated galaxies under a selection function. — [arXiv:2503.21736](https://arxiv.org/abs/2503.21736)
- Dalal & Percival 2025, "Estimating non-gaussian bias using counts of tracers", arXiv:2503.21024, JCAP 04 (2026) 037. $b_\Phi$ is estimated from $d\ln\bar n/dz$. It works "reasonably well" for halos across survey redshifts and masses, including concentration assembly bias. The worst discrepancies, about 40%, occur near $z\sim0$. Selection-function redshift dependence is mitigated by k-corrected cuts. — [arXiv:2503.21024](https://arxiv.org/abs/2503.21024) (40% from full text Sec. 2)
- Nascimento & Dalal 2026, "Breaking the $f_{\rm NL}$–$b_\phi$ degeneracy with the time evolution of tracer number counts", arXiv:2607.18229 (preprint). Tests the time-evolution estimator against SU in CAMELS TNG and SIMBA. It extends to observer-frame photometric selections with SED-derived corrections, including passive fading for magnitude cuts. — [arXiv:2607.18229](https://arxiv.org/abs/2607.18229)
- Hadzhiyska & Ferraro 2025, "Refining local-type primordial non-Gaussianity: Sharpened $b_\phi$ constraints through bias expansion", arXiv:2501.14873, PRD 111, 103521. Uses HEFT field-level Lagrangian biases ($b_1, b_2, b_s, b_\nabla$) to predict $b_\phi$ by linear regression on AbacusSummit DESI-like LRG and QSO samples. This reduces $b_\phi$ uncertainty by >70% for fixed-mass halos, mostly via $b_\nabla$ as a concentration proxy, and by 80% for galaxy samples. — [arXiv:2501.14873](https://arxiv.org/abs/2501.14873)

### Inferences
- For a field-level forward model like ours, the Hadzhiyska & Ferraro result suggests a practical route: let the higher-derivative or Lagrangian bias that the forward model already fits ($b_\nabla$-like terms) inform the $b_\phi$ prior. This is an inference. Their regression is on Lagrangian HEFT biases, not on our Eulerian-at-Lagrangian-position scheme.
- Sullivan & Chen's finding that field-level inference does not escape the $f_{\rm NL}b_\phi$ degeneracy supports the CLAUDE.md plan to run both "$b_\phi$ fixed" and "$b_\phi$ with a 40% prior". The forecast with $b_\phi$ free should be expected to degrade by order 2–3×.

### Gaps
- No public separate-universe $b_\phi$ measurement was found for Quijote-PNG-resolution halos using the Quijote $\pm A_s$ derivative boxes. Such an estimate would give an independent $p$ for our Stage 3 sample.

---

## Q3. Galaxy-level calibration and priors on $b_\phi$ (hydro, SAM, HOD, ML)

### Takeaway
For real galaxies, $p$ is selection-dependent:
- stellar-mass selection gives $p\approx0.4$–0.7 (TNG);
- colour and $\dot M_{\rm BH}$ selections give non-linear, redshift-dependent relations;
- Hα/emission-line selection shows little assembly bias in Galacticus.

Several groups now produce informative priors by marginalizing over galaxy-formation physics (CAMELS-SAM, CAMELS hydro), HOD posteriors, or bias-expansion proxies. Prior widths in the literature range from $\Delta p\approx0.5$ (Barreira 2020, "sufficient" for unbiased $f_{\rm NL}$) to $\sigma_{b_\phi}\approx0.02$–0.08 (Moore et al. 2026, ELG-like, conditioned on SMF/SHMR).

### Cited Findings
- Barreira 2020, "On the impact of galaxy bias uncertainties on primordial non-Gaussianity constraints", arXiv:2009.06622, JCAP 12 (2020) 031. Findings from an idealized multitracer P+B forecast:
  - The $f_{\rm NL}$–$p$ posterior is markedly bimodal.
  - Assuming $p=1$ when the truth is $p=0.55$ biases $f_{\rm NL}$ by 0.5–1σ.
  - A prior $\Delta p\approx0.5$ around the truth is enough to be unbiased.
  - Fitting $f_{\rm NL}b_\phi$ directly detects $f_{\rm NL}\neq0$ with equal significance but loses some bispectrum benefit.
  — [arXiv:2009.06622](https://arxiv.org/abs/2009.06622)
- Barreira 2022a, "Predictions for local PNG bias in the galaxy power spectrum and bispectrum and the consequences for $f_{\rm NL}$ constraints", arXiv:2107.06887, JCAP 01 (2022) 033. TNG separate universe. $b_\phi(b_1)$ and $b_{\phi\delta}(b_1)$ for galaxies are redshift-dependent and very sensitive to selection (total, stellar and BH mass, $\dot M_{\rm BH}$, colour). For $V=100\,{\rm Gpc}^3h^{-3}$ at $z=1$, $\Delta b_\phi\lesssim1$ and $\Delta b_{\phi\delta}\lesssim5$ give relatively unbiased $f_{\rm NL}$ from P+B. — [arXiv:2107.06887](https://arxiv.org/abs/2107.06887)
- Barreira 2022b, "Can we actually constrain $f_{\rm NL}$ using the scale-dependent bias effect? An illustration of the impact of galaxy bias uncertainties using the BOSS DR12 galaxy power spectrum", arXiv:2205.05673, JCAP 11 (2022) 013. On the same BOSS data, $\sigma(f_{\rm NL})$ varies by an order of magnitude with the $b_\phi(b_1)$ choice, from $f_{\rm NL}=16\pm16$ to $230\pm226$. Marginalizing over $b_\phi$ with wide priors is not conservative and biases results through projection effects. $f_{\rm NL}b_\phi\neq0$ is detected at 1.6σ. — [arXiv:2205.05673](https://arxiv.org/abs/2205.05673)
- Barreira & Krause 2023, "Towards optimal and robust $f_{\rm NL}$ constraints with multi-tracer analyses", arXiv:2302.09066, JCAP 10 (2023) 044. Multi-tracer power scales as $|b_1^Bb_\phi^A - b_1^Ab_\phi^B|$. In TNG, a $g-r$ colour split more than doubles the significance of $f_{\rm NL}b_\phi\neq0$. Multi-tracer is more robust to $b_\phi$ misspecification. — [arXiv:2302.09066](https://arxiv.org/abs/2302.09066)
- Marinucci, Desjacques & Benson 2023, "Non-Gaussian assembly bias from a semi-analytic galaxy formation model", arXiv:2303.10337, MNRAS 524, 325. Galacticus at $z=1$:
  - $g-r$ colour selection shows large non-Gaussian assembly bias.
  - Hα-selected galaxies show no significant assembly bias, so it is less of a concern for ELG surveys.
  - SAM parameter variations change $\Delta b_\phi$ by up to order unity, but the trends persist.
  — [arXiv:2303.10337](https://arxiv.org/abs/2303.10337)
- Sullivan, Prijon & Seljak 2023, "Learning to concentrate: multi-tracer forecasts on local primordial non-Gaussianity with machine-learned bias", arXiv:2303.08901, JCAP 08 (2023) 004. ML on TNG observables predicts $b_\phi$ for DESI-like samples. Forecasts: $\sigma(f_{\rm NL})$ = 2.3 for ELGs, 3.7 for LRGs, and 1.5 in the overlap multi-tracer case, with <1 for an ideal $b_\phi$ split. — [arXiv:2303.08901](https://arxiv.org/abs/2303.08901)
- Fondi, Verde, Villaescusa-Navarro, Baldi et al. 2024, "Taming assembly bias for primordial non-Gaussianity", arXiv:2311.10088, JCAP 02 (2024) 048. In hydro simulations, a photometric proxy for the halo properties that drive assembly bias can be built. A prior guided by it degrades $\sigma(f_{\rm NL})$ only mildly compared with perfect knowledge, with controlled systematic error. It is the methodological basis for Fondi 2026. — [arXiv:2311.10088](https://arxiv.org/abs/2311.10088)
- Perez, Genel, Krause & Somerville 2026, "The Impact of Galaxy Formation on Galaxy Biasing, and Implications for Primordial non-Gaussianity Constraints", arXiv:2602.04987 (preprint). CAMELS-SAM: separate-universe N-body ($L=205\,h^{-1}$cMpc, $1280^3$) and 55 Santa-Cruz SAM variants spanning stellar and AGN feedback. These span the $b_\phi$ behaviour previously seen in TNG, the Munich SAM and Galacticus. sSFR selections are particularly robust to modelling. The paper includes a literature table of the $p$ values assumed by past observational analyses. — [arXiv:2602.04987](https://arxiv.org/abs/2602.04987)
- Moore, Perez & Krause 2026, "Informative Priors on Primordial Non-Gaussianity Bias $b_\phi$ From Galaxy Formation", arXiv:2604.21790 (preprint). Uses CAMELS-SAM plus separate universe for a DESI-ELG-like selection. Conditioning on the SMF or SHMR cuts $\sigma_{b_\phi}$ from 0.69 to 0.08 or 0.02, reductions of 88% and 97%. — [arXiv:2604.21790](https://arxiv.org/abs/2604.21790)
- Yu & Nguyen 2026, "How I stop worrying about non-universality and $b_\phi$: Constraining local $f_{\rm NL}$ with $b_\phi$ priors from HOD posteriors", arXiv:2607.01314 (preprint). Samples an HOD posterior fit to DESI EDR, measures $b_\phi$ in the resulting mocks, and builds a prior. It recovers unbiased $f_{\rm NL}$ on mocks with different $f_{\rm NL}$, even with assembly bias. — [arXiv:2607.01314](https://arxiv.org/abs/2607.01314)
- Merino, Avila, Gutiérrez Adame et al. 2026, "Unclustered tracers remain unclustered – The lack of primordial non-Gaussianity response of bias-zero tracers", arXiv:2602.04698, A&A 707, L15. In PNG-UNITsims ($f_{\rm NL}=100$ and $-20$), $b_1=0$ tracers built from local-density selections have $b_\phi\approx0$. High-mass, QSO and LRG HOD samples follow roughly universality for $b_1>1$, but $b_\phi(b_1)$ flattens to 0 for $|b_1|<1$. — [arXiv:2602.04698](https://arxiv.org/abs/2602.04698)
- Voivodic & Barreira 2021, "Responses of Halo Occupation Distributions: a new ingredient in the halo model & the impact on galaxy bias", arXiv:2012.04637, JCAP 05 (2021) 069. HOD parameters respond to long modes, which changes galaxy bias predictions. Relevant to HOD-based $b_\phi$ priors. — [arXiv:2012.04637](https://arxiv.org/abs/2012.04637)

### Inferences
- The 40% Gaussian prior on $b_\phi$ in CLAUDE.md and Andrews et al. is broader than recent physics-informed priors for specific selections (Moore 2026: ≤12% for ELGs). It is comparable to the spread across selection criteria in TNG. For QSOs, the Fondi 2026 $p$ distribution (Q4) translates at $b_1\approx2.4$ into $b_\phi\approx2\delta_c(1.0$–$1.4)$, about 3.4–4.7. A $\pm$40% prior is therefore roughly consistent with the full 16th–84th percentile spread in $p$, and slightly conservative. This is my arithmetic, not from a paper.
- Barreira 2022b warns that wide $b_\phi$ priors create projection effects. In an MCMC field-level analysis (Stage 2+), a 40% prior may shift posterior means, so Stage 2 mock tests should check $f_{\rm NL}$ bias as well as $\sigma$.

### Gaps
- No LRG-specific hydro/SAM $p$ prior paper comparable to Fondi 2026 for QSOs was found. DESI LRG analyses still use $p=1$, with $1.0\pm0.1$ in Brown et al. 2026 (Q4).

---

## Q4. Quasar-specific $p$ calibration and how DESI/eBOSS analyses treat $p$

### Takeaway
$p = 1.6$ for quasars is a hypothesis from Slosar et al. 2008 (QSOs as recent mergers), not a calibration. TNG separate-universe results show that $\dot M_{\rm BH}$-selected objects are not described by $p=1.6$. The first dedicated DESI-QSO calibration (Fondi et al. 2026, TNG300 merger trees with the EPS $p$ formula) gives a skewed distribution with mean $\bar p\simeq1.4$ at $z\simeq1.5$, weak $z$-dependence, and an EPS hard upper bound of 1.6. Applied to DESI DR1 QSOs this gives $f_{\rm NL}=-3.3\pm9.2$.

### Cited Findings
- Chaussidon et al. 2024, "Constraining primordial non-Gaussianity with DESI 2024 LRG and QSO samples", arXiv:2411.17623, JCAP 06 (2025) 029. Uses $p=1.6$ for QSOs, following Slosar 2008, and notes that "we assume that all the quasars have a recent merger history which it is not known". Uses $p=1.0$ for LRGs. Combined $f_{\rm NL}=-3.6^{+9.0}_{-9.1}$. Universality for QSOs gives $3.5^{+10.7}_{-7.4}$. — [arXiv:2411.17623](https://arxiv.org/abs/2411.17623) (full text Sec. on $b_\Phi$)
  - Note: the INSPIRE abstract quotes 1,631,716 LRGs at $0.6<z<1.1$, while the full text (and CLAUDE.md) uses 2,130,621 LRGs at $0.4<z<1.1$. The abstract appears to describe a different selection or version, so check before using sample sizes.
- Barreira 2022a (TNG): "at least in the IllustrisTNG model, the relation $b_\phi(b_1)=2\delta_c(b_1-1.6)$ is not a good description of the objects selected by $\dot M_{\rm BH}$". The $\dot M_{\rm BH}$-selected $b_\phi(b_1)$ is especially steep and has no regular, redshift-independent form. — [arXiv:2107.06887 (full text)](https://arxiv.org/abs/2107.06887)
- Fondi, Verde, Chaussidon, Aguilar et al. 2026, "Assembly bias and local Primordial non-Gaussianity from DESI DR1 quasars", arXiv:2602.12357, JCAP 06 (2026) 062. Method and results:
  - Selection: TNG300-1 at $z_o=1.5$, $\log M>12.2$, BHs in quasar mode with Eddington ratio $\chi>0.1$, 258 objects to match the DESI QSO density. Mean host $\log\bar M = 12.65$.
  - $p$ comes from the formation redshift ($M=0.5M_o$) through the EPS formula. QSO hosts have $\bar p=1.41$. A mass-matched control sample with random histories has $\bar p=0.99$.
  - Robust to replacing ~34% of objects. CAMELS variations change it only mildly ($\bar p\to1.38$). There is weak $z$-dependence from $z=1$ to 3; the paper's Fig. 6 shows 16th–84th percentile bars of roughly ±0.1–0.2 in $p$ (read from the figure axis, approximate).
  - Fixing $p=\langle p\rangle$ or marginalizing over $\Pi(p)$ gives indistinguishable DR1 constraints.
  - DR1: $f_{\rm NL}=-3.3\pm9.2$. They recommend OQE weights with $p_w=1.4$ in future DESI pipelines.
  — [arXiv:2602.12357](https://arxiv.org/abs/2602.12357) (abstract and full text Secs. 2–5, App.)
- Brown, Levi, Randall, Chaussidon et al. 2026, "Measuring local primordial non-Gaussianity from the clustering of DESI DR1 LRGs and QSOs", arXiv:2606.24651 (preprint). Configuration-space 2pcf with simulation-based modelling. Gaussian priors $p_{\rm LRG}=1.0\pm0.1$ and $p_{\rm QSO}=1.6\pm0.1$. Results: LRG $-3^{+22}_{-21}$, QSO $0^{+17}_{-16}$, joint $-3\pm12$. — [arXiv:2606.24651](https://arxiv.org/abs/2606.24651)
- Chiarenza, Krolewski, Bonici, Chaussidon et al. 2025, "Constraining primordial non-Gaussianity from DESI DR1 quasars and Planck PR4 CMB lensing", arXiv:2512.17865, JCAP 05 (2026) 098. $f_{\rm NL}=2^{+28}_{-34}$ for $p=1.6$ and $6^{+20}_{-24}$ for $p=1.0$, showing the $p$-dependence of $\sigma$. — [arXiv:2512.17865](https://arxiv.org/abs/2512.17865)
- Rosado-Marín, Ross, Seo, Chaussidon et al. 2026, "Local primordial non-Gaussianity using cross-correlations of DESI tracers", arXiv:2604.05213 (preprint). LRG×QSO cross-correlation improves DR1 by ~9%, giving $f_{\rm NL}=2.1^{+8.8}_{-8.3}$. ELGs add nothing. — [arXiv:2604.05213](https://arxiv.org/abs/2604.05213)
- Chudaykin, Ivanov & Philcox 2025, "Reanalyzing DESI DR1. III. Constraints on inflation from galaxy power spectra and bispectra", arXiv:2512.04266, PRD 113, 063552. EFT P+B gives $f_{\rm NL}^{\rm loc}=-0.1\pm7.4$, and $-0.0\pm4.1$ combined with CMB. — [arXiv:2512.04266](https://arxiv.org/abs/2512.04266)
- Earlier QSO analyses report both $p$ values:
  - Castorina et al. 2019 (eBOSS DR14 QSO), arXiv:1904.08859, JCAP 09 (2019) 010. — [arXiv:1904.08859](https://arxiv.org/abs/1904.08859)
  - Mueller et al. 2022 (eBOSS DR16 QSO), arXiv:2106.13725. — [arXiv:2106.13725](https://arxiv.org/abs/2106.13725)
  - Cagliari et al. 2024 (eBOSS DR16 QSO, optimal), arXiv:2309.15814, JCAP 08 (2024) 036. — [arXiv:2309.15814](https://arxiv.org/abs/2309.15814)
  - Krolewski et al. 2024 (DESI QSO targets × Planck lensing), arXiv:2305.07650, JCAP 03 (2024) 021. — [arXiv:2305.07650](https://arxiv.org/abs/2305.07650)
- Perez et al. 2026 tabulates published constraints with their $p$ choices. Examples: Castorina 2019 $-81<f_{\rm NL}<26$ ($p=1.6$, 95%) vs $-51<f_{\rm NL}<21$ ($p=1$). Mueller 2022 $-23<f_{\rm NL}<21$ ($p=1.6$). Krolewski 2024 $-26^{+45}_{-40}$ ($p=1.6$) vs $-18^{+29}_{-27}$ ($p=1$). Cabass et al. 2022 $-33\pm28$ ($p=0.55$). — [arXiv:2602.04987 (Table 1)](https://arxiv.org/abs/2602.04987)
- Hadzhiyska et al. 2024 fit $b_\phi$ to early DESI QSO/LRG galaxy-halo models and found $\sigma(f_{\rm NL})\approx7$ for QSOs at $z=1.4$ from $b_\phi$ uncertainty alone. — [arXiv:2402.10881](https://arxiv.org/abs/2402.10881)
- Shiveshwarkar et al. 2025 argue that non-universality in very high-$z$ ($z\gtrsim3$), high-bias tracers is likely a selection artifact. — [arXiv:2508.11798](https://arxiv.org/abs/2508.11798)

### Inferences
- For our DR1 QSO forecasts and Stage 2–4 mocks, $p=1.4$ (Fondi 2026) is the best-motivated single value. A prior roughly uniform or skewed on $[1.0,1.6]$ captures the TNG distribution. The CLAUDE.md runs at $p=1.6$ and $p=1.0$ bracket it.
- Because $b_\phi\propto(b_1-p)$, moving from $p=1.6$ to 1.4 at $b_1=2.44$ (FKP) raises $b_\phi$ by $(2.44-1.4)/(2.44-1.6)\approx1.24$. That would shrink $\sigma(f_{\rm NL})$ by about 20% at fixed $\sigma(b_\phi f_{\rm NL})$. This is my arithmetic.

### Gaps
- No QSO $p$ calibration from a semi-analytic model or a second hydro code (e.g. SIMBA, ASTRID) was found. Fondi 2026 uses TNG300 with CAMELS robustness tests only.
- The exact 16th–84th percentile values of Fondi's $\Pi(p)$ were read approximately from Fig. 6. A tabulated version was not found in the text.

---

## Q5. $b_{\phi\delta}$ and second-order PNG bias

### Takeaway
The multivariate bias expansion in $(\delta,\phi)$ goes back to Giannantonio & Porciani 2010, was made complete under renormalization by Assassi, Baumann & Schmidt 2015, and is reviewed in Desjacques, Jeong & Schmidt 2018. It introduces $b_{\phi\delta}$, which enters the galaxy bispectrum at leading order and the field-level model. The standard UMF expression (used by Euclid FLI and Andrews 2026) is $b_{\phi\delta} = b_\phi - b_1 + 1 + \delta_c[b_2 - \tfrac{8}{21}(b_1-1)]$. The first direct simulation measurement (Barreira 2022a) shows UMF **overpredicts** halo $b_{\phi\delta}(b_1)$ by up to $\Delta b_{\phi\delta}\sim3$ for $1\lesssim b_1\lesssim3$, and gives a quadratic fit. The first field-level measurement is Sullivan & Chen 2024.

### Cited Findings
- Giannantonio & Porciani 2010, "Structure formation from non-Gaussian initial conditions: multivariate biasing, statistics, and comparison with N-body simulations", arXiv:0911.0017, PRD 81, 063530. Introduces multivariate bias in $\delta$ and Gaussian $\phi$, with PBS coefficients from non-Gaussian mass functions. The scale-dependent bias appears at leading order. Validated to $k<0.1$–0.3 $h$/Mpc. — [arXiv:0911.0017](https://arxiv.org/abs/0911.0017)
- Baldauf, Seljak & Senatore 2011, "Primordial non-Gaussianity in the bispectrum of the halo density field", arXiv:1011.1513, JCAP 04 (2011) 006. Tree-level halo bispectrum with multivariate bias. Non-Gaussian modifications of bias parameters (the $b_{\phi\delta}$-type terms) amplify the bispectrum signal. — [arXiv:1011.1513](https://arxiv.org/abs/1011.1513)
- Schmidt & Kamionkowski 2010, "Halo clustering with non-local non-Gaussianity", arXiv:1008.0638, PRD 82, 103002. Generalized PBS $b_\Phi$ for arbitrary bispectra. — [arXiv:1008.0638](https://arxiv.org/abs/1008.0638)
- Assassi, Baumann & Schmidt 2015, "Galaxy bias and primordial non-Gaussianity", arXiv:1510.03723, JCAP 12 (2015) 043. A complete, renormalized PNG bias expansion. The equivalence principle links the power-spectrum scale-dependent bias to the dipolar bispectrum part, which gives a consistency check. — [arXiv:1510.03723](https://arxiv.org/abs/1510.03723)
- Tellarini, Ross, Tasinato & Wands 2015 and 2016: "Non-local bias in the halo bispectrum with primordial non-Gaussianity", arXiv:1504.00324, JCAP 07 (2015) 004; "Galaxy bispectrum, primordial non-Gaussianity and redshift space distortions", arXiv:1603.06814, JCAP 06 (2016) 014. Tree-level bispectrum including $b_{\phi\delta}$-type and tidal terms, plus RSD. Relevant to the CLAUDE.md Tier-2 cross-check. — [arXiv:1504.00324](https://arxiv.org/abs/1504.00324); [arXiv:1603.06814](https://arxiv.org/abs/1603.06814)
- Desjacques, Jeong & Schmidt 2018, "Large-Scale Galaxy Bias", arXiv:1611.09787, Phys. Rept. 733, 1. Review, Sec. 7 on PNG bias. — [arXiv:1611.09787](https://arxiv.org/abs/1611.09787)
- Barreira 2022a: first measurement of $b_{\phi\delta}$ in gravity-only and galaxy-formation simulations. UMF overpredicts halo $b_{\phi\delta}(b_1)$ by up to ~3. Using $0.85\times$UMF for $b_\phi$ inside the formula helps only slightly, so the $b_{\phi\delta}$ breakdown is not inherited only from $b_\phi$. A semi-analytic Tinker-based calculation agrees with the measurements. Best-fit halo relation for $b_1<4$, all $z$: $b_{\phi\delta}(b_1) = 3.85 - 9.49\,b_1 + 3.44\,b_1^2$. — [arXiv:2107.06887 (full text Eq. 3.2)](https://arxiv.org/abs/2107.06887)
- Moradinezhad Dizgah, Biagetti, Sefusatti, Desjacques & Noreña 2021, "Primordial non-Gaussianity from biased tracers: likelihood analysis of real-space power spectrum and bispectrum", arXiv:2010.14523, JCAP 05 (2021) 015. On Eos simulations ($f_{\rm NL}=\pm250, 10$), the P(1-loop)+B(tree) MCMC with all terms linear and quadratic in $f_{\rm NL}$ improves on P alone by >5× with loose priors. Strong priors on $b_\phi$, or UMF relations for both $b_\phi$ and $b_{\phi\delta}$, improve this further by a factor of a few but produce a **significant systematic shift** in $f_{\rm NL}$ at the same $k$ range. — [arXiv:2010.14523](https://arxiv.org/abs/2010.14523)
- Sullivan & Chen 2024: first field-level $b_{\delta\phi}$, in qualitative agreement with SU/UMF but different in detail. Halo $f_{\rm NL}$ is biased when $b_\phi$ and $b_{\delta\phi}$ are marginalized. — [arXiv:2410.18039](https://arxiv.org/abs/2410.18039)
- Lazeyras, Barreira & Schmidt 2021, "Assembly bias in quadratic bias parameters of dark matter halos from forward modeling", arXiv:2106.14713, JCAP 10 (2021) 063, and Barreira, Lazeyras & Schmidt 2021, "Galaxy bias from forward models: linear and second-order bias of IllustrisTNG galaxies", arXiv:2105.02876, JCAP 08 (2021) 029. Field-level $b_2$, $b_K$ and assembly bias, not PNG-specific. They feed into the $b_2$ input of the $b_{\phi\delta}$ UMF formula. — [arXiv:2106.14713](https://arxiv.org/abs/2106.14713); [arXiv:2105.02876](https://arxiv.org/abs/2105.02876)

### Inferences
- The two halo $b_{\phi\delta}$ estimates disagree by a non-trivial amount at our Stage 3 biases. At $b_1=2$ the Barreira fit gives $b_{\phi\delta}\approx3.85-18.98+13.76=-1.4$. The UMF formula gives $b_{\phi\delta}=b_\phi-b_1+1+\delta_c[b_2-\tfrac{8}{21}(b_1-1)] = 2.37+\delta_c(b_2-0.38)$ at $b_1=2$. With the Lazeyras et al. 2016 $b_2(b_1)$ cubic fit ($b_2(2)\approx-0.09$; fit coefficients quoted from memory, arXiv:1511.01096 **[unverified here]**) this is about $+1.6$, compared with $-1.4$ from the Barreira fit, a difference of about 3. This matches Barreira's "up to $\Delta b_{\phi\delta}\sim3$". The arithmetic is mine and should be rechecked with the $b_2$ fit actually used in the code.

  The field-level model in CLAUDE.md fixes $b_{\phi\delta}$ by UMF. A test that swaps in the Barreira fit (or samples $b_{\phi\delta}$) would show whether part of the Andrews $p=1.0\to1.3$ shift is really a $b_{\phi\delta}$ mismatch.
- Moradinezhad Dizgah et al.'s systematic shift under UMF priors is a warning for field-level runs at $k_{\max}=0.1$, which are bispectrum-dominated: fixing both $b_\phi$ and $b_{\phi\delta}$ to UMF may bias $f_{\rm NL}$.

### Gaps
- No measurement of $b_{\phi\delta}$ for quasar-like or LRG-like galaxy samples beyond Barreira 2022a (TNG selections) was found.
- No $b_{\phi\delta}$ prior paper analogous to Moore 2026 or Fondi 2026 exists yet, as far as these searches show.

---

## Q6. PNG simulation suites, IC codes and related products

### Takeaway
Suites available as of 2026:
- **Quijote-PNG**: 1 Gpc/h, $512^3$, 500 realizations per shape and sign at $f_{\rm NL}^{\rm loc}=\pm100$, 2LPTPNG ICs, FoF halos.
- **PNG-UNITsims**: 1 Gpc/h, $4096^3$, $f_{\rm NL}=100$, fixed-and-matched ICs; later $-20$ runs.
- **AbacusPNG**: 2 Gpc/h, $4096^3$, $f_{\rm NL}\in\{-100,-30,0,30,100\}$ × 2 phases, CompaSO halos, light cones and merger trees, DESI-oriented.
- **Eos**: 2 Gpc/h, $1536^3$, $f_{\rm NL}=10,\pm250$ × 10 realizations.
- The older Grossi/Dalal/Pillepich/Desjacques/Wagner–Verde runs, and fast-PM suites such as PNG-pmwd.

IC generators are 2LPTPNG (Scoccimarro et al. 2012), the Wagner & Verde scheme (now GENGARS), and 3LPT at low $z_{\rm start}$, which Gutiérrez Adame et al. 2025 recommend.

### Cited Findings
- Coulton, Villaescusa-Navarro, Jamieson, Baldi et al. 2023a, "Quijote-PNG: Simulations of primordial non-Gaussianity and the information content of the matter field power spectrum and bispectrum", arXiv:2206.01619, ApJ (2023). Shapes: local, equilateral, orthogonal-CMB and orthogonal-LSS. Box 1000 Mpc/h, $512^3$ particles, 500 realizations, 2LPTPNG ICs at $z=127$, particle mass $6.56\times10^{11}\,h^{-1}M_\odot$. Snapshots at $z=0, 0.5, 1, 2, 3$. — [arXiv:2206.01619](https://arxiv.org/abs/2206.01619) (full text Table 1)
- Coulton et al. 2023b, "Quijote-PNG: The information content of the halo power spectrum and bispectrum", arXiv:2206.15450, ApJ 943, 178. Uses FoF halos with $M>3.2\times10^{13}\,h^{-1}M_\odot$ ($\bar n=1.55\times10^{-4}$ at $z=0$) and 15,000 fiducial sims. For local PNG, power-spectrum scale-dependent bias with sample-variance cancellation beats the halo bispectrum, and small scales add minimal information. — [arXiv:2206.15450](https://arxiv.org/abs/2206.15450)
- Jung et al. Quijote-PNG series:
  - arXiv:2206.01624, ApJ 940, 71 (QML estimator, matter). — [arXiv:2206.01624](https://arxiv.org/abs/2206.01624)
  - arXiv:2211.07565, ApJ 948, 135 (QML, halos). — [arXiv:2211.07565](https://arxiv.org/abs/2211.07565)
  - arXiv:2305.10597, ApJ 957, 50 (HMF information). — [arXiv:2305.10597](https://arxiv.org/abs/2305.10597)
  - arXiv:2403.00490, ApJ 976, 109 (optimized summaries). — [arXiv:2403.00490](https://arxiv.org/abs/2403.00490)
- Andrews et al. 2026 used the first 30 Quijote-PNG boxes ($f_{\rm NL}=0,\pm100$) at $z=0.5$ with FoF $b=0.2$. — [arXiv:2603.20855](https://arxiv.org/abs/2603.20855)
- Gutiérrez Adame et al. 2024 (PNG-UNITsims): see Q1. $4096^3$ particles in 1 Gpc/h, $m_p=1.24\times10^9\,h^{-1}M_\odot$, $f_{\rm NL}=100$, phases matched to UNIT, plus lower-resolution LR-UNIT and FastPM companions. — [arXiv:2312.12405](https://arxiv.org/abs/2312.12405)
- Gutiérrez Adame, Avila, Gonzalez-Perez, Hahn et al. 2025, "Accurate N-body simulations with local primordial non-Gaussianities: Initial conditions and aliasing", arXiv:2506.06200, A&A 705, A20. 186 runs varying $z_{\rm start}$, LPT order, $f_{\rm NL}$ and $g_{\rm NL}$. 3LPT from $z_{\rm start}\simeq11.5$ reproduces P, B and HMF to <1%. PNG-term aliasing gives ≤3% small-scale excess in the IC power, which drops below 0.1% by $z=0$. Removing PNG aliasing is "unnecessary" for current simulations. — [arXiv:2506.06200](https://arxiv.org/abs/2506.06200)
- Hadzhiyska, Garrison, Eisenstein & Ferraro 2024 (AbacusPNG). The full text says "10 simulations: five variations of $f_{\rm NL}$ with two realizations each", $f_{\rm NL}=\{-100,-30,0,30,100\}$, $4096^3$ in 2 Gpc/h, $m_p\approx1\times10^{10}$ (4.8× coarser than AbacusSummit), final $z=0.3$, plus two non-PNG "pngbase" boxes. The abstract says "9 boxes", so the count is inconsistent within the paper. — [arXiv:2402.10881](https://arxiv.org/abs/2402.10881)
- Eos suite: used in Moradinezhad Dizgah et al. 2021 (10 realizations per set, $f_{\rm NL}=10,\pm250$, $z=1$ snapshot) and Biagetti, Cole & Shiu 2021 (arXiv:2009.04819, JCAP 04 (2021) 061). The specs (2 Gpc/h, $1536^3$, Gadget2, Rockstar) come from a web-search summary of those papers and were not checked in the full text. I found no dedicated Eos release paper. — [arXiv:2010.14523](https://arxiv.org/abs/2010.14523); [arXiv:2009.04819](https://arxiv.org/abs/2009.04819)
- Wagner, Verde & Boubekeur 2010, "N-body simulations with generic non-Gaussian initial conditions I: Power spectrum and halo mass function", arXiv:1006.5793, JCAP 10 (2010) 022, and Wagner & Verde 2012 (II: halo bias), arXiv:1102.3229. Generic-bispectrum IC method. Outputs were released via the "non-Gaussian comparison project" website. — [arXiv:1006.5793](https://arxiv.org/abs/1006.5793); [arXiv:1102.3229](https://arxiv.org/abs/1102.3229)
- Fondi, Verde, Baldi, Coulton et al. 2025, "GENGARS: accurate non-Gaussian initial conditions with arbitrary bispectrum for N-body simulations", arXiv:2508.01855, JCAP 08 (2026) 020. Builds on Wagner & Verde 2012 and is benchmarked against 2LPTPNG. It has improved accuracy and lower noise by suppressing spurious contributions to the primordial power. — [arXiv:2508.01855](https://arxiv.org/abs/2508.01855)
- Scoccimarro, Hui, Manera & Chan 2012 (2LPTPNG; arXiv:1108.5512) and Crocce, Pueblas & Scoccimarro 2006, "Transients from initial conditions in cosmological simulations", arXiv:astro-ph/0606505, MNRAS 373, 369 (2LPTic). — [arXiv:1108.5512](https://arxiv.org/abs/1108.5512); [arXiv:astro-ph/0606505](https://arxiv.org/abs/astro-ph/0606505)
- Baldi, Fondi, Karagiannis, Moscardini et al. 2024, "Cosmological simulations of scale-dependent primordial non-Gaussianity", arXiv:2407.06641, JCAP 11 (2024) 053. Runs with power-law $f_{\rm NL}(k)$ saturating on small scales. — [arXiv:2407.06641](https://arxiv.org/abs/2407.06641)
- Calles, Contardo, Noreña, Yip et al. 2025, "Primordial non-Gaussianity – fast simulations and persistent summary statistics", arXiv:2512.09852, JCAP 08 (2026) 051. PNG-pmwd: >20,000 fast halo catalogues varying local and equilateral $f_{\rm NL}$, $\Omega_m$ and $\sigma_8$. — [arXiv:2512.09852](https://arxiv.org/abs/2512.09852)
- Breton, Fosalba, Avila, Baldi et al. 2026, "Euclid preparation. Simulated galaxy catalogues for non-standard cosmological models", arXiv:2603.13148 (preprint). Public HOD galaxy catalogues (Rockstar halos) for non-standard cosmologies, including PNG. — [arXiv:2603.13148](https://arxiv.org/abs/2603.13148)
- Historical PNG N-body sets: Dalal et al. 2008 (arXiv:0710.4560), Grossi et al. 2009 (arXiv:0902.2013), Pillepich et al. 2010 (arXiv:0811.4176), Desjacques et al. 2009 (arXiv:0811.2748). — [arXiv:0710.4560](https://arxiv.org/abs/0710.4560); [arXiv:0902.2013](https://arxiv.org/abs/0902.2013); [arXiv:0811.4176](https://arxiv.org/abs/0811.4176); [arXiv:0811.2748](https://arxiv.org/abs/0811.2748)
- Separate-universe $b_\phi$ simulation sets:
  - TNG SU: Barreira et al. 2020, arXiv:2006.09368.
  - Lazeyras et al. 2023 gravity-only sets, including a 250 Mpc/h $1024^3$ Flagship-cosmology box.
  - CAMELS-SAM SU: Perez et al. 2026, 205 $h^{-1}$cMpc, $1280^3$.
  — [arXiv:2006.09368](https://arxiv.org/abs/2006.09368); [arXiv:2209.07251](https://arxiv.org/abs/2209.07251); [arXiv:2602.04987](https://arxiv.org/abs/2602.04987)
- Field-level PNG inference products: Andrews et al. 2024/2026 Euclid FLI (arXiv:2412.11945, A&A 711, A94) released mock data with non-zero $f_{\rm NL}$ and curvature maps. — [arXiv:2412.11945](https://arxiv.org/abs/2412.11945)

### Inferences
- For Stage 3, AbacusPNG gives the closest DESI-like match:
  - $f_{\rm NL}=\pm30$ is closer to DR1 sensitivity than ±100, so response non-linearity is less of a concern.
  - CompaSO halos match DESI AbacusSummit HODs.
  - Two phases per $f_{\rm NL}$ allow matched-phase differencing.
  - The 2 Gpc/h box gives more large-scale modes than Quijote.

  PNG-UNIT has the resolution needed for QSO-host masses ($\sim5\times10^{12}$). Quijote std-res Rockstar is incomplete there (see memory note).
- Quijote-PNG ICs are 2LPTPNG at $z=127$. Gutiérrez Adame 2025 shows that start-redshift and LPT-order choices matter at the percent level but PNG aliasing does not. A DISCO-DJ forward model with de-aliasing is therefore unlikely to mismatch Quijote-PNG because of IC aliasing.

### Gaps
- **No PNG-specific EZmock, Pinocchio, GLAM-PNG or FastPM-PNG mock suite release paper was found** in these searches. A web search for "GLAM-PNG" returned nothing specific. Chaussidon et al. 2024 validated with $f_{\rm NL}$-injected mocks, but their construction was not extracted here. PNG-UNIT companions include FastPM runs, per the PNG-UNIT full text.
- I did not confirm the non-local-shape amplitudes in Quijote-PNG (equilateral and orthogonal $f_{\rm NL}$ values). Only local $\pm100$ was confirmed, via Andrews 2026 and Sullivan & Chen 2024.

---

## Q7. Reconciling $p$ = 1.0 (our Stage 3), 1.3 (Andrews 2026) and 1.6 (DESI QSO)

### Takeaway
The three numbers measure different things, so they are not in conflict:
- **1.6** is a theoretical upper bound for halos observed at formation (EPS, Slosar 2008), adopted by DESI for QSOs. The TNG-based DESI-QSO calibration revises it to $\bar p\simeq1.4$.
- **1.3** is an effective value tuned so that BORG's Eulerian, mass-weighted, fixed-$b_{\phi\delta}$ forward model gives consistent $b_1$ across $f_{\rm NL}=\pm100$ for FoF $M>3\times10^{13}$ halos.
- **1.0** from a direct $\pm100$ response measurement agrees with PNG-UNIT ($p=0.955\pm0.013$ over $2\times10^{10}$–$5\times10^{13}$) and with the 10–20% high-mass suppression ($p_{\rm eff}\approx1.1$–1.2) within plausible errors.

### Cited Findings
- Andrews 2026 tuned $p$ from 1.0 to 1.3 to remove a $b_1$ offset across $f_{\rm NL}$ at fixed ICs. — [arXiv:2603.20855 (App. A)](https://arxiv.org/abs/2603.20855)
- PNG-UNIT: $p=1$ for $10^{12}$–$5\times10^{13}$, $p<1$ below $10^{12}$, combined $0.955\pm0.013$. — [arXiv:2312.12405](https://arxiv.org/abs/2312.12405)
- High-mass halos have $b_\phi$ 10–15% below UMF for all finders. SO and FoF differ at low mass. — [arXiv:1611.04901](https://arxiv.org/abs/1611.04901)
- $b_\phi\approx0.85\times$UMF (TNG gravity-only total-mass), $\approx0.8\times$UMF (FastPM field level), $c\approx0.9$ (AbacusPNG). — [arXiv:2107.06887](https://arxiv.org/abs/2107.06887); [arXiv:2410.18039](https://arxiv.org/abs/2410.18039); [arXiv:2402.10881](https://arxiv.org/abs/2402.10881)
- The field-level $b_\phi$ for Quijote-PNG halos deviates from SU/UMF more than for FastPM, possibly a resolution effect. — [arXiv:2410.18039](https://arxiv.org/abs/2410.18039)
- DESI QSO $\bar p\simeq1.41$, EPS upper bound 1.6. A mass-matched random-history control gives $\bar p=0.99$. — [arXiv:2602.12357](https://arxiv.org/abs/2602.12357)
- The Euclid FLI default $p=0.55$ comes from TNG stellar-mass selection. — [arXiv:2006.09368](https://arxiv.org/abs/2006.09368)

### Inferences
- In Stage 3, report $p$ as measured by the response together with the halo finder, mass definition, mass threshold and weighting (number or mass). Do not compare it directly with Andrews' tuned value without matching those choices.
- For a proper apples-to-apples check:
  1. Rerun the Stage-3 response on FoF $M>10^{13.5}$ at $z=0.5$, mass-weighted, to see whether an effective $p\approx1.3$ appears.
  2. Separately, test whether letting $b_{\phi\delta}$ float, or using the Barreira 2022a fit, removes the need for $p>1$.
- For DR1 forecasting and production, use $p_{\rm QSO}=1.4$ (with the Fondi distribution as prior) and $p_{\rm LRG}=1.0\pm0.1$, consistent with Brown 2026 and Fondi 2026. Keep the 1.0 and 1.6 bracketing runs for comparison with Chaussidon et al.

### Gaps
- No published study varies the halo finder (FoF vs Rockstar/SO) **and** the field weighting in Quijote-PNG at fixed mass cut to measure $p$. This would directly test the Stage-3 vs Andrews difference.
- I could not verify the size of the $b_1$ offset Andrews observed with $p=1.0$. The appendix is qualitative and gives no number.
