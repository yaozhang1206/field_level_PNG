# Methodology literature for problems hit when applying a field-level (BORG-style) forward model to N-body halos

Scope: the specific failure modes we hit on Quijote halos. These are (1) grid window and aliasing mismatch, which we fixed with a 4x CIC supersample, CIC deconvolution and Fourier truncation; (2) the Laplacian bias $b_{\nabla^2}$ and stochasticity; (3) "sigma collapse" when the noise amplitude is free; (4) Lagrangian vs Eulerian bias; (5) model-mismatch validation on halos; (6) posterior widths too narrow by 10–45% for dense halo samples.

Verification convention:
- **[V-INSPIRE]**: the arXiv ID, title, authors and journal were checked against the INSPIRE-HEP API on 2026-10-13.
- **[V-arXiv]**: the abstract was read on the arXiv abstract page.
- **[V-text]**: the full text (arXiv HTML) was read and the quoted passage extracted.
- **[UNVERIFIED]**: taken from memory and not checked in this session. The journal is listed as "preprint" when INSPIRE returned no journal reference.

Several titles in the brief do not exist as written. Section 1 lists each one and the paper that actually matches.

---

## 1. Mass assignment, aliasing, interlacing and grid windows (and how field-level codes handle them)

### Takeaway
The aliasing/window problem is classic. Jing 2005 gives the alias-sum formula. Sefusatti+2016 show that interlacing plus higher-order kernels gives better than 0.01% accuracy up to Nyquist. Field-level EFT codes avoid it by construction: LEFTfield uses NUFFT assignment onto a supersampled grid plus sharp-k cutoffs on the initial field (Λ) and on the likelihood (k_max). LEFTfield's authors state explicitly that using the same CIC kernel in model and data cancels the window only partially. Our fix (4x supersample, CIC deconvolution, truncation to the coarse Fourier modes) is close to the LEFTfield practice. Adding interlacing, or a higher-order/NUFFT kernel, is the natural next refinement.

### Cited Findings

**Classical mass assignment / aliasing**
- Hockney, R. W. & Eastwood, J. W. (1981; 2nd ed. 1988), *Computer Simulation Using Particles* (McGraw-Hill / Adam Hilger; book, no arXiv) **[UNVERIFIED bibliographic details; standard citation]**. This is the origin of the NGP/CIC/TSC assignment windows $W(k)=\prod_i {\rm sinc}^p(k_i\Delta/2)$ and of interlacing. Andrews+2026 (Quijote) cite it for their CIC step — [Andrews+2026 text](https://arxiv.org/abs/2603.20855).
- Jing, Y. P. (2005), "Correcting for the alias effect when measuring the power spectrum using FFT", arXiv:astro-ph/0409240, ApJ 620, 559 **[V-INSPIRE, V-arXiv]**. Derives $\langle|\delta^f(k)|^2\rangle=\sum_{\vec n}[|W(\vec k+2k_N\vec n)|^2P(\vec k+2k_N\vec n)+|W|^2/N]$, which shows that the window and the alias images enter together. Gives analytic shot-noise alias sums for NGP/CIC/TSC and an iterative deconvolution. *Relevance:* this is the formula that predicts why CIC-gridded halos and a CIC-gridded model with a different particle load (one particle per cell vs. discrete halos) have different alias sums, which is our mismatch — [arXiv](https://arxiv.org/abs/astro-ph/0409240).
- Cui, W., Liu, L., Yang, X., Wang, Y., Feng, L., Springel, V. (2008), "An ideal mass assignment scheme for measuring the Power Spectrum with FFTs", arXiv:0804.0070, ApJ 687, 738 **[V-INSPIRE, V-arXiv]**. Instead of deconvolving, choose a kernel that is compact in real space and close to top-hat in Fourier space (Daubechies-wavelet scaling functions). *Relevance:* an alternative to our CIC + deconvolution, with less leakage across the coarse Nyquist — [arXiv](https://arxiv.org/abs/0804.0070).
- Colombi, S., Jaffe, A., Novikov, D., Pichon, C. (2009), "Accurate estimators of power spectra in N-body simulations", arXiv:0811.0313, MNRAS 393, 511 **[V-INSPIRE]**. Uses Taylor/foldover techniques to reduce aliasing. Background for choosing the supersampling factor — [arXiv](https://arxiv.org/abs/0811.0313).
- Sefusatti, E., Crocce, M., Scoccimarro, R., Couchman, H. (2016), "Accurate Estimators of Correlation Functions in Fourier Space", arXiv:1512.07295, MNRAS 460, 3624 **[V-INSPIRE, V-arXiv]**. Interlaced grids remove the odd alias images, which carry most of the aliasing. Fourth-order kernels suppress the rest. Fourth-order assignment plus interlacing gives P and B with systematic bias below 0.01% up to the grid Nyquist frequency. *Relevance:* interlacing on our 4x mesh would remove the dominant remaining alias images at about 2x the assignment cost — [arXiv](https://arxiv.org/abs/1512.07295).
- Hand, N., Feng, Y., Beutler, F., Li, Y., et al. (2018), "nbodykit: an open-source, massively parallel toolkit for large-scale structure", arXiv:1712.05834, AJ 156, 160 **[V-INSPIRE]**. Reference implementation of compensated CIC/TSC/PCS windows and interlacing; useful for checking our JAX deconvolution — [arXiv](https://arxiv.org/abs/1712.05834).

**How BORG and the EFT field-level codes treat the grid**
- Jasche, J. & Wandelt, B. D. (2013), "Bayesian physical reconstruction of initial conditions from large scale structure surveys", arXiv:1203.3639, MNRAS 432, 894 **[V-INSPIRE]**. The original BORG paper (2LPT, HMC on the initial conditions) — [arXiv](https://arxiv.org/abs/1203.3639).
- Jasche, J. & Lavaux, G. (2019), "Physical Bayesian modelling of the non-linear matter distribution: new insights into the Nearby Universe", arXiv:1806.11117, A&A 625, A64 **[V-INSPIRE]**. BORG-PM: a particle-mesh forward model with CIC particle-to-grid assignment — [arXiv](https://arxiv.org/abs/1806.11117).
- Nguyen, N.-M., Schmidt, F., Lavaux, G., Jasche, J. (2021), "Impacts of the physical data model on the forward inference of initial conditions from biased tracers", arXiv:2011.06587, JCAP 03 (2021) 058 **[V-INSPIRE, V-text]**. BORG discretizes the matter and halo fields by CIC projection. The authors note that the CIC filter "is not localized in Fourier space and thus does not completely remove modes with $k>k_{\rm Ny}$. This might allow non-linear modes to slip into the inference", in contrast to a sharp-k filter. They test grid resolution (§2.4, §4.1.2). *Relevance:* this is the closest BORG-side description of our gridding failure — [arXiv](https://arxiv.org/abs/2011.06587).
- Schmidt, F., Elsner, F., Jasche, J., Nguyen, N.-M., Lavaux, G. (2019), "A rigorous EFT-based forward model for large-scale structure", arXiv:1808.02002, JCAP 01 (2019) 042 **[V-INSPIRE]**. The brief calls this "cosmological parameters from the field level", but that is not the actual title. It introduces the EFT likelihood with a sharp-k cutoff Λ that integrates out small-scale modes — [arXiv](https://arxiv.org/abs/1808.02002).
- Elsner, F., Schmidt, F., Jasche, J., Lavaux, G., Nguyen, N.-M. (2020), "Cosmology inference from a biased density field using the EFT-based likelihood", arXiv:1906.07143, JCAP 01 (2020) 029 **[V-INSPIRE]**. First cosmology inference with the EFT likelihood — [arXiv](https://arxiv.org/abs/1906.07143).
- Cabass, G. & Schmidt, F. (2020), "The EFT Likelihood for Large-Scale Structure", arXiv:1909.04022, JCAP 04 (2020) 042 **[V-INSPIRE, V-arXiv]**. Functional derivation of the Fourier-space conditional likelihood, including corrections from non-Gaussian stochasticity and higher-derivative terms, "with a particular eye towards primordial non-Gaussianity" — [arXiv](https://arxiv.org/abs/1909.04022).
- Schmidt, F., Cabass, G., Jasche, J., Lavaux, G. (2020), "Unbiased Cosmology Inference from Biased Tracers using the EFT Likelihood", arXiv:2004.06707, JCAP 11 (2020) 008 **[V-INSPIRE, V-arXiv]**. Adds a sharp cutoff on the *initial* conditions, which lowers the σ8 bias by a factor of several. With Λ=0.1 h/Mpc they recover σ8 to ≲6% (2nd order) and ≤4% (3rd order) on halos. *Relevance:* our model has no separate Λ on ε, so 2LPT/b2/bK/fNLφ² generate power from modes near the grid scale — [arXiv](https://arxiv.org/abs/2004.06707).
- Schmidt, F. (2021), "An n-th order Lagrangian Forward Model for Large-Scale Structure", arXiv:2012.09837, JCAP 04 (2021) 033 **[V-INSPIRE, V-text]**. Gives two grid-sizing rules for nonlinear operators. The "no mode left behind" rule captures all couplings up to Λ. The "generalized Orszag rule" (Eq. 26) only requires the lowest aliased modes to lie above the analysis k_max; Orszag's 3/2 rule is the 2nd-order special case. *Relevance:* this gives a principled grid size for computing $\phi^2$, $\delta^2$, $K^2$ and $\delta\phi$ before truncating to the coarse modes — [arXiv](https://arxiv.org/abs/2012.09837).
- Schmidt, F. (2021), "Sigma-Eight at the Percent Level: The EFT Likelihood in Real Space", arXiv:2009.14176, JCAP 04 (2021) 032 **[V-INSPIRE, V-arXiv]**. A real-space likelihood formulation (allows survey windows) with a Lagrangian forward model. σ8 is unbiased at 2% for Λ ≤ 0.14 h/Mpc on halos — [arXiv](https://arxiv.org/abs/2009.14176).
- Stadler, J., Schmidt, F., Reinecke, M. (2023), "Cosmology inference at the field level from biased tracers in redshift-space", arXiv:2303.09876, JCAP 10 (2023) 069 **[V-INSPIRE, V-arXiv]**. LEFTfield RSD extension. Requires a systematic expansion of velocity bias. Gives unbiased f at the few-percent level on halos for 0.08 ≤ Λ ≤ 0.20 h/Mpc — [arXiv](https://arxiv.org/abs/2303.09876).
- Stadler, J., Schmidt, F., Reinecke, M. (2025), "Fast, accurate and perturbative forward modeling of galaxy clustering. Part I. Galaxies in the restframe", arXiv:2409.10937, JCAP 04 (2025) 089 **[V-INSPIRE, V-text]**. **The most directly relevant paper for Problem 1.**
  - §3.2: "There are two instances of density assignment in a field-level analysis … If the same kernels are used in either operation, one can expect the kernel to cancel to some extent. On the other hand, data and forward model contain different particle numbers and small scales are populated differently … rendering the cancellation incomplete."
  - They compare CIC with NUFFT assignment ("exponential of semicircle" kernel on a super-sampled grid). Model–simulation residuals "converge much faster at increasing resolution for the NUFFT scheme than for the CIC kernel".
  - §3.3 separates truncation from aliasing errors and advises against low-pass filtering the shift vector.
  - §3.5 covers Lagrangian-bias weighted displacements.
  - Appendix B covers Fourier-space resizing of arrays.
  - [arXiv](https://arxiv.org/abs/2409.10937)
- Stadler, J., Schmidt, F., Reinecke, M., Esposito, M. (2025), "Fast, accurate and perturbative forward modeling of galaxy clustering. Part II. Redshift space", arXiv:2411.04513, JCAP 11 (2025) 055 **[V-INSPIRE, V-arXiv]**. Numerical and perturbative error budget for momentum/velocity/RSD density. Recovers f from halos at <1% statistical error with the third-order model. Uses fixed initial conditions with marginalized bias and noise — [arXiv](https://arxiv.org/abs/2411.04513).
- Babić, I., Schmidt, F., Tucci, B. (2022), "BAO scale inference from biased tracers using the EFT likelihood", arXiv:2203.06177, JCAP 08 (2022) 007 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2203.06177).
- Babić, I., Schmidt, F., Tucci, B. (2025), "Straightening the ruler: field-level inference of the BAO scale with LEFTfield", arXiv:2407.01524, JCAP 11 (2025) 066 **[V-INSPIRE]**. Mock data generated at a higher cutoff introduce deliberate model mismatch. Residual BAO-scale bias stays below 2% and the error is about 30% smaller than standard reconstruction (search-result summary) — [arXiv](https://arxiv.org/abs/2407.01524).
- Babić, I., Schmidt, F., Tucci, B. (2026), "Improving constraints on baryon acoustic oscillations with field-level inference" (arXiv v1 title: "Forward vs Backward: Improving BAO Constraints with Field-Level Inference"), arXiv:2505.13588, PRD 113, L061304 **[V-INSPIRE, V-text]**. Field-level inference on rest-frame halos, sampling the ICs, bias and noise. Noise RMS is $\sigma_\epsilon$ plus a subleading $k^2$ term. BAO constraint about 1.2–1.4x tighter than reconstruction. No 2025 *redshift-space* BAO paper by Babić was found — [arXiv](https://arxiv.org/abs/2505.13588).
- Brief's title "Nguyen et al. 2021 'Taming redshift-space distortion effects'": **no such field-level paper found.** That title belongs to D'Amico, Senatore, Zhang, Nishimichi, arXiv:2110.00016 (EFTofLSS multipoles, JCAP 2024; not field level). The intended Nguyen+2021 reference is almost certainly arXiv:2011.06587 ("Impacts of the physical data model") — [search/arXiv](https://arxiv.org/abs/2110.00016).

**DISCO-DJ / sheet-based discreteness control**
- List, F., Hahn, O., Flöss, T., Winkler, L. (2026), "DISCO-DJ II: a differentiable particle-mesh code for cosmology", arXiv:2510.05206, JCAP 06 (2026) 033 **[V-INSPIRE, V-arXiv]**. The abstract says the code "incorporates a suite of advanced techniques" to control discreteness, "for example a custom non-uniform FFT implementation for force evaluation". It supports forward- and reverse-mode differentiation with an adjoint, and includes a field-level σ8 demo on a Gadget matter field. *Note:* the full HTML could not be retrieved. The specific de-aliasing and sheet-resampling options cited in CLAUDE.md were **not independently verified** in this session — [arXiv](https://arxiv.org/abs/2510.05206).
- Hahn, O., List, F., Porqueres, N. (2024), "DISCO-DJ I: a differentiable Einstein-Boltzmann solver for cosmology", arXiv:2311.03291, JCAP 06 (2024) 063 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2311.03291).
- Rampf, C., List, F., Hahn, O. (2025), "BullFrog: multi-step perturbation theory as a time integrator for cosmological simulations", arXiv:2409.19049, JCAP 02 (2025) 020 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2409.19049).
- Abel, T., Hahn, O., Kaehler, R. (2012), "Tracing the Dark Matter Sheet in Phase Space", arXiv:1111.3944, MNRAS 427, 61 **[V-INSPIRE]**. Tetrahedral sheet tessellation as a discreteness-free density estimator — [arXiv](https://arxiv.org/abs/1111.3944).
- Hahn, O., Abel, T., Kaehler, R. (2013), "A new approach to simulating collisionless dark matter fluids", arXiv:1210.6652, MNRAS 434, 1171 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/1210.6652).
- Hahn, O. & Angulo, R. E. (2016), "An adaptively refined phase-space element method for cosmological simulations and collisionless dynamics", arXiv:1501.01959, MNRAS 455, 1115 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/1501.01959).
- Stücker, J., Hahn, O., Angulo, R. E., White, S. D. M. (2020), "Simulating the Complexity of the Dark Matter Sheet I: Numerical Algorithms", arXiv:1909.00008, MNRAS 495, 4943 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/1909.00008).
- Hahn, O., Rampf, C., Uhlemann, C. (2021), "Higher order initial conditions for mixed baryon–CDM simulations", arXiv:2008.09124, MNRAS 503, 426 **[V-INSPIRE]**. monofonIC nLPT with Orszag-type de-aliasing of the LPT source terms. This is the lineage of DISCO-DJ's nLPT — [arXiv](https://arxiv.org/abs/2008.09124).
- Michaux, M., Hahn, O., Rampf, C., Angulo, R. E. (2020), "Accurate initial conditions for cosmological N-body simulations: Minimizing truncation and discreteness errors", arXiv:2008.09588, MNRAS 500, 663 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2008.09588).
- Other differentiable PM codes, for comparison of assignment choices:
  - Modi, Lanusse, Seljak (2021), FlowPM, arXiv:2010.11847, Astron. Comput. 37, 100505 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2010.11847).
  - Li, Lu, Modi, Jamieson et al. (2022), pmwd, arXiv:2211.09958, preprint in INSPIRE **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2211.09958).

### Inferences
- Our symptom (σ collapse plus parameter shifts that disappear after supersample/deconvolve/truncate) matches the Stadler+2025 Part I statement: identical kernels only partially cancel when the data are discrete halos and the model is one particle per cell. It also matches Nguyen+2021's observation that CIC leaks super-Nyquist nonlinear modes into the likelihood.
- Cheap next steps, ordered by expected value:
  1. Interlacing on the 4x mesh (Sefusatti+2016).
  2. A sharp-k cut Λ on ε, separate from the likelihood k_max (Schmidt+2020).
  3. Sizing the mesh for the quadratic operators ($\phi^2$, $\delta^2$, $K^2$, $\delta\phi$) by the generalized Orszag rule (Schmidt 2021).
  4. Optionally, swapping CIC for an ES/NUFFT kernel (Stadler+2025 Part I) or the DISCO-DJ NUFFT.
- The likelihood should then use only modes with $|k| \le k_{\max}$, preferably a sphere rather than the coarse-grid cube. Every LEFTfield paper does this.

### Gaps
- I could not retrieve the DISCO-DJ II full text to confirm which de-aliasing and sheet options are differentiable at our settings. Stage 0 of CLAUDE.md still needs this check.
- No paper was found that quantifies the f_NL bias caused specifically by CIC window mismatch between halo data and a one-particle-per-cell model. The literature quantifies it for σ8, f and the BAO scale only.
- I did not find a Porqueres paper on gridding or aliasing; the only Porqueres paper verified here is DISCO-DJ I.

---

## 2. Higher-derivative (Laplacian) bias and stochasticity in field-level likelihoods

### Takeaway
$b_{\nabla^2\delta}$ is physical, negative, and roughly $-R_L^2(M)$ for halos. Stochasticity of halos is generically non-Poisson: sub-Poisson for massive halos because of exclusion, with scale-dependent $k^2$ (and $k^2\mu^2$ in redshift space) corrections. Good field-level bias models give a white residual close to Poisson. Linear-bias residuals can exceed Poisson by factors of a few and are scale-dependent. Our 62.5 Mpc/h cell gives a CIC window $k^2$ coefficient of about $\Delta^2/12 \approx 325\,(h^{-1}{\rm Mpc})^2$, far larger than $R_L^2$. A free $b_{\nabla^2}$ will therefore absorb any residual window mismatch, not just halo physics.

### Cited Findings
- Desjacques, V., Jeong, D., Schmidt, F. (2018), "Large-Scale Galaxy Bias", arXiv:1611.09787, Phys. Rept. 733, 1 **[V-INSPIRE]**. The standard reference for the complete bias basis, higher-derivative terms, stochasticity and PNG bias ($b_\phi$, $b_{\phi\delta}$) — [arXiv](https://arxiv.org/abs/1611.09787).
- Lazeyras, T. & Schmidt, F. (2019), "A robust measurement of the first higher-derivative bias of dark matter halos", arXiv:1904.11294, JCAP 11 (2019) 041 **[V-INSPIRE, V-arXiv]**. Amplified-mode simulations find $b_{\nabla^2\delta}<0$ at all masses, roughly following $-R_L^2(M)$. *Relevance:* gives a physical prior on $b_{\nabla^2}$ for our mass-cut halo samples — [arXiv](https://arxiv.org/abs/1904.11294).
- Fujita, T., Mauerhofer, V., Senatore, L., Vlah, Z., Angulo, R. (2020), "Very Massive Tracers and Higher Derivative Biases", arXiv:1609.00717, JCAP 01 (2020) 009 **[V-INSPIRE, V-arXiv]**. Highly biased tracers underperform at fixed PT order because of larger higher-derivative biases. Adding them makes all tracers perform comparably. *Relevance:* QSO-like, very massive halos need $b_{\nabla^2}$ — [arXiv](https://arxiv.org/abs/1609.00717).
- Abidi, M. M. & Baldauf, T. (2018), "Cubic Halo Bias in Eulerian and Lagrangian Space", arXiv:1802.07622, JCAP 07 (2018) 029 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/1802.07622).
- Hamaus, N., Seljak, U., Desjacques, V., Smith, R. E., Baldauf, T. (2010), "Minimizing the Stochasticity of Halos in Large-Scale Structure Surveys", arXiv:1004.5377, PRD 82, 043515 **[V-INSPIRE, V-arXiv]**.
  - The stochasticity matrix has non-zero off-diagonal mass-bin correlations, and the highest-mass halos are sub-Poisson.
  - Mass weighting minimizes the stochasticity.
  - *Relevance:* dense, high-mass samples have true noise below $1/\bar n$. This is consistent with σ wanting to drop, and means a Poisson-fixed σ is wrong for them.
  - [arXiv](https://arxiv.org/abs/1004.5377)
- Baldauf, T., Seljak, U., Smith, R. E., Hamaus, N., Desjacques, V. (2013), "Halo stochasticity from exclusion and nonlinear clustering", arXiv:1305.2917, PRD 88, 083507 **[V-INSPIRE, V-arXiv]**. Exclusion gives a negative low-k correction to $1/\bar n$, and nonlinear clustering a positive one. The stochasticity is scale-dependent "even for k<0.1 h/Mpc", with mass and redshift dependence — [arXiv](https://arxiv.org/abs/1305.2917).
- Schmittfull, M., Simonović, M., Assassi, V., Zaldarriaga, M. (2019), "Modeling Biased Tracers at the Field Level", arXiv:1811.10640, PRD 100, 043514 **[V-INSPIRE, V-arXiv]**.
  - With shifted (bulk-flow-resummed) operators, the model error is "close to the Poisson shot noise" and "rather scale-independent".
  - With linear bias it is up to a few times Poisson on large scales and already scale-dependent in the linear regime.
  - Mass weighting reduces the error by up to 10x.
  - [arXiv](https://arxiv.org/abs/1811.10640)
- Schmittfull, M., Simonović, M., Ivanov, M. M., Philcox, O. H. E., Zaldarriaga, M. (2021), "Modeling Galaxies in Redshift Space at the Field Level", arXiv:2012.03334, JCAP 05 (2021) 059 **[V-INSPIRE, V-arXiv]**. Residual noise is white on large scales and comparable to shot noise, with a $k^2\mu^2$ correction (larger noise along the line of sight). *Relevance:* our redshift-space Quijote runs need an anisotropic noise term or a lower k_max — [arXiv](https://arxiv.org/abs/2012.03334).
- Obuljen, A., Simonović, M., Schneider, A., Feldmann, R. (2023), "Modeling HI at the field level", arXiv:2207.12398, PRD 108, 083528 **[V-INSPIRE, V-arXiv]**. The field-level model error is nearly Gaussian with a flat spectrum, "significantly lower than the standard noise inferred from the HI power spectrum". This shows that the field-level noise and the P(k)-inferred shot noise are different quantities — [arXiv](https://arxiv.org/abs/2207.12398).
- Kokron, N., DeRose, J., Chen, S.-F., White, M., Wechsler, R. H. (2022), "Priors on red galaxy stochasticity from hybrid effective field theory", arXiv:2112.00012, MNRAS 514, 2198 **[V-INSPIRE, V-arXiv]**. HEFT used as a field-level model. The host halos are sub-Poisson, while the galaxies range from sub- to super-Poisson. The paper gives priors on non-Poisson stochasticity — [arXiv](https://arxiv.org/abs/2112.00012).
- Cabass, G. & Schmidt, F. (2020), "The Likelihood for LSS: Stochasticity of Bias Coefficients at All Orders", arXiv:2004.00617, JCAP 07 (2020) 051 **[V-INSPIRE]**. Theory for density-dependent (stochastic-bias) noise in the likelihood — [arXiv](https://arxiv.org/abs/2004.00617).
- Rubira, H. & Schmidt, F. (2024), "The Renormalization Group for Large-Scale Structure: Origin of Galaxy Stochasticity", arXiv:2404.16929, JCAP 10 (2024) 092 **[V-INSPIRE, V-arXiv]**. Noise parameters run with the cutoff Λ, and "a single nonlinear bias term generates all stochastic moments through RG evolution". *Relevance:* the effective noise depends on the analysis grid/cutoff, so it should not be fixed at $1/\bar n$ — [arXiv](https://arxiv.org/abs/2404.16929).
- Rubira, H. & Schmidt, F. (2024), "Galaxy bias renormalization group", arXiv:2307.15031, JCAP 01 (2024) 031 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2307.15031).
- Ivanov, M. M., Cuesta-Lazaro, C., Mishra-Sharma, S., Obuljen, A., Toomey, M. W. (2024), "Full-shape analysis with simulation-based priors: Constraints on single field inflation from BOSS", arXiv:2402.13310, PRD 110, 063538 **[V-INSPIRE]**. HOD-informed priors on EFT bias and stochastic parameters. A template for priors on $b_{\nabla^2}$ and noise — [arXiv](https://arxiv.org/abs/2402.13310).
- Andrews, A., Jasche, J., Lavaux, G., Coulton, W., Villaescusa-Navarro, F., et al. (2026), "Field-Level Inference of Primordial Non-Gaussianity with the Quijote Simulation Suite", arXiv:2603.20855, preprint **[V-INSPIRE, V-text]**. Their bias model includes a Laplacian term acting on $\nabla^2\delta$ "at the assumed resolution of the inference". It is evaluated in Lagrangian space, with Gaussian prior σ = 3 or 5 on $b_2$, $b_K$ and $b_\nabla$ — [arXiv](https://arxiv.org/abs/2603.20855).

### Inferences
- Order of magnitude (my estimate): for $M\sim3\times10^{13}\,h^{-1}M_\odot$, $R_L\approx4.3\,h^{-1}$Mpc, so $R_L^2\approx19\,(h^{-1}{\rm Mpc})^2$ and $k^2R_L^2\approx0.2$ at k=0.1. The CIC window at Δ=62.5 has $1-W(k)\approx k^2\Delta^2/12$, a coefficient of about $325\,(h^{-1}{\rm Mpc})^2$, roughly 15x larger. A fitted $b_{\nabla^2}$ much larger than $R_L^2$ therefore signals residual window mismatch, not halo physics. Check the posterior of $b_{\nabla^2}$ against the Lazeyras & Schmidt $-R_L^2$ scaling as a diagnostic for the gridding fix.
- Because $b_{\nabla^2}k^2$ and a scale-dependent noise $k^2$ term both act at high k, while $b_\phi f_{\rm NL}/T(k)\propto k^{-2}$ acts at low k, $b_{\nabla^2}$ should be only weakly degenerate with $f_{\rm NL}$. Its main effect is on the noise level and on $b_1$, which feeds into $b_\phi$.

### Gaps
- I found no field-level paper that jointly fits $b_{\nabla^2}$ and $f_{\rm NL}$ and reports their correlation. Andrews+2026 include $b_\nabla$, but I did not extract a correlation.
- No published field-level prior on scale-dependent noise specifically for QSO-like halos at z≈1 was found.

---

## 3. "Sigma collapse" / noise-level inference in BORG and EFT likelihoods

### Takeaway
Sigma collapse is a known, named failure mode. The term comes from Nguyen+2021 [V-text, as attributed by Rubira & Schmidt 2025]. Current practice is one of:
- fix σ to the Poisson value (Nguyen+2021);
- fix σ from a fixed-IC calibration run (Andrews+2026);
- impose an informative lower bound (Nguyen+2024 PRL, Beyond-2pt).

The first principled cure is the non-Gaussian, density-dependent "noise-field" likelihood of Rubira & Schmidt (2025/2026). With it, σ stays near the Poisson expectation on halos. Andrews+2026 also report that CIC smoothing alone pushes the inferred noise to about 1/4 of Poisson.

### Cited Findings
- Nguyen, N.-M., Schmidt, F., Lavaux, G., Jasche, J. (2021), arXiv:2011.06587, JCAP 03 (2021) 058 **[V-INSPIRE, V-text]**. "Letting the Gaussian variance … free is generally problematic for cases where the average Poisson variance of tracer counts-in-cell is small … the variance parameter collapses to a very small value … and the chain only explores a narrow region around this value." They fix σ² in all Gaussian-likelihood runs and suggest a density-dependent variance as future work — [arXiv](https://arxiv.org/abs/2011.06587).
- Rubira, H. & Schmidt, F. (2026), "Non-Gaussian galaxy stochasticity and the noise-field formulation", arXiv:2511.05484, JCAP 08 (2026) 023 **[V-INSPIRE, V-text]**.
  - "The noise amplitude drifts to unphysically small values in FLI inferences applied to nonlinear tracers such as halos [Nguyen+2021, Nguyen+2024], HOD-based catalogs [Beyond-2pt] or mock data with non-Gaussian noise [Akitsu+2025] … The term 'sigma collapse' was coined for this phenomenon [Nguyen+2021]."
  - Switching off the non-Gaussian and density-dependent noise terms reproduces the collapse. With them on, "'sigma collapse' does not occur" and σ stays near the Poisson value.
  - This is the first EFT field-level inference with the full stochastic model, applied to halos with LEFTfield.
  - [arXiv](https://arxiv.org/abs/2511.05484)
- Nguyen, N.-M., Schmidt, F., Tucci, B., Reinecke, M., Kostić, A. (2024), "How Much Information Can Be Extracted from Galaxy Clustering at the Field Level?", arXiv:2403.03220, PRL 133, 221006 **[V-INSPIRE, V-text]**. Supplementary A.3: "We find it necessary to impose an informative lower limit on the noise level, as the FBI MCMC chains tend to get stuck in a region of unphysically low σ. The causes of this drift are still under investigation." The lower limit is set well below published halo-stochasticity measurements — [arXiv](https://arxiv.org/abs/2403.03220).
- Krause, E., Kobayashi, Y., Salcedo, A. N., Ivanov, M. M., et al. (Beyond-2pt Collaboration) (2025), "A Parameter-Masked Mock Data Challenge for Beyond-Two-Point Galaxy Clustering Statistics", arXiv:2405.02252, ApJ 990, 99 **[V-INSPIRE, V-text]**. The LEFTfield EFT field-level entry analysed 10 real-space HOD boxes with only σ8 free and reported the average of 10 posteriors. Rubira & Schmidt cite it as a sigma-collapse instance on HOD catalogues — [arXiv](https://arxiv.org/abs/2405.02252).
- Akitsu, K., Simonović, M., Chen, S.-F., Cabass, G., Zaldarriaga, M. (2025), "Cosmology inference with perturbative forward modeling at the field level: a comparison with joint power spectrum and bispectrum analyses", arXiv:2509.09673, preprint **[V-INSPIRE, V-text]**. A Gaussian field-level likelihood fit to mocks with Poissonian or density-dependent noise gives "significant biases in the inferred cosmological parameters". "The success of field-level inference critically depends on using the correct likelihood." §VII.2 adds a density-dependent variance in position space — [arXiv](https://arxiv.org/abs/2509.09673).
- Andrews+2026 (Quijote-PNG), arXiv:2603.20855 **[V-text]**:
  - They estimate σ with the ICs fixed to truth, then fix σ during IC sampling "to ensure stabilisation of the inference process", citing Nguyen+2021 on "sigma collapse".
  - σ is fixed at the *upper* end of the fixed-IC posterior "to provide additional robustness to the mismatch of the model".
  - The inferred noise is about one-fourth of the Poisson expectation, "primarily due to the use of the CIC kernel for mass assignment".
  - In Appendix D, a linear bias model shows collapse, while their full bias model "appears to stabilise it". The inferred σ correlates with $f_{\rm NL}$.
  - [arXiv](https://arxiv.org/abs/2603.20855)
- Kostić, A., Nguyen, N.-M., Schmidt, F., Reinecke, M. (2023), "Consistency tests of field level inference with the EFT likelihood", arXiv:2212.07875, JCAP 07 (2023) 063 **[V-INSPIRE, V-arXiv, V-text]**. With mis-specification injected by generating data at $\Lambda_0>\Lambda$, they still recover the amplitude, ICs, bias *and noise* parameters without bias. They note that gravitational coupling generates density-dependent noise (non-diagonal in Fourier space), which they neglect — [arXiv](https://arxiv.org/abs/2212.07875).
- Tucci, B. & Schmidt, F. (2024), "EFTofLSS meets simulation-based inference: σ8 from biased tracers", arXiv:2310.03741, JCAP 05 (2024) 063 **[V-INSPIRE]**. SBI on P+B from the LEFTfield forward model, the SBI counterpart used for comparison in the PRL — [arXiv](https://arxiv.org/abs/2310.03741).
- McAlpine, S., Jasche, J., Ata, M., Lavaux, G., et al. (2025), "The Manticore Project I: a digital twin of our cosmic neighbourhood from Bayesian field-level analysis", arXiv:2505.10682, MNRAS 540, 716 **[V-INSPIRE]**. Source of the "physics-informed" variance prior on the white-noise field adopted by Andrews+2026 to regularize the ICs. Note this is a different paper from the Manticore-Deep reference in CLAUDE.md — [arXiv](https://arxiv.org/abs/2505.10682).

### Inferences
- Mechanism (my reading of the sources): with a Gaussian likelihood whose variance is a free constant, the ICs (about $10^5$–$10^6$ degrees of freedom) can absorb part of the noise realization. The likelihood then rewards smaller σ, which in turn lets the ICs fit more noise, a runaway that only the white-noise prior resists. Non-Gaussian or density-dependent noise that a constant-σ Gaussian cannot describe strengthens the drive. The dense-sample regime (small Poisson variance per cell) is exactly where Nguyen+2021 saw collapse.
- For our pipeline:
  1. Keep σ fixed from a fixed-IC calibration run, as Andrews+2026 do, and report sensitivity to that value.
  2. Do not fix σ at $1/\bar n$: the CIC (or deconvolved-CIC) window and sub-Poisson halos both change the effective noise.
  3. Treat Rubira & Schmidt's noise-field likelihood as the principled Stage 3 fix.

### Gaps
- Whether sigma collapse persists with a deconvolved, sharp-k-truncated Fourier-space likelihood (our current setup) is not addressed in any paper found. All the reported cases are real-space or CIC.
- No paper reports how much fixing σ biases σ(f_NL) specifically.

---

## 4. Lagrangian vs Eulerian bias at the field level

### Takeaway
At fixed perturbative order the two bases are equivalent and differ only at higher order. Lagrangian bias advected by the (N-body or LPT) displacement, as in HEFT and LEFTfield's Lagrangian mode, needs no cut on the evolved field and resums bulk flows. Eulerian shifted operators (Schmittfull+2019) achieve the same resummation. Our "one particle per cell + Lagrangian weights + CIC" is the HEFT/LEFTfield-Lagrangian construction. Its numerical subtlety is the weighted-displacement assignment step, which Stadler+2025 analyse.

### Cited Findings
- Matsubara, T. (2008), "Nonlinear perturbation theory with halo bias and redshift-space distortions via the Lagrangian picture", arXiv:0807.1733, PRD 78, 083519 **[V-INSPIRE]**. Foundational Lagrangian-bias PT — [arXiv](https://arxiv.org/abs/0807.1733).
- Modi, C., Castorina, E., Seljak, U. (2017), "Halo bias in Lagrangian Space: Estimators and theoretical predictions", arXiv:1612.01621, MNRAS 472, 3959 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/1612.01621).
- Modi, C., Chen, S.-F., White, M. (2020), "Simulations and symmetries", arXiv:1910.07097, MNRAS 492, 5754 **[V-INSPIRE, V-arXiv]**. The HEFT origin paper: N-body displacements combined with a quadratic Lagrangian bias expansion. More than a factor of two gain in scale at low z, more modest gains for highly biased tracers at high z. *Relevance:* our QSO-like regime (high bias, high z) is where Lagrangian-vs-model-dynamics gains are smallest — [arXiv](https://arxiv.org/abs/1910.07097).
- Kokron, N., DeRose, J., Chen, S.-F., White, M., Wechsler, R. H. (2021), "The cosmology dependence of galaxy clustering and lensing from a hybrid N-body–perturbation theory model", arXiv:2101.11014, MNRAS 505, 1422 **[V-INSPIRE]**. Aemulus-ν HEFT emulator — [arXiv](https://arxiv.org/abs/2101.11014).
- Zennaro, M., Angulo, R. E., Pellejero-Ibáñez, M., Stücker, J., Contreras, S., Aricò, G. (2023), "The BACCO simulation project: biased tracers in real space", arXiv:2101.12187, MNRAS 524, 2407 **[V-INSPIRE]** (the brief cites it as "Zennaro et al. 2021 hybrid Lagrangian bias"). Lagrangian bias expansion advected with N-body displacements, including a Laplacian term — [arXiv](https://arxiv.org/abs/2101.12187).
- Chen, S.-F., Vlah, Z., Castorina, E., White, M. (2021), "Redshift-Space Distortions in Lagrangian Perturbation Theory", arXiv:2012.04636, JCAP 03 (2021) 100 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2012.04636).
- Schmittfull+2019, arXiv:1811.10640 **[V-arXiv]**. Eulerian *shifted* operators automatically resum bulk flows and give near-Poisson, scale-independent residuals. The standard Eulerian bias on the nonlinear matter field gives larger, more scale-dependent errors — [arXiv](https://arxiv.org/abs/1811.10640).
- Nguyen+2024 PRL, arXiv:2403.03220, Supplementary A.4 **[V-text]**. "Both bias expansions are equivalent at fixed order in perturbation theory but differ in higher-order terms." With Lagrangian bias the displacement "contains only perturbative modes", so "a cut in the evolved density field … is unnecessary in the Lagrangian-bias forward model". Their fiducial run uses the Eulerian basis — [arXiv](https://arxiv.org/abs/2403.03220).
- Stadler+2025 Part I, arXiv:2409.10937, §3.5 and Appendices H.2/H.3 **[V-text]**. Lagrangian bias operators are built on a Lagrangian grid and moved with weighted displacements, with truncation and aliasing errors analysed separately (their Eqs. 31–34). Parameter recovery is tested with both Lagrangian and Eulerian bias. Higher-derivative operators are taken in the Eulerian frame, since the difference can be absorbed into the bias coefficients — [arXiv](https://arxiv.org/abs/2409.10937).
- Barreira, A., Lazeyras, T., Schmidt, F. (2021), "Galaxy bias from forward models: linear and second-order bias of IllustrisTNG galaxies", arXiv:2105.02876, JCAP 08 (2021) 029 **[V-INSPIRE]**. Field-level bias fitting with fixed ICs, a cheap diagnostic we can reuse on Quijote — [arXiv](https://arxiv.org/abs/2105.02876).
- Andrews+2026, arXiv:2603.20855 **[V-text]**. Bias evaluated in Lagrangian space. The Laplacian term and the redshift-space mapping are noted as applied in different frames, with "minimal impact at these large scales" — [arXiv](https://arxiv.org/abs/2603.20855).
- Memory note (project memory, not a source): Andrews+2026 use Eulerian bias including b∇, with p=1.3 and $\sigma_G^2\approx0.4\bar N$ on 32³ FoF halos above $3\times10^{13}$. This partly conflicts with the "Lagrangian space" wording extracted from the v1 HTML; check the paper's Eq. 6 directly.

### Inferences
- In a one-particle-per-cell Lagrangian-weight model, the weighted particles carry the bias operators to Eulerian space. Their CIC footprint differs from that of the unweighted matter particles, and from that of halos. The window and alias mismatch in Problem 1 therefore hits Lagrangian-bias models harder than Eulerian ones, where the operators are built on the already-gridded field. That is one reason the supersample-and-truncate fix mattered so much.
- LEFTfield's recommendation implies that a Λ cut on ε plus the Lagrangian basis removes the need for a separate filter on the evolved field.

### Gaps
- I did not find a direct field-level comparison of Lagrangian vs Eulerian bases *for f_NL* (as opposed to σ8 or f).
- I did not find a 2023–2026 field-level HEFT inference with sampled ICs on halos; INSPIRE searches returned only HEFT emulator/halo-model papers.

---

## 5. Model-mismatch and validation tests on N-body halos/galaxies

### Takeaway
The EFT/LEFTfield programme has the most systematic validation ladder:
1. data from the model itself;
2. data at a higher cutoff $\Lambda_0$;
3. sub-sampled matter;
4. halos with fixed ICs;
5. halos with sampled ICs;
6. blind HOD challenge.

On halos it gets unbiased σ8, f and BAO, with field-level gains over P+B of 1.2–5x depending on setup. Recent perturbative work argues that on large scales FLI ≈ P+B+T. The 3.5–5.2x gain in the PRL is therefore partly a noise-level effect (low-noise and nonlinear regime), and its origin is still debated. For f_NL specifically, the only N-body halo test is Andrews+2026, which finds about 1.3x over P+B at k_max≈0.1 with BORG.

### Cited Findings
- Kostić+2023, arXiv:2212.07875, JCAP 07 (2023) 063 **[V-INSPIRE, V-arXiv]**. Consistency and convergence tests: HMC on the ICs plus slice sampling for the parameters (our Andrews-style Gibbs scheme). Mis-specification is introduced through $\Lambda_0>\Lambda$, and recovery is unbiased. In the linear case they check the sampler against the analytic posterior. *Relevance:* a template for a "linear-limit posterior = analytic" sampler test, which pairs with our Tier-1 Fisher check — [arXiv](https://arxiv.org/abs/2212.07875).
- Nguyen+2024 PRL, arXiv:2403.03220, PRL 133, 221006 **[V-INSPIRE]**. On N-body halos, the field-level σ8 constraint is 3.5–5.2x tighter than P+B with the same model and modes (search summary of the abstract) — [arXiv](https://arxiv.org/abs/2403.03220).
- Beyond-2pt Collaboration (Krause+2025), arXiv:2405.02252, ApJ 990, 99 **[V-INSPIRE, V-text]**. Blind challenge including LEFTfield field-level inference on real-space boxes, with only σ8 free because LEFTfield "currently cannot perform a full inference over the ΛCDM cosmological parameter space" — [arXiv](https://arxiv.org/abs/2405.02252).
- Babić+2025 (BAO, JCAP 11 066) and Babić+2026 (BAO on halos, PRD 113 L061304). Listed in §1 — [arXiv](https://arxiv.org/abs/2505.13588).
- Stadler+2023/2025 Parts I and II. Listed in §1. Halo tests at z=0, 0.5, 1 over a range of Λ — [arXiv](https://arxiv.org/abs/2409.10937).
- Spezzati, F., Marinucci, M., Simonović, M. (2026), "Equivalence of the field-level inference and conventional analyses on large scales", arXiv:2507.05378, PRD 113, 063546 **[V-INSPIRE, V-arXiv]**. For halos in real space with amplitude A, joint P+B+T agrees with FLI. The trispectrum adds only 20–30% over P+B. FLI "does not get significant information from large displacements beyond … Eulerian kernels or … beyond the trispectrum". The paper disagrees with some SBI P+B results — [arXiv](https://arxiv.org/abs/2507.05378).
- Akitsu+2025, arXiv:2509.09673 **[V-arXiv, V-text]**. With nuisance parameters free, adding the bispectrum already recovers FLI in the perturbative regime. The wrong likelihood (Gaussian vs non-Gaussian noise) produces biases — [arXiv](https://arxiv.org/abs/2509.09673).
- Nikolac, I., Schmidt, F., Tucci, B. (2026), "Field-level vs summaries: convergence of information in non-Gaussian density fields", arXiv:2606.18227, preprint **[V-INSPIRE, V-arXiv]**. In a toy model with a quadratic coupling λ, FLI = P+B in the Gaussian limit. As λ grows and noise falls, summaries lose information that is "largely, but not completely" recovered up to the 6-point function. *Relevance:* explains why dense halo samples (low noise) are where FLI gains, and also where calibration problems appear — [arXiv](https://arxiv.org/abs/2606.18227).
- Andrews, A., Jasche, J., Lavaux, G., Schmidt, F. (2023), "Bayesian field-level inference of primordial non-Gaussianity using next-generation galaxy surveys", arXiv:2203.08838, MNRAS 520, 5746 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2203.08838).
- Andrews, A., et al. (Euclid Collaboration) (2026), "Euclid: Field-level inference of primordial non-Gaussianity and cosmic initial conditions", arXiv:2412.11945, A&A 711, A94 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2412.11945).
- Andrews+2026 Quijote-PNG, arXiv:2603.20855 **[V-INSPIRE, V-arXiv, V-text]**. The first joint f_NL + nuisance + IC inference on N-body halos. It reaches k_max≈0.1, gains about 1.3x over P+B, and tests the structure-formation model and resolution. Constraints tighten with resolution — [arXiv](https://arxiv.org/abs/2603.20855).
- Andrews, A., Loureiro, A., Jasche, J., McAlpine, S., et al. (2026), "Reconstructing the largest scales of the Universe with field-level inference applied to the Quaia Quasar Catalogue", arXiv:2602.02363, preprint **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2602.02363).
- Stopyra, S., Peiris, H. V., Pontzen, A., Jasche, J., Lavaux, G. (2024), "Towards accurate field-level inference of massive cosmic structures", arXiv:2304.09193, MNRAS 527, 1244 **[V-INSPIRE, V-arXiv]**. A 20-step COLA integrator is accurate for the large-scale density around clusters but not converged for virial masses. Introduces "posterior resimulations". *Relevance:* integrator and resolution requirements for BORG on halos — [arXiv](https://arxiv.org/abs/2304.09193).
- Doeser, L., Jamieson, D., Stopyra, S., Lavaux, G., Leclercq, F., Jasche, J. (2024), "Bayesian inference of initial conditions from non-linear cosmic structures using field-level emulators", arXiv:2312.09271, MNRAS 535, 1258 **[V-INSPIRE, V-arXiv]**. A V-net emulator correcting 1LPT inside BORG, with percent-level P and B up to Nyquist at 1.95 h⁻¹Mpc (Gaussian-noise mocks) — [arXiv](https://arxiv.org/abs/2312.09271).
- Nguyen+2021, arXiv:2011.06587 **[V-text]**. Tests gravity (LPT vs 2LPT), bias (linear, power-law, broken power-law) and likelihood (Poisson vs Gaussian) on GADGET-2 halos at several grid resolutions. The Poisson likelihood fails for halos. Its §4.2 is the closest BORG analogue to our Stage 3 test — [arXiv](https://arxiv.org/abs/2011.06587).
- Leclercq, F. (2025), "Field-level inference in cosmology" (lecture notes), arXiv:2509.13435, preprint **[V-INSPIRE, V-arXiv]**. A review entry point — [arXiv](https://arxiv.org/abs/2509.13435).
- Bayer, A. E., Seljak, U., Modi, C. (2023), "Field-Level Inference with Microcanonical Langevin Monte Carlo", arXiv:2307.09504, preprint in INSPIRE **[V-INSPIRE]**. The MCLMC candidate sampler from CLAUDE.md — [arXiv](https://arxiv.org/abs/2307.09504).
- Adjacent PNG references:
  - Chen, E., Green, D., Lee, V. S. H. (2026), "Primordial Non-Gaussianity and the Field-Level Cramer-Rao Bound", arXiv:2603.22415, preprint **[V-INSPIRE, V-arXiv]**. Multi-tracer scale-dependent bias can beat conservative higher-point analyses for local PNG but falls short of the optimum with halo-scale modes. Useful for our Stage 1 field-level Fisher — [arXiv](https://arxiv.org/abs/2603.22415).
  - Barreira, A. (2020), arXiv:2009.06622, JCAP 12 (2020) 031 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2009.06622).
  - Barreira, A. (2022), arXiv:2205.05673, JCAP 11 (2022) 013 **[V-INSPIRE]** — [arXiv](https://arxiv.org/abs/2205.05673).
  - Lazeyras, T., Barreira, A., Schmidt, F., Desjacques, V. (2023), "Assembly bias in the local PNG halo bias and its implication for f_NL constraints", arXiv:2209.07251, JCAP 01 (2023) 023 **[V-INSPIRE]**. Relevant to the halo $b_\phi$ (p) value for the mismatch test — [arXiv](https://arxiv.org/abs/2209.07251).
  - Yu, J. & Nguyen, N.-M. (2026), "How I stop worrying about non-universality and $b_\phi$: Constraining local f_NL with $b_\phi$ priors from HOD posteriors", arXiv:2607.01314, preprint **[V-INSPIRE, V-arXiv]**. $b_\phi$ priors from DESI EDR HOD posteriors; unbiased f_NL even with assembly bias. Relevant to the 40% $b_\phi$ prior choice — [arXiv](https://arxiv.org/abs/2607.01314).

### Inferences
- A validation ladder to copy, adapted from the LEFTfield and Kostić et al. sequence:
  1. Self-consistent mocks.
  2. Mocks at a finer grid or higher cutoff than the inference.
  3. Fixed-IC fits to halos to calibrate bias and noise.
  4. Sampled-IC fits to halos over many realizations.
  5. ZA vs 2LPT.
- Expect FLI/P+B gains for f_NL of order 1.3x on halos (Andrews+2026), not the 3.5–5x seen for σ8 (PRL). f_NL information is dominated by low-k scale-dependent bias, where FLI ≈ P (Spezzati+2026; Akitsu+2025).

### Gaps
- None of the field-level f_NL halo tests (Andrews+2026 only) use an EFT sharp-k likelihood. The EFT halo tests (σ8, f, BAO) do not include f_NL. No paper combines both.
- The brief's "Stadler et al. 2024 (cosmology from the field level with halos)" has no exact match; the closest are Stadler+2023 (arXiv:2303.09876) and Parts I and II (arXiv:2409.10937, 2411.04513).

---

## 6. Posterior calibration / coverage tests for field-level and simulation-based inference

### Takeaway
SBC (Talts+2018; refined by Modrák+2023 and Säilynoja+2022) and TARP (Lemos+2023) are the standard tools. Hermans+2022 show that overconfidence is the common failure of approximate posteriors. In field-level cosmology, calibration has so far been checked only lightly:
- Andrews+2026: truth inside 1σ in about 68% of 30 Quijote runs.
- Nguyen+2024: SBC for the SBI branch only.
- Kostić+2023: analytic linear-case posterior.

Alokda+2026 warn that average coverage can pass while per-realization widths and tails are wrong. Our 10–45% too-narrow posteriors on dense samples match the overconfidence expected from a mis-specified (Gaussian, fixed or low σ) noise model. They are most likely model mismatch, not a sampler bug. SBC on self-consistent mocks can separate the two.

### Cited Findings
- Cook, S. R., Gelman, A., Rubin, D. B. (2006), "Validation of Software for Bayesian Models Using Posterior Quantiles", J. Comput. Graph. Stat. 15, 675 (no arXiv) **[UNVERIFIED bibliographic details]**. The precursor of SBC.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., Gelman, A. (2018), "Validating Bayesian Inference Algorithms with Simulation-Based Calibration", arXiv:1804.06788, preprint **[V-INSPIRE, V-arXiv]**. Rank statistics of the true parameter among posterior draws are uniform if the computation is correct. A ∪-shaped histogram means overconfidence — [arXiv](https://arxiv.org/abs/1804.06788).
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., et al. (2022/2023), "Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes Sensitivity", arXiv:2211.02383 **[V-arXiv; journal (Bayesian Analysis) UNVERIFIED]**. Recommends data-dependent test quantities, especially the joint log-likelihood, and distinguishes SBC from data-averaged-posterior checks. *Relevance:* use the log-likelihood (χ²) as an SBC quantity; we already record χ² — [arXiv](https://arxiv.org/abs/2211.02383).
- Säilynoja, T., Bürkner, P.-C., Vehtari, A. (2021/2022), "Graphical Test for Discrete Uniformity and its Applications in Goodness of Fit Evaluation and Multiple Sample Comparison", arXiv:2103.10522 **[V-arXiv; journal (Statistics and Computing) UNVERIFIED]**. ECDF simultaneous confidence bands for SBC rank histograms, suited to the small numbers of realizations we can afford — [arXiv](https://arxiv.org/abs/2103.10522).
- Lemos, P., Coogan, A., Hezaveh, Y., Perreault-Levasseur, L. (2023), "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference" (TARP), arXiv:2302.03026 **[V-arXiv; venue (ICML 2023) UNVERIFIED]**. A coverage test that is necessary and sufficient for posterior accuracy, works in high dimensions and needs only samples — [arXiv](https://arxiv.org/abs/2302.03026).
- Hermans, J., Delaunoy, A., Rozet, F., Wehenkel, A., Begy, V., Louppe, G. (2022), "A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful", arXiv:2110.06581 **[V-INSPIRE, V-arXiv; venue (TMLR) UNVERIFIED]**. All benchmarked SBI algorithms can be overconfident, and ensembling helps — [arXiv](https://arxiv.org/abs/2110.06581).
- Alokda, T., Porciani, C., Eggemeier, A. (2026), "Coverage is not enough: Frequentist tests of simulation-based inference for primordial non-Gaussianity", arXiv:2605.00980, A&A 714, A66 **[V-INSPIRE, V-arXiv]**.
  - Prior-predictive coverage "probe[s] calibration only in an averaged sense and do[es] not constrain posterior behavior at fixed parameter value".
  - On 1000 halo-field realizations, the SBI and likelihood-based variances for f_NL agree on average but with weaker per-realization consistency, and the tails differ.
  - *Relevance:* our Quijote ensembles test at fixed f_NL (0, ±100), which is exactly the fixed-parameter frequentist check they advocate.
  - [arXiv](https://arxiv.org/abs/2605.00980)
- Nguyen+2024 PRL, arXiv:2403.03220, Supplementary B.4 **[V-text]**. SBC was applied to the SBI P+B branch, with uniform ranks. The field-level branch was validated through convergence diagnostics, not SBC — [arXiv](https://arxiv.org/abs/2403.03220).
- Andrews+2026, arXiv:2603.20855 **[V-text]**. Over 30 runs, ground truth lies within 1σ in "approximately 68% of the cases", with σ fixed at the upper range of the fixed-IC posterior for robustness — [arXiv](https://arxiv.org/abs/2603.20855).
- Hahn, C., Eickenberg, M., Ho, S., Hou, J., et al. (2023), SimBIG, arXiv:2211.00723, PNAS 120, e2218810120 **[V-INSPIRE]**. Example of coverage validation in a cosmology forward-modelling analysis — [arXiv](https://arxiv.org/abs/2211.00723).
- Akitsu+2025 and Rubira & Schmidt 2026 (above) show that a Gaussian-likelihood fit to non-Gaussian noise biases parameters and (through σ collapse) shrinks posteriors — [arXiv](https://arxiv.org/abs/2509.09673); [arXiv](https://arxiv.org/abs/2511.05484).

### Inferences
- Diagnosing the 10–45% too-narrow widths:
  1. Run SBC or a TARP-style coverage test on **self-consistent mocks** (same forward model, same gridding). If coverage is fine there, the sampler, Fisher and mass matrix are fine, and the halo undercoverage is model mismatch.
  2. Run the fixed-parameter frequentist test (Alokda+2026) on the Quijote ensemble: the ratio of the scatter of posterior means to the mean posterior σ directly measures the width deficit.
  3. Vary the noise model (fixed σ from a fixed-IC fit, σ inflated to the upper posterior range as in Andrews+2026, a $k^2$-dependent noise, density-dependent variance). Check which one restores calibration.
- The narrowing growing with sample density is consistent with:
  - Nguyen+2021's observation that collapse occurs when the Poisson variance per cell is small;
  - Nikolac+2026's result that low-noise data are where the field carries the most non-summary information and hence where model error matters most.

### Gaps
- No published SBC or rank-statistic test of a *field-level* sampler (as opposed to SBI) on N-body halos was found. Field-level calibration evidence is limited to the roughly 68%-in-1σ count of Andrews+2026 and Kostić+2023's linear-case analytic check.
- No paper reports width miscalibration of field-level posteriors as a function of tracer number density; our 10–45% result appears new.
