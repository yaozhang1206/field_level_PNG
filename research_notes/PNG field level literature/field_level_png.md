# Field-level inference of primordial non-Gaussianity (local $f_{\rm NL}$) from large-scale structure: categorised bibliography

How IDs were checked: every arXiv ID below was looked up on 2026-10-13 against the arXiv API (export.arxiv.org/api/query?id_list=...), which returned the authors, title, date and any journal_ref quoted here. "Journal: —" means arXiv had no journal_ref; the paper may still be published. Two journal references come from somewhere other than arXiv, and they are marked where they appear. I did not cite any arXiv ID that I could not find.

---

## Q1. Which papers do Bayesian forward-modelling / field-level inference of $f_{\rm NL}$ (BORG and others)?

### Takeaway
Bayesian field-level inference of local $f_{\rm NL}$ with an explicit forward model has been done mostly by Andrews, Jasche, Lavaux and collaborators with BORG. There are four BORG $f_{\rm NL}$ papers: SDSS-III-like mocks (2022), Euclid mocks (2024/2026), Quijote-PNG N-body halos (2026), plus a Quaia quasar reconstruction (2026) that has no $f_{\rm NL}$ and is relevant as a QSO/light-cone template. The only field-level $f_{\rm NL}$ test on real N-body halos (Quijote-PNG, $k_{\max}\approx0.1$) finds roughly a 1.3× gain over a P+B Fisher. I found no field-level $f_{\rm NL}$ analysis of real survey data.

### Cited Findings (BORG / forward-modelling $f_{\rm NL}$)
- **Andrews, Jasche, Lavaux, Schmidt (2022/2023)**, "Bayesian field-level inference of primordial non-Gaussianity using next-generation galaxy surveys", arXiv:2203.08838. Journal: MNRAS 520, 5746 (2023) per the task brief. arXiv has no journal_ref and my web search could not confirm the volume and page, so treat it as unverified. First BORG $f_{\rm NL}$ paper: the full 3D survey is forward-modelled with PNG, survey geometry, noise and unknown biases, and $f_{\rm NL}$ comes out unbiased. Gets $\sigma(f_{\rm NL})\approx8.78$ on SDSS-III/BOSS-like mocks, which the authors call a ~2.5× improvement over published constraints at the time. Stage-IV mocks show further gains at higher resolution — [arXiv:2203.08838](https://arxiv.org/abs/2203.08838)
- **Andrews, Jasche, Lavaux, Leclercq, Finelli, et al. (Euclid Collaboration) (2024; publ. 2026)**, "Euclid: Field-level inference of primordial non-Gaussianity and cosmic initial conditions", arXiv:2412.11945. Journal: A&A 711, A94 (2026), DOI 10.1051/0004-6361/202553802. This reference comes from the UWC repository and IPAC/ADS, not from arXiv. Euclid-like mock MCMC with full-field forward modelling (2LPT, 8 Gpc/h box). The arXiv v1 abstract gives $\sigma(f_{\rm NL})=2.3$ at $\Delta L=62.5\,h^{-1}$Mpc. **The published version's abstract instead gives $\sigma(f_{\rm NL})=2.6$ at $\Delta L=31.25\,h^{-1}$Mpc with cutoff $k_{\rm NF}=0.1\,h$/Mpc.** The headline number and setup changed between v1 and the published paper, so check which one the project brief quotes. The paper also releases N-body sims with non-zero $f_{\rm NL}$ and maps of adiabatic curvature — [arXiv:2412.11945](https://arxiv.org/abs/2412.11945); [UWC repository record (published abstract)](https://uwcscholar.uwc.ac.za/items/d874fd68-c38e-41ed-a2df-bba46f645ea1); [IPAC/ADS 2026A&A...711A..94A](https://www.ipac.caltech.edu/publication/2026A&A...711A..94A)
- **Andrews, Jasche, Lavaux, Coulton, Villaescusa-Navarro, et al. (2026)**, "Field-Level Inference of Primordial Non-Gaussianity with the Quijote Simulation Suite", arXiv:2603.20855. Journal: —. First joint field-level inference of $f_{\rm NL}$, bias, noise and initial conditions on N-body halo catalogues. Setup: Quijote-PNG at 1 (Gpc/h)³, z=0.5, FoF halos $M>3.16\times10^{13}\,h^{-1}M_\odot$, redshift space, 30 sims with $f_{\rm NL}=0,\pm100$. Model: 2LPT (1LPT and tCOLA also tested), bias $b_1,b_2,b_K,b_\nabla,b_\phi,b_{\phi\delta}$ with $p=1.3$ and $b_{\phi\delta}$ from UMF, Gaussian likelihood with fixed $\sigma_G^2=1.2$, $32^3$ grid (31.25 Mpc/h, $k_{\max}\approx0.1$). Results: $\sigma(f_{\rm NL})\approx23$ on the reference sim. Fisher gives P alone 43 and P + tree-level B 27–29, so the abstract's headline is "~1.3× improvement over P+B". 1LPT, 2LPT and tCOLA differ only marginally. $f_{\rm NL}$ ESS is 1175–2731 per chain. Truth falls within 1σ in ~68% of runs — [arXiv:2603.20855](https://arxiv.org/abs/2603.20855) (details from [HTML full text](https://arxiv.org/html/2603.20855))
- **Andrews, Loureiro, Jasche, McAlpine, Lavaux, et al. (2026)**, "Reconstructing the largest scales of the Universe with field-level inference applied to the Quaia Quasar Catalogue", arXiv:2602.02363. Journal: —. BORG with an LPT forward model, light cone, RSD, quasar bias and selection, applied to Quaia (G<20.0 and G<20.5) in a $(10\,h^{-1}{\rm Gpc})^3$ box at 39.1 Mpc/h resolution. The posterior density cross-correlates with Planck CMB lensing at ~4σ. The abstract reports no $f_{\rm NL}$ constraint; the paper is a framework and template for field-level work on quasar surveys — [arXiv:2602.02363](https://arxiv.org/abs/2602.02363)
- **Euclid Collaboration: Finelli, Akrami, Andrews, Ballardini, et al. (2025)**, "Euclid preparation: Expected constraints on initial conditions", arXiv:2507.15819. Journal: —. A Fisher review of Euclid PNG reach with P+B for local, equilateral and orthogonal shapes, plus a section reviewing Bayesian field-level inference of local $f_{\rm NL}$ — [arXiv:2507.15819](https://arxiv.org/abs/2507.15819)
- **Seljak, Aslanyan, Feng, Modi (2017)**, "Towards optimal extraction of cosmological information from nonlinear data", arXiv:1706.06645. Journal: —. A forerunner: it expands the likelihood around the MAP of the linear modes found by optimisation through 2LPT/N-body, and explicitly generalises to forward-model parameters including PNG. It gives a near-optimal initial-power-spectrum estimator — [arXiv:1706.06645](https://arxiv.org/abs/1706.06645)

### Inferences
- The BORG line on real N-body halos (2603.20855) is much less optimistic than the self-consistent Euclid mocks: about 1.3× over P+B, versus the large gains implied when mocks are generated with the inference model. A DESI field-level forecast should be calibrated against the mismatch test, not the self-consistent one.
- The Euclid headline (2.3 vs 2.6, and 62.5 vs 31.25 Mpc/h) differs between arXiv v1 and the A&A version. Quote the published one, or quote both and give the setup.

### Gaps
- No paper found applying field-level inference to measure $f_{\rm NL}$ from **real** galaxy or quasar data, whether BOSS, eBOSS, DESI or Quaia.
- I could not confirm the MNRAS 520, 5746 reference for 2203.08838 through arXiv or web search.
- No independent (non-BORG) Bayesian field-level $f_{\rm NL}$ analysis on N-body halos turned up, e.g. with LEFTfield, JaxPM or pmwd.

---

## Q2. EFT-based field-level likelihoods (LEFTfield etc.) and the "is field-level information beyond P+B?" debate

### Takeaway
The EFT-likelihood/LEFTfield programme (Schmidt, Cabass, Nguyen, Stadler, Babić, Tucci, Kostić, Rubira, Reinecke) built the theory and code for field-level inference with perturbative forward models. It has been applied to $\sigma_8$, $f$ and BAO, not yet to $f_{\rm NL}$. Its $f_{\rm NL}$-related pieces are formal extensions: Cabass & Schmidt 2019 comment on PNG, and Ke, Wang & He 2025 compute the EFT likelihood with PNG.

The information-content debate is still open:
- Nguyen et al. 2024 find field-level $\sigma_8$ 3.5–5.2× tighter than SBI P+B.
- Cabass–Simonović–Zaldarriaga 2023, Spezzati–Marinucci–Simonović 2025 and Akitsu et al. 2025 argue that on perturbative scales field-level ≈ P+B(+T).
- Schmidt 2025 and Pietroni & Schmidt 2026 show formally that an $m$-th order model captures $n\le m+1$-point information.
- Nikolac, Schmidt & Tucci 2026 and Stadler 2026 find real gains in specific regimes: low noise, and extended cosmologies in redshift space.

### Cited Findings — EFT likelihood foundations
- **Schmidt, Elsner, Jasche, Nguyen, Lavaux (2018)**, "A rigorous EFT-based forward model for large-scale structure", arXiv:1808.02002, JCAP 01 (2019) 042. Field-level likelihood under perturbative control. Its information equals NLO galaxy P + LO bispectrum + BAO reconstruction. Gives unbiased $\sigma_8$ from halos — [arXiv:1808.02002](https://arxiv.org/abs/1808.02002)
- **Elsner, Schmidt, Jasche, Lavaux, Nguyen (2019)**, "Cosmology Inference from Biased Tracers using the EFT-based Likelihood", arXiv:1906.07143, JCAP 01 (2020) 029. First application to real-space halo catalogs — [arXiv:1906.07143](https://arxiv.org/abs/1906.07143)
- **Cabass & Schmidt (2019)**, "The EFT Likelihood for Large-Scale Structure", arXiv:1909.04022. Journal: —. Derives the conditional tracer likelihood with functional methods, including non-Gaussian stochasticity and higher-derivative terms, and "comment[s] on possible extensions … with a particular eye towards primordial non-Gaussianity" — [arXiv:1909.04022](https://arxiv.org/abs/1909.04022)
- **Schmidt, Cabass, Jasche, Lavaux (2020)**, "Unbiased Cosmology Inference from Biased Tracers using the EFT Likelihood", arXiv:2004.06707, JCAP 11 (2020) 008. Adds the initial-condition cutoff Λ. $\sigma_8$ is recovered within ≲6% (2nd-order bias) and ≤4% (3rd-order) at Λ=0.1 h/Mpc — [arXiv:2004.06707](https://arxiv.org/abs/2004.06707)
- **Cabass (2020)**, "The EFT Likelihood for Large-Scale Structure in Redshift Space", arXiv:2007.14988 — [arXiv:2007.14988](https://arxiv.org/abs/2007.14988)
- **Schmidt (2020)**, "Sigma-Eight at the Percent Level: The EFT Likelihood in Real Space", arXiv:2009.14176, JCAP 04 (2021) 032. Real-space formulation that can include a survey window; unbiased $\sigma_8$ at 2% for Λ≤0.14 — [arXiv:2009.14176](https://arxiv.org/abs/2009.14176)
- **Nguyen, Schmidt, Lavaux, Jasche (2020)**, "Impacts of the physical data model on the forward inference of initial conditions from biased tracers", arXiv:2011.06587, JCAP 03 (2021) 058. Phase recovery responds only weakly to the model. **The amplitude bias of the inferred ICs depends strongly on the bias model and the likelihood** — [arXiv:2011.06587](https://arxiv.org/abs/2011.06587)
- **Kostić, Nguyen, Schmidt, Reinecke (2022)**, "Consistency tests of field level inference with the EFT likelihood", arXiv:2212.07875, JCAP 07 (2023) 063. HMC on ICs plus slice sampling on parameters, 2LPT. Joint inference stays unbiased under cutoff mismatch $\Lambda_0>\Lambda$ — [arXiv:2212.07875](https://arxiv.org/abs/2212.07875)
- **Stadler, Schmidt, Reinecke (2023)**, "Cosmology inference at the field level from biased tracers in redshift-space", arXiv:2303.09876, JCAP 10 (2023) 069. LEFTfield in redshift space. A higher-order velocity-bias expansion is required. Unbiased $f$ at a few % for $0.08\le\Lambda\le0.20$ — [arXiv:2303.09876](https://arxiv.org/abs/2303.09876)
- **Stadler, Schmidt, Reinecke (2024)**, "Fast, Accurate and Perturbative Forward Modeling of Galaxy Clustering Part I: Galaxies in the Restframe", arXiv:2409.10937; and **Stadler, Schmidt, Reinecke, Esposito (2024)**, "… Part II: Redshift Space", arXiv:2411.04513. LEFTfield forward-model papers — [arXiv:2409.10937](https://arxiv.org/abs/2409.10937); [arXiv:2411.04513](https://arxiv.org/abs/2411.04513)
- **Babić, Schmidt, Tucci (2022)**, "BAO scale inference from biased tracers using the EFT likelihood", arXiv:2203.06177, JCAP 08 (2022) 007 — [arXiv:2203.06177](https://arxiv.org/abs/2203.06177)
- **Babić, Schmidt, Tucci (2024)**, "Straightening the Ruler: Field-Level Inference of the BAO Scale with LEFTfield", arXiv:2407.01524, JCAP 11 (2025) 066. BAO systematic bias <2%, and BAO error 30–50% smaller than standard post-reconstruction P — [arXiv:2407.01524](https://arxiv.org/abs/2407.01524)
- **Babić, Schmidt, Tucci (2025)**, "Forward vs Backward: Improving BAO Constraints with Field-Level Inference", arXiv:2505.13588, PRD 113, L061304 (2026) — [arXiv:2505.13588](https://arxiv.org/abs/2505.13588)
- **Tucci & Schmidt (2023)**, "EFTofLSS meets simulation-based inference: σ8 from biased tracers", arXiv:2310.03741, JCAP 05 (2024) 063. SBI on P+B with the same LEFTfield forward model; this is the P+B baseline used by Nguyen et al. 2024 — [arXiv:2310.03741](https://arxiv.org/abs/2310.03741)
- **Rubira & Schmidt (2025)**, "Non-Gaussian Galaxy Stochasticity and the Noise-Field Formulation", arXiv:2511.05484, JCAP 08 (2026) 023. Noise-field formulation of the EFT partition function. First LEFTfield FLI with non-Gaussian, density-dependent stochasticity on halos — [arXiv:2511.05484](https://arxiv.org/abs/2511.05484)
- **Ke, Wang, He (2025)**, "Calculating the EFT likelihood via saddle-point expansion", arXiv:2502.16082, JCAP 04 (2025) 064; and **"Precision calculation of the EFT likelihood with primordial non-Gaussianities"**, arXiv:2505.10915, JCAP 09 (2025) 029. With leading-order PNG, higher-order noise and saddle-point corrections give *irreducible field-dependent contributions* to the conditional likelihood, so extra effective terms are needed in forward models with non-Gaussian ICs. This is the only EFT-likelihood paper found that targets PNG directly — [arXiv:2502.16082](https://arxiv.org/abs/2502.16082); [arXiv:2505.10915](https://arxiv.org/abs/2505.10915)
- **Schmittfull, Simonović, Assassi, Zaldarriaga (2018)**, "Modeling Biased Tracers at the Field Level", arXiv:1811.10640, PRD 100, 043514 (2019). Field-level test of perturbative bias with shifted operators. Model error ≈ Poisson. Mass-weighting cuts the error by up to 10×. Linear bias gives errors several times Poisson on large scales — [arXiv:1811.10640](https://arxiv.org/abs/1811.10640)
- **Schmittfull, Simonović, Ivanov, Philcox, Zaldarriaga (2020)**, "Modeling Galaxies in Redshift Space at the Field Level", arXiv:2012.03334. Residual noise is white on large scales, with a $k^2\mu^2$ correction — [arXiv:2012.03334](https://arxiv.org/abs/2012.03334)

### Cited Findings — "information beyond P+B?" debate
- **Cabass, Simonović, Zaldarriaga (2023)**, "Cosmological Information in Perturbative Forward Modeling", arXiv:2307.04706. Journal: — (arXiv). I take this to be the "Cabass et al. 2024" in the brief; arXiv lists it as 2023. In the small-noise limit the perturbative field-level posterior holds the same information as $n$-point functions at the same order. Long–short couplings can degrade standard analyses but not FLI — [arXiv:2307.04706](https://arxiv.org/abs/2307.04706)
- **Nguyen, Schmidt, Tucci, Reinecke, Kostić (2024)**, "How much information can be extracted from galaxy clustering at the field level?", arXiv:2403.03220, PRL 133, 221006 (2024). LEFTfield FLI on halos in 8 (Gpc/h)³ versus SBI P+B with the same model and modes. $\sigma_8$ improves by **3.5× (20.0%→5.7%) at $k_{\max}=0.10$ and 5.2× (17.0%→3.3%) at 0.12** — [arXiv:2403.03220](https://arxiv.org/abs/2403.03220)
- **Schmidt (2025)**, "On the Connection between Field-Level Inference and n-point Correlation Functions", arXiv:2504.15351, JCAP 09 (2025) 056. Zero-noise, IC-marginalised EFT posterior: an $m$-th-order forward model captures $n\le m+1$-point information. Also shows how to estimate the bias from an incomplete model — [arXiv:2504.15351](https://arxiv.org/abs/2504.15351)
- **Spezzati, Marinucci, Simonović (2025)**, "Equivalence of the field-level inference and conventional analyses on large scales", arXiv:2507.05378. Real-space halos with amplitude $A$ as the only cosmological parameter. P+B+trispectrum matches recent FLI, and the trispectrum adds only 20–30% over P+B. Disagrees with some SBI P+B analyses, implicitly including Nguyen et al. — [arXiv:2507.05378](https://arxiv.org/abs/2507.05378)
- **Akitsu, Simonović, Chen, Cabass, Zaldarriaga (2025)**, "Cosmology inference with perturbative forward modeling at the field level: a comparison with joint power spectrum and bispectrum analyses", arXiv:2509.09673. Perturbative mocks with Gaussian noise, $\{A,\omega_{\rm cdm},H_0\}$, real and redshift space. FLI errors are **comparable to P+B**. A Gaussian field-level likelihood fitted to non-Gaussian-noise data gives **significant parameter biases** — [arXiv:2509.09673](https://arxiv.org/abs/2509.09673)
- **Pietroni & Schmidt (2026)**, "On the Relation Between Field-Level Posteriors, Correlators, and their Likelihoods", arXiv:2604.25385, JCAP 09 (2026) 011. Field-level posterior marginalised over ICs and noise. Derives a general Fisher matrix and splits it into connected-correlator contributions; the Gaussian bispectrum likelihood reproduces the corresponding FLP term — [arXiv:2604.25385](https://arxiv.org/abs/2604.25385)
- **Nikolac, Schmidt, Tucci (2026)**, "Field-level vs summaries: convergence of information in non-Gaussian density fields", arXiv:2606.18227. Toy model: linear field, local quadratic coupling λ, Gaussian noise. P+B, operator correlators and FLI agree in the Gaussian limit. As λ grows, summaries lose information. Operator correlators up to the 6-point function recover most but not all of it, and the loss is worse at low noise — [arXiv:2606.18227](https://arxiv.org/abs/2606.18227)
- **Stadler (2026)**, "Cosmological information from field-level analysis in redshift space", arXiv:2609.16719. LEFTfield mocks resembling **DESI LRGs**. FLI measures the primordial amplitude ~2× more precisely than P+B in ΛCDM, and up to 3× in extended cosmologies — [arXiv:2609.16719](https://arxiv.org/abs/2609.16719)

### Inferences
- No LEFTfield/EFT-likelihood $f_{\rm NL}$ inference exists yet. A project on DISCO-DJ/BORG would be the first independent cross-check, and EFT results (cutoff Λ, $b_\nabla$, noise-field stochasticity) show which bias and noise terms the $f_{\rm NL}$ model needs.
- Akitsu et al. 2025 and Rubira & Schmidt 2025 both point to the likelihood/noise model, not the gravity model, as the main bias risk. This fits the Stage 3 mismatch focus in the project brief.
- For local $f_{\rm NL}$ the dominant signal is scale-dependent bias at low $k$, where all camps agree FLI ≈ P (+B). The disputed FLI gains are in nonlinear-information parameters such as $\sigma_8$. So expect only a modest FLI advantage for $f_{\rm NL}$, consistent with the ~1.3× of Andrews et al. 2026 on Quijote-PNG.

### Gaps
- No EFT-likelihood paper performing $f_{\rm NL}$ inference on mocks or data was found.
- I did not reconcile the Nguyen et al. 2024 (3.5–5.2×) vs Spezzati et al. 2025 / Akitsu et al. 2025 (≈1×) disagreement. The setups differ: halos vs perturbative mocks, and SBI P+B covariance vs analytic.

---

## Q3. Field-level Fisher forecasts / information-content studies for $f_{\rm NL}$

### Takeaway
There are three directly relevant analytic or field-level information studies for local $f_{\rm NL}$:
- Sullivan & Chen 2024: field-level $b_\phi$/$b_{\phi\delta}$ fits at fixed ICs. Marginalising $b_\phi f_{\rm NL}$ degrades $f_{\rm NL}$ by 2–3×, and $f_{\rm NL}$ comes out biased on halos without a $b_\phi$ prior.
- Chen, Green & Lee 2026: field-level Cramér–Rao bound. Multi-tracer scale-dependent bias beats conservative higher-point analyses but falls short of the optimum.
- Baumann & Green 2021: map-level locality argument.

Related analytic work on reconstruction information includes McQuinn 2020 and Darwish et al. 2020.

### Cited Findings
- **Sullivan & Chen (2024)**, "Local Primordial Non-Gaussian Bias at the Field Level", arXiv:2410.18039. Journal: —. First field-level measurement of $b_\phi$ and $b_{\phi\delta}$, using Lagrangian bias with nonlinear N-body displacements. Results are qualitatively consistent with UMF and separate-universe predictions but disagree in detail. At fixed ICs: **marginalising over $b_\phi f_{\rm NL}$ gives no precise $f_{\rm NL}$, with a 2–3× degradation from linear to quadratic bias on perturbative mocks**. On halos, $f_{\rm NL}$ is **biased** when $b_\phi$ and $b_{\phi\delta}$ are marginalised, so a $b_\phi$ prior is needed — [arXiv:2410.18039](https://arxiv.org/abs/2410.18039)
- **Chen, Green, Lee (2026)**, "Primordial Non-Gaussianity and the Field-Level Cramer-Rao Bound", arXiv:2603.22415. Field-level Cramér–Rao bound for PNG. For local shape, **multi-tracer scale-dependent bias can exceed conservative higher-point analyses but falls short of the optimum** once all modes down to halo scales are included. Equilateral is very prior-sensitive — [arXiv:2603.22415](https://arxiv.org/abs/2603.22415)
- **Baumann & Green (2021)**, "The Power of Locality: Primordial Non-Gaussianity at the Map Level", arXiv:2112.14645. Primordial information lives in long-range (nonlocal) correlations that local nonlinear evolution cannot create or destroy. A position-space map-level analysis breaks the equilateral–nonlinear degeneracy, which motivates SBI/forward modelling — [arXiv:2112.14645](https://arxiv.org/abs/2112.14645)
- **McQuinn (2020)**, "On the primordial information available to galaxy redshift surveys", arXiv:2008.12312. This is the "McQuinn 2021" in the brief; arXiv date 2020-08-27, no journal_ref. Toy model of Zel'dovich-displaced Lagrangian peaks with a linearised, convex reconstruction bound. Reconstruction fidelity is set by the **number of galaxies (constraints), not where shot noise exceeds signal**. Linear modes can be reconstructed below the shot-noise expectation if Lagrangian displacements are constrained — [arXiv:2008.12312](https://arxiv.org/abs/2008.12312)
- **Hamaus, Seljak, Desjacques (2011)**, "Optimal Constraints on Local Primordial Non-Gaussianity from the Two-Point Statistics of Large-Scale Structure", arXiv:1104.2321, PRD 84, 083509. Classic optimal-weighting and multi-tracer sample-variance-cancellation baseline for $f_{\rm NL}$ from two-point statistics — [arXiv:1104.2321](https://arxiv.org/abs/1104.2321)
- **Coulton, Villaescusa-Navarro, Jamieson, Baldi, Jung, et al. (2022)**, "Quijote PNG: The information content of the halo power spectrum and bispectrum", arXiv:2206.15450. Redshift-space halos with $M>3.2\times10^{13}$, $k_{\max}=0.5$. For local PNG, **scale-dependent bias in P with sample-variance cancellation beats the halo bispectrum**, and small scales add little — [arXiv:2206.15450](https://arxiv.org/abs/2206.15450)
- **Coulton et al. (2022)**, "Quijote-PNG: Simulations of primordial non-Gaussianity and the information content of the matter field power spectrum and bispectrum", arXiv:2206.01619. Releases Quijote-PNG. Matter-field P+B information saturates around $k\approx0.3$ — [arXiv:2206.01619](https://arxiv.org/abs/2206.01619)
- **Jung, Ravenni, Baldi, Coulton, Jamieson, et al. (2023)**, "Quijote-PNG: The Information Content of the Halo Mass Function", arXiv:2305.10597, ApJ 957, 50 (2023). Motivated by a preliminary **GNN field-level** analysis. Adding the HMF breaks the $f_{\rm NL}$–$b_\phi$ degeneracy without external priors. $\Delta f_{\rm NL}^{\rm loc}=40$ for 1 (Gpc/h)³, z=1, $\bar n\sim5.1\times10^{-5}$, $k_{\max}=0.5$ — [arXiv:2305.10597](https://arxiv.org/abs/2305.10597)
- **Hadzhiyska & Ferraro (2025)**, "Refining local-type primordial non-Gaussianity: Sharpened $b_\phi$ constraints through bias expansion", arXiv:2501.14873. Field-level HEFT on AbacusSummit DESI-like LRG and QSO samples. Regressing $b_\phi$ on Lagrangian biases (especially $b_\nabla$ as a concentration proxy) cuts the $b_\phi$ uncertainty by >70% (halos) and 80% (galaxies). Directly useful for a $b_\phi$ prior — [arXiv:2501.14873](https://arxiv.org/abs/2501.14873)
- **Barreira (2022)**, "Can we actually constrain $f_{\rm NL}$ using the scale-dependent bias effect?", arXiv:2205.05673, JCAP 11 (2022) 013. On BOSS DR12, $\sigma(f_{\rm NL})$ changes by up to an order of magnitude depending on the assumed $b_\phi(b_1)$. Strongest bound $f_{\rm NL}=16\pm16$ — [arXiv:2205.05673](https://arxiv.org/abs/2205.05673)
- **Sharma, Sullivan, Akitsu, Ivanov (2025)**, "Equilateral non-Gaussian Bias at the Field Level", arXiv:2511.14677. Equilateral-shape analogue: field-level EFT measurement of $b_\psi$ via noise cancellation, with a $b_\psi(b_1)$ fitting formula — [arXiv:2511.14677](https://arxiv.org/abs/2511.14677)
- **Chaussidon, Yèche, de Mattia, et al. (2024)**, "Constraining primordial non-Gaussianity with DESI 2024 LRG and QSO samples", arXiv:2411.17623. Journal: —. The P(k) baseline the project compares against. Note that the arXiv abstract says **1,631,716 LRGs (0.6<z<1.1)**, whereas the project brief lists 2,130,621 LRGs at 0.4<z<1.1; check which LRG sample was used — [arXiv:2411.17623](https://arxiv.org/abs/2411.17623)
- **Nakano, Takada, Kurita, Terasawa (2026)**, "Modeling survey-window and integral-constraint effects on PNG in the galaxy power spectrum with light-cone mocks", arXiv:2606.29735. Adjacent: window and integral-constraint modelling for PNG with light-cone mocks — [arXiv:2606.29735](https://arxiv.org/abs/2606.29735)

### Inferences
- Sullivan & Chen 2024 and Barreira 2022 together imply that field-level information does **not** remove the $b_\phi f_{\rm NL}$ degeneracy. The project's plan to run both "$b_\phi$ fixed (UMF)" and "40% prior" is consistent with the literature. Hadzhiyska & Ferraro 2025 give a DESI-tracer-specific route to a tighter prior.
- Sullivan & Chen's 2–3× linear→quadratic degradation is the closest published analogue to the brief's Part-B Schur/Laplace Fisher with $b_{\phi\delta}$ free. It is a useful sanity range for the Stage-1 field-level Fisher.
- The LRG sample size in the brief (2.13M, 0.4<z<1.1) differs from the arXiv abstract of 2411.17623 (1.63M, 0.6<z<1.1). This could change the Stage-1 calibration and should be checked.

### Gaps
- No paper found that does a Laplace/Schur-complement field-level Fisher for $f_{\rm NL}$ with a 2LPT forward model, i.e. the method the project brief proposes. The closest are Sullivan & Chen (fixed-IC field fits), Chen–Green–Lee (Cramér–Rao) and Pietroni & Schmidt (FLP Fisher expansion).
- I did not extract specific $\sigma(f_{\rm NL})$ numbers from Chen–Green–Lee or Sullivan–Chen; the abstracts give only ratios and qualitative results.

---

## Q4. Simulation-based / neural / implicit-likelihood field-level $f_{\rm NL}$ inference

### Takeaway
Neural "field-level" $f_{\rm NL}$ work is mostly hybrid. Networks estimate local small-scale statistics (local $\sigma_8$/power, halo properties), or patch summaries, which are then correlated with the large-scale field or combined with P+B. Reported gains:
- Giri et al. 2022: 3.5× over halo-based approach (matter field).
- Kvasiuk et al. 2024: "factor of a few" when adding concentration, but no gain over mass binning on AbacusPNG with mass only.
- Cagliari et al. 2025: 30–45% at $k_{\max}\sim0.1$ over P+B.
- Reconstruction plus P+B: Flöss & Meerburg 2023 report 3.65× on matter, Bottema et al. 2025 report 1.33× on halos.

No end-to-end CNN/GNN $f_{\rm NL}$ posterior on halo fields was found as a standalone paper.

### Cited Findings
- **Giri, Münchmeyer, Smith (2022)**, "Robust Neural Network-Enhanced Estimation of Local Primordial Non-Gaussianity", arXiv:2205.12964. NN estimates local $\sigma_8$, correlated with the large-scale field. **$\sigma(f_{\rm NL})$ 3.5× better than the standard halo approach** on N-body. Robust like scale-dependent bias: baryons can change normalisation, not detection — [arXiv:2205.12964](https://arxiv.org/abs/2205.12964)
- **Giri, Münchmeyer, Smith (2023)**, "Constraining $f_{NL}$ using the Large-Scale Modulation of Small-Scale Statistics", arXiv:2305.03070, PRD 112, 123544 (2025). Generalises to local power spectra and combines the squeezed bispectrum with the collapsed trispectrum. Strong for matter; for Quijote halos the gain is only marginal — [arXiv:2305.03070](https://arxiv.org/abs/2305.03070)
- **Kvasiuk, Münchmeyer, Smith (2024)**, "A Tale of Two Fields: Neural Network-Enhanced non-Gaussianity Search with Halos", arXiv:2410.01007. Any N halo fields compress losslessly to two fields. On AbacusSummit/AbacusPNG, mass-binning is already optimal. **Adding concentration improves $f_{\rm NL}$ by "a factor of a few"**. ML does not lower shot noise — [arXiv:2410.01007](https://arxiv.org/abs/2410.01007)
- **Flöss & Meerburg (2023)**, "Improving constraints on primordial non-Gaussianity using neural network based reconstruction", arXiv:2305.07018. U-Net IC reconstruction of z=0 **matter**: 90% accuracy to k≤0.4. Pre+post-reconstruction P+B to $k_{\max}=0.52$ improves $f_{\rm NL}^{\rm loc}$ by **3.65×** over pre-reconstruction only — [arXiv:2305.07018](https://arxiv.org/abs/2305.07018)
- **Bottema, Flöss, Meerburg (2025)**, "Neural Network Reconstruction of Non-Gaussian Initial Conditions from Dark Matter Halos", arXiv:2502.11846, JCAP 08 (2025) 030. U-Net on redshift-space **halos** ($\bar n=4\times10^{-4}$, Quijote 1 Gpc³). 44% cross-correlation to k=0.4. Gains in $f_{\rm NL}$ from reconstruction: local 1.33×, equilateral 1.88×, orthogonal 1.57× — [arXiv:2502.11846](https://arxiv.org/abs/2502.11846)
- **Bairagi & Wandelt (2025)**, "PatchNet: A hierarchical approach for neural field-level inference from Quijote Simulations", arXiv:2509.03165. Patch CNN summaries plus P+B. Fisher information matches wavelet statistics at ~7.8 Mpc/h on DM — [arXiv:2509.03165](https://arxiv.org/abs/2509.03165)
- **Cagliari, Bairagi, Wandelt (2025)**, "Hierarchical summaries for primordial non-Gaussianities", arXiv:2510.12715, A&A 710, A12 (2026). PatchNet halo-field summaries plus P+B on Quijote-PNG give **30–45% gains on $f_{\rm NL}$ at $k_{\max}\sim0.1$**, capturing information beyond the bispectrum — [arXiv:2510.12715](https://arxiv.org/abs/2510.12715)
- **Jung, Ravenni, Liguori, Baldi, Coulton, et al. (2024)**, "Quijote-PNG: Optimizing the summary statistics to measure Primordial non-Gaussianity", arXiv:2403.00490. P, B, HMF, marked P and marked modal B up to $k_{\max}=0.5$. NN likelihood-free inference on summaries. Marked P beats B for equilateral — [arXiv:2403.00490](https://arxiv.org/abs/2403.00490)
- **Jung, Karagiannis, Liguori, Baldi, Coulton, et al. (2022)**, "Quijote-PNG: Quasi-maximum likelihood estimation of PNG in the non-linear halo density field", arXiv:2211.07565, ApJ 948, 135 (2023); and the dark-matter version arXiv:2206.01624, ApJ 940, 71 (2022). Local PNG in halos is dominated by scale-dependent bias — [arXiv:2211.07565](https://arxiv.org/abs/2211.07565); [arXiv:2206.01624](https://arxiv.org/abs/2206.01624)
- **Peron, Jung, Liguori, Pietroni (2024)**, "Constraining PNG from LSS with the Wavelet Scattering Transform", arXiv:2403.17657. On halos, WST beats P alone by ~27% on $f_{\rm NL}^{\rm loc}$ but is ~15% weaker than P+B — [arXiv:2403.17657](https://arxiv.org/abs/2403.17657)
- **Calles, Contardo, Noreña, Yip, Shiu (2025)**, "Primordial non-Gaussianity — Fast simulations and persistent summary statistics", arXiv:2512.09852, JCAP 08 (2026) 051. **PNG-pmwd**: >20,000 halo catalogs from differentiable pmwd varying local and equilateral $f_{\rm NL}$. Transfer to Quijote-PNG degrades when small scales or small halos are included — [arXiv:2512.09852](https://arxiv.org/abs/2512.09852)
- **Wang & He (2024)**, arXiv:2408.13876, PRD 111, L041302 (2025); and **Zhou, Wang, Zhang, Peron, Deng, et al. (2026)**, arXiv:2610.04486. Non-neural nonlinear-transform summaries (peak/valley functions; Zel'dovich-inspired transform) on Quijote-PNG. The second reports large gains for equilateral/orthogonal over $P_h$ — [arXiv:2408.13876](https://arxiv.org/abs/2408.13876); [arXiv:2610.04486](https://arxiv.org/abs/2610.04486)
- **Rouhiainen & Münchmeyer (2022)**, "De-noising non-Gaussian fields in cosmology with normalizing flows", arXiv:2211.15161 — [arXiv:2211.15161](https://arxiv.org/abs/2211.15161)
- **Melsen, Flöss, Meerburg (2024)**, arXiv:2412.12377, MNRAS (2025). Spherical CNNs for CMB PNG; adjacent only — [arXiv:2412.12377](https://arxiv.org/abs/2412.12377)

### Inferences
- For **local** $f_{\rm NL}$ on halos, the neural gains are much smaller than on the matter field (Giri 2023; Bottema 1.33× vs Flöss–Meerburg 3.65×). Kvasiuk et al. find mass-binned weighting already optimal unless extra halo properties are used. This matches the ~1.3× BORG gain on Quijote-PNG halos and suggests DESI QSO/LRG field-level gains for local $f_{\rm NL}$ will be modest.
- Halo-property (concentration) information is the one route reported to give "factor of a few" gains. It is unavailable for QSOs/LRGs except through proxies, which links to Hadzhiyska & Ferraro's $b_\nabla$–$b_\phi$ relation.

### Gaps
- No standalone paper found by Jung/Ravenni/Liguori et al. doing a CNN/GNN end-to-end field-level $f_{\rm NL}$ posterior. The GNN work appears only as a "preliminary field-level analysis" inside arXiv:2305.10597.
- No neural field-level $f_{\rm NL}$ paper applied to real survey data was found.

---

## Q5. Field-level reconstruction approaches aimed at PNG

### Takeaway
Reconstruction approaches fall into three families:
- Quadratic-estimator reconstruction of large-scale modes from small-scale modes, an effective multi-tracer: Darwish et al. 2020.
- Standard or NN IC reconstruction to Gaussianise covariance before P+B: Shirasaki et al. 2020; Flöss & Meerburg 2023; Bottema et al. 2025.
- External velocity reconstruction via kSZ for sample-variance cancellation: Laguë et al. 2024.

Each gives "tens of percent" to ~1.3–3× gains in specific setups.

### Cited Findings
- **Darwish, Foreman, Abidi, Baldauf, Sherwin, et al. (2020)**, "Density reconstruction from biased tracers and its application to primordial non-Gaussianity", arXiv:2007.08472. Quadratic estimator for large-scale modes including gravity, nonlinear bias and local PNG, validated on N-body. Gives **tens-of-percent tightening of local $f_{\rm NL}$** versus single-tracer, partly through cosmic-variance cancellation (an effective multitracer) — [arXiv:2007.08472](https://arxiv.org/abs/2007.08472)
- **Shirasaki, Sugiyama, Takahashi, Kitaura (2020)**, "Constraining Primordial Non-Gaussianity with Post-reconstructed Galaxy Bispectrum in Redshift Space", arXiv:2010.04567, PRD 103, 023506 (2021). Post-reconstruction bispectrum covariance is close to Gaussian at k<0.2, from 4000 sims. BOSS-like forecast $\Delta f_{\rm NL}^{\rm loc}\sim13$, a 1.3–3.2× improvement over pre-reconstruction — [arXiv:2010.04567](https://arxiv.org/abs/2010.04567)
- **Laguë, Madhavacheril, Smith, Ferraro, Schaan (2024)**, "Constraints on local primordial non-Gaussianity with 3d Velocity Reconstruction from the Kinetic Sunyaev-Zeldovich Effect", arXiv:2411.08240. ACT DR5 + Planck × SDSS. kSZ detected at 7.2σ. $f_{\rm NL}=-90^{+210}_{-350}$ from galaxy–velocity alone, as a pathfinder for sample-variance cancellation — [arXiv:2411.08240](https://arxiv.org/abs/2411.08240)
- **Flöss & Meerburg (2023)** and **Bottema et al. (2025)**: see Q4 — [arXiv:2305.07018](https://arxiv.org/abs/2305.07018); [arXiv:2502.11846](https://arxiv.org/abs/2502.11846)
- **McQuinn (2020)**: see Q3. Bounds what reconstruction can achieve — [arXiv:2008.12312](https://arxiv.org/abs/2008.12312)
- **Kokron (2025)**, "Local primordial non-Gaussianity from 'zero-bias' 21cm radiation during reionization", arXiv:2504.20025. Notes that new noise-like terms can only be mitigated with "field-level reconstruction", and that $\sigma(f_{\rm NL})\le1$ is unreachable even in simplified forecasts. Adjacent tracer — [arXiv:2504.20025](https://arxiv.org/abs/2504.20025)
- **Vislosky, Brown, Demina, Chaussidon (2025)**, "Beyond the two-point correlation: constraining primordial non-gaussianity with density perturbation moments", arXiv:2508.13381, A&A 704, A297 (2025). DESI-adjacent non-field-level alternative — [arXiv:2508.13381](https://arxiv.org/abs/2508.13381)

### Inferences
- Darwish et al.'s "single-tracer multitracer" mechanism is one way a field-level likelihood could beat P(k) for local $f_{\rm NL}$: small-scale modes implicitly reconstruct large-scale modes and cancel cosmic variance. A Schur-complement field-level Fisher should capture it automatically if the forward model includes the relevant couplings, i.e. 2LPT mode coupling and $b_2$/$b_K$.

### Gaps
- No paper found that does an "optimal quadratic estimator at field level" for local $f_{\rm NL}$ beyond Darwish et al. and the Giri/Münchmeyer/Smith local-power approach.

---

## Q6. Differentiable forward-model codes and samplers relevant to field-level PNG

### Takeaway
The main differentiable engines and samplers are:
- BORG (2LPT, PM, COLA, emulators; Jasche, Lavaux et al.)
- pmwd (JAX PM with adjoint)
- FlowPM (TensorFlow)
- JaxPM, used by the Simon–Lanusse–de Mattia sampler benchmark
- DISCO-DJ (JAX Einstein–Boltzmann and PM with adjoint and nuFFT)
- LEFTfield (EFT, not public on arXiv as code)

Sampler literature (Bayer et al. 2023; Simon et al. 2025) shows MCLMC is >10× more efficient than HMC at ≥10⁶ dimensions. This bears directly on the brief's $f_{\rm NL}$ mixing bottleneck.

### Cited Findings
- **Hahn, List, Porqueres (2023)**, "DISCO-DJ I: a differentiable Einstein-Boltzmann solver for cosmology", arXiv:2311.03291, JCAP 06 (2024) 063. JAX EB solver with Jacobians of all outputs. Per-mille agreement with CAMB/CLASS, including massive ν and $w$ — [arXiv:2311.03291](https://arxiv.org/abs/2311.03291)
- **List, Hahn, Flöss, Winkler (2025)**, "DISCO-DJ II: a differentiable particle-mesh code for cosmology", arXiv:2510.05206. Journal: —. GPU PM with the BullFrog integrator: ~6 steps give per-cent P(k) at k≈0.2 with 512³. Custom nuFFT forces. Forward and reverse AD with **memory independent of step count (adjoint)**. Demonstrates FLI of $\sigma_8$ and ICs from a noisy Gadget field — [arXiv:2510.05206](https://arxiv.org/abs/2510.05206)
- **Li, Lu, Modi, Jamieson, Zhang, et al. (2022)**, "pmwd: A Differentiable Cosmological Particle-Mesh N-body Library", arXiv:2211.09958; and **Li, Modi, Jamieson, Zhang, Lu, et al. (2022)**, "Differentiable Cosmological Simulation with Adjoint Method", arXiv:2211.09815. JAX PM with adjoint and reverse-time integration — [arXiv:2211.09958](https://arxiv.org/abs/2211.09958); [arXiv:2211.09815](https://arxiv.org/abs/2211.09815)
- **Modi, Lanusse, Seljak (2020)**, "FlowPM: Distributed TensorFlow Implementation of the FastPM Cosmological N-body Solver", arXiv:2010.11847 — [arXiv:2010.11847](https://arxiv.org/abs/2010.11847)
- **Simon, Lanusse, de Mattia (2025)**, "Benchmarking field-level cosmological inference from galaxy redshift surveys", arXiv:2504.20130, JCAP 12 (2025) 039. **JaxPM**-based benchmark of HMC, NUTS (with and without Gibbs), MAMS and MCLMC. Preconditioning matters. **Unadjusted MCLMC beats adjusted samplers by >10×** with mild dimension scaling at ≥10⁶ dimensions. Code is public. The DESI-adjacent group (de Mattia) is relevant — [arXiv:2504.20130](https://arxiv.org/abs/2504.20130)
- **Bayer, Seljak, Modi (2023)**, "Field-Level Inference with Microcanonical Langevin Monte Carlo", arXiv:2307.09504. MCLMC is >10× more efficient than HMC for a ~2.6×10⁵-dimensional IC + $\sigma_8$ + $\Omega_m$ problem, and the gain grows with dimension — [arXiv:2307.09504](https://arxiv.org/abs/2307.09504)
- **Jasche & Wandelt (2012)**, "Bayesian physical reconstruction of initial conditions from large scale structure surveys", arXiv:1203.3639. The original BORG (2LPT, HMC) — [arXiv:1203.3639](https://arxiv.org/abs/1203.3639)
- **Jasche & Lavaux (2018)**, "Physical Bayesian modelling of the non-linear matter distribution: new insights into the Nearby Universe", arXiv:1806.11117, A&A 625, A64 (2019). BORG-PM — [arXiv:1806.11117](https://arxiv.org/abs/1806.11117)
- **Tassev, Zaldarriaga, Eisenstein (2013)**, "Solving Large Scale Structure in Ten Easy Steps with COLA", arXiv:1301.0322, JCAP 06 (2013) 036. The COLA/tCOLA method used in BORG — [arXiv:1301.0322](https://arxiv.org/abs/1301.0322)
- **Stopyra, Peiris, Pontzen, Jasche, Lavaux (2023)**, "Towards Accurate Field-Level Inference of Massive Cosmic Structures", arXiv:2304.09193, MNRAS 527, 1244 (2024). BORG with a 20-step COLA integrator — [arXiv:2304.09193](https://arxiv.org/abs/2304.09193)
- **Doeser, Jamieson, Stopyra, Lavaux, Leclercq, et al. (2023)**, "Bayesian Inference of Initial Conditions from Non-Linear Cosmic Structures using Field-Level Emulators", arXiv:2312.09271, MNRAS 535, 1258 (2024). V-net emulator inside BORG HMC — [arXiv:2312.09271](https://arxiv.org/abs/2312.09271)
- **Doeser & Jasche (2026)**, "Learning the Universe: Posterior Reliability of Neural Generative Models in High-Dimensional Field-Level Inference of Cosmic Initial Conditions", arXiv:2606.10023. Compares stochastic interpolants and GLOW flows with HMC reference posteriors — [arXiv:2606.10023](https://arxiv.org/abs/2606.10023)
- **McAlpine, Jasche, Ata, Lavaux, Stiskalek, et al. (2025)**, "The Manticore Project I: a digital twin of our cosmic neighbourhood from Bayesian field-level analysis", arXiv:2505.10682 — [arXiv:2505.10682](https://arxiv.org/abs/2505.10682)
- **McAlpine, Jasche, Lavaux, Doeser, Loureiro (2026)**, "The Manticore Project II: Bayesian digital twins of cosmic structure across the SDSS and BOSS volumes", arXiv:2606.10020, MNRAS 550 (4), stag1366 (2026). Manticore-Deep: (4 Gpc/h)³ to z≈0.7 at ~4 Mpc/h with tiled BORG inference over 5 surveys. CMB-lensing cross-correlation at 7.4σ and kSZ at 3.5σ — [arXiv:2606.10020](https://arxiv.org/abs/2606.10020)
- **Crespi, Bonici, Loureiro, et al. (2025)**, "Flinch: A Differentiable Framework for Field-Level Inference of Cosmological parameters from curved sky data", arXiv:2510.26691. Adjacent: angular maps, MCLMC-type sampling, up to 40% smaller errors than pseudo-$C_\ell$ — [arXiv:2510.26691](https://arxiv.org/abs/2510.26691)

### Inferences
- The project's choice of a JAX stack (DISCO-DJ) can reuse the open Simon–Lanusse–de Mattia benchmark code (JaxPM, MCLMC) almost directly for the Stage-2 sampler comparison. Their >10× MCLMC result is the strongest published evidence that the brief's ≤10³ $f_{\rm NL}$-correlation-length target is achievable. They sample cosmology, not $f_{\rm NL}$, so this is untested for PNG.
- BORG's $f_{\rm NL}$ chains use Gibbs with slice sampling on $f_{\rm NL}$: Andrews et al. 2026 report ESS ~2000, and the brief says correlation length ~10⁴ in Euclid. No paper found tests joint HMC/MCLMC on (ICs, $f_{\rm NL}$, bias), so this would be novel.

### Gaps
- JaxPM has no dedicated arXiv paper; it is cited via GitHub and the Simon et al. benchmark. Not verified as a citable paper.
- No paper found using DISCO-DJ, pmwd, JaxPM or FlowPM for $f_{\rm NL}$ field-level inference. The only PNG use of these codes is PNG-pmwd for SBI training sets (arXiv:2512.09852).
- No specific "BORG-tCOLA" methods paper was verified beyond Stopyra et al. 2023 (COLA within BORG) and Tassev et al. 2013 (COLA).
- LEFTfield has no standalone code paper on arXiv; the Stadler et al. 2024 Part I/II papers describe the forward model.

---

## Q7. Other adjacent items seen during the search (brief)
### Takeaway
Some related papers surfaced that are outside the core scope but could matter for robustness or bias priors.
### Cited Findings
- **Hoyland, Saadeh, Koyama, Desmond (2026)**, "Disentangling modified gravity and galaxy bias with field-level inference", arXiv:2607.03514. BORG-type FLI applied to modified gravity, not PNG — [arXiv:2607.03514](https://arxiv.org/abs/2607.03514)
- **Bragança, Donath, Senatore, Zheng (2023)**, "Peeking into the next decade in LSS Cosmology with its EFT", arXiv:2307.04992. 1-loop P + B Fisher with $\sigma(f_{\rm NL}^{\rm loc})=3.3$ for DESI and 0.26 for MegaMapper — [arXiv:2307.04992](https://arxiv.org/abs/2307.04992)
- **Leclercq & Heavens (2021)**, "On the accuracy and precision of correlation functions and field-level inference in cosmology", arXiv:2103.04158, MNRAS Lett. 506, L85 — [arXiv:2103.04158](https://arxiv.org/abs/2103.04158)
### Inferences
- The DESI P+B forecast of 3.3 (Bragança et al.) sits close to the project's field-level go/no-go threshold of ≲5. A field-level analysis would need to show it beats a full P+B analysis, not just the P-only DESI DR1 result.
### Gaps
- Not searched exhaustively.
