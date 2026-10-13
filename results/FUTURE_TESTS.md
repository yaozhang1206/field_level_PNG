# Future tests and model changes (backlog)

Written 2026-10-13. Items deferred while Stages 3 and 4 run. Each entry says why it matters, what to change and how
to test it. Literature references are from `reports/PNG field level literature.md`.

---

## 1. Redshift errors and line-of-sight smearing (the field-level analogue of $\Sigma_s$)

**Why.** Chaussidon et al. 2024 (arXiv:2411.17623, §3.1.2 and Eq. 2.4) absorb residual QSO redshift errors and the
velocity dispersion inside host halos into the damping $[1 + (k\mu\Sigma_s)^2/2]^{-2}$ of the redshift-space power
spectrum. Our forward model displaces objects only by the 2LPT velocity field (Kaiser term), with no random
line-of-sight smearing, so it has no counterpart of $\Sigma_s$.

**Size.** A velocity error $v$ shifts an object along the line of sight by $\Delta\chi = v(1+z)/H(z) \approx 1\,h^{-1}$Mpc
per 100 km/s across the QSO shells. A few hundred km/s is 2–5 $h^{-1}$Mpc, bracketing DESI's fiducial $\Sigma_s = 3\,h^{-1}$Mpc.
The power suppression is $\approx (k\mu\Sigma_s)^2$ along the line of sight, about one third of that averaged over $\mu$:

| $k$ [$h/$Mpc] | $\Sigma_s = 3$ | $\Sigma_s = 5$ |
|---|---|---|
| 0.01 (where most $f_{\rm NL}$ information sits) | < 0.1 % | 0.25 % |
| 0.05 (Nyquist of the $160^3$ DR1 grid) | 2 % | 6 % |
| 0.1 (Nyquist of a $320^3$ grid) | 9 % | 25 % |

Small at the current resolution, comparable to the other model errors at $31\,h^{-1}$Mpc cells.

**Catastrophic redshifts** (a few per cent of QSOs at a wrong redshift, often from line confusion) are a separate effect
that $\Sigma_s$ does not absorb either:
- *Dilution.* A fraction $f$ uncorrelated with the shell scales the clustering amplitude to $(1-f)b_1$. The fitted $b_1$
  absorbs this, but the universality relation then gives $b_\phi = 2\delta_c[(1-f)b_1 - p]$ instead of the true
  $(1-f)\,2\delta_c(b_1 - p)$. For $f = 0.02$, $b_1 = 2.4$, $p = 1.6$ this is a 4 % multiplicative error on $f_{\rm NL}$,
  well below the uncertainty in $p$, and present in the P(k) analysis too.
- *Leakage between shells.* Line confusion shifts objects by a fixed $\Delta z$, correlating distant shells; the likelihood
  treats shells as independent given the field.

**Changes to make.**
1. Add an anisotropic counterterm $c_\parallel\,\partial_\parallel^2\delta$ per tracer, with the derivative along the local line
   of sight. This is the leading order of the $\Sigma_s$ damping, $1 - \tfrac12k^2\mu^2\Sigma_s^2$, and the standard EFT
   counterterm for redshift-space smearing (Schmittfull et al. 2021). The isotropic Laplacian bias $b_\nabla$ absorbs only
   part of it.
2. Optionally, a dilution fraction $f$ per QSO shell entering $b_\phi$ as above, with a prior from DESI's measured
   catastrophic rate.

**Test.** Add Gaussian line-of-sight scatter (300–500 km/s) and 1–2 % catastrophic redshifts (uniform within the
tracer's range, plus a line-confusion component) to the Stage 3 halo boxes. Rerun one $f_{\rm NL} = 0$ ensemble and the
$\pm100$ pair with and without $c_\parallel$, and compare the mean offset and the response with the clean Stage 3 runs.
Stage 3 so far used true halo velocities, so it did not test this.

## 2. Spectroscopic-efficiency (redshift-failure) systematics

**Why.** DESI's `WEIGHT_ZFAIL` corrects completeness variations with observing conditions (seeing, transparency,
fibre and CCD effects). In the forward model this is an angular, and partly focal-plane, modulation of the selection,
the same kind of term as the imaging templates of Stage 4.

**Change.** Add an observing-condition template, e.g. the effective exposure depth (TSNR2) per tile or healpix pixel,
to the Stage 4 template set, with a free amplitude per tracer and region.

**Test.** Contaminate a Stage 4 mock with $1/\langle w_{\rm zfail}\rangle$ mapped on the sky (as done for $w_{\rm sys}$) and
check the bias with and without the new template.

## 3. $b_{\phi\delta}$ is probably mis-specified

**Why.** Our model (like Andrews et al. 2024, 2026) fixes
$b_{\phi\delta} = b_\phi - b_1 + 1 + \delta_c[b_2 - \tfrac{8}{21}(b_1 - 1)]$ from the universal mass function. Barreira 2022
(arXiv:2107.06887) measured $b_{\phi\delta}$ directly with separate-universe simulations. The formula overpredicts the
halo values by up to $\approx 3$, and he gives the fit
$$b_{\phi\delta} = 3.85 - 9.49\,b_1 + 3.44\,b_1^2 .$$
Moradinezhad Dizgah et al. 2021 show that fixing both $b_\phi$ and $b_{\phi\delta}$ to the universality values shifts
$f_{\rm NL}$. $b_{\phi\delta}$ multiplies $\delta\,\phi$, so an error in it feeds directly into the $f_{\rm NL}$ coupling, and the
Stage 1–2 results show that much of the field-level information beyond P(k) comes through that coupling. It is a
plausible cause of the possible $+18 \pm 9$ offset at $f_{\rm NL} = 0$ in Stage 3.

**Changes to make.** Two variants:
1. $b_{\phi\delta}$ from the Barreira 2022 fit instead of the universality formula;
2. $b_{\phi\delta}$ free per shell (flat or wide Gaussian prior), with $b_\phi$ kept fixed so the test isolates $b_{\phi\delta}$.

**Test.** On the same Stage 3 boxes (LRG-like, both QSO-like samples, the control and the $\pm100$ pair), rerun with each
variant and compare the $f_{\rm NL} = 0$ offset and the $\pm100$ response with the baseline. If the offset disappears with
variant 1, adopt it for DR1; if only with variant 2, quantify the $\sigma(f_{\rm NL})$ cost of freeing $b_{\phi\delta}$ on the DR1
mocks (Stage 2 setup).

## 4. Better Stage 3 test sets: AbacusPNG and PNG-UNIT

**Why.** Quijote-PNG has only $f_{\rm NL} = \pm100$, and its PNG halo catalogues (standard resolution) are complete only
above $\approx 6\times10^{13}\,M_\odot/h$. That forced the response test onto a high-mass sample ($b_1 \approx 3$,
$\bar n \approx 3\times10^{-5}$) unlike DESI LRGs or QSOs, and $\pm100$ is far from realistic values.

| suite | box, particles | $f_{\rm NL}$ values | halos | why useful |
|---|---|---|---|---|
| AbacusPNG | 2 $h^{-1}$Gpc, $4096^3$ | $-100, -30, 0, 30, 100$, two phases each | CompaSO halos, light cones | response near realistic $f_{\rm NL}$; resolves DESI-like hosts; light cone tests the shell treatment |
| PNG-UNITsims (arXiv:2312.12405) | 1 $h^{-1}$Gpc, $4096^3$ | 100 (and $-20$ runs used later) | Rockstar, resolved down to QSO-host masses | $p$ measured as $0.955 \pm 0.013$ for $10^{12}$–$5\times10^{13}$; tests the QSO regime Quijote cannot reach |

**Test.** Repeat the Stage 3 protocol (same DR1 model, same gridding, $16^3$ and $32^3$ cells for 1 $h^{-1}$Gpc, or the
DR1 cell size in 2 $h^{-1}$Gpc boxes): the $f_{\rm NL} = 0$ offset and the response for DESI-like LRG and QSO samples, then
the $\pm30$ response with $b_\phi$ fixed and sampled. Data volume: download in batches, cut, delete raw (scratch is
nearly full); check access and file layout first.

## 5. Other items carried forward

- From the literature on gridding (Stadler, Schmidt & Reinecke 2025; Sefusatti et al. 2016; Schmidt et al. 2020;
  Schmidt 2021): interlacing for the assignment mesh, a separate sharp-$k$ cut on the white-noise field, and sizing the
  fine mesh for the quadratic terms.
- A fitted $b_\nabla$ far from the physical $\approx -R_L^2$ (Lazeyras & Schmidt 2019) flags residual window mismatch;
  our Stage 3 values are $+25$ to $+70\,(h^{-1}{\rm Mpc})^2$.
- Noise model: the density-dependent noise likelihood of Rubira & Schmidt 2025 (arXiv:2511.05484) is the principled fix
  for $\sigma$ collapse and may explain the too-narrow halo posteriors.
- Compare with the stronger DR1 baselines: Chudaykin, Ivanov & Philcox 2025 (P+B, $\sigma = 7.4$; check their $b_\phi$
  treatment) and the one-loop P+B forecast of Bragança et al. 2023 ($\sigma \approx 3.3$ for DESI).
- Stage 1 Part A LRG configuration: the DESI baseline uses $0.6 < z < 1.1$ (1,631,716 LRGs), not $0.4 < z < 1.1$.
