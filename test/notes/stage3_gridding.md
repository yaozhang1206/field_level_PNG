# Stage 3: gridding of data and model (window, aliasing, noise level)

Sample fiducial_HR_z0.5_LRG, realisations 0–3, $16^3$ grid (62.5 Mpc/h). Figure: `test/figures/stage3_gridding.png`.

**Problem.** With one particle per cell scattered to the analysis grid (the Stage 1–2 model), the model's grid power falls toward $k_{\rm Ny}$ much faster than CIC- or NGP-gridded halos (left panel, factor 2 at $k = 0.05$). Halo structure below the cell size aliases into the grid, which the model has no modes for. On halos the posterior compensated with $b_1 \approx 1.05$, $f_{\rm NL} \approx +700$ (CIC data) or $b_\nabla \approx 270$, noise $\times 5.8$, $f_{\rm NL} \approx -190$ (block-averaged 4× model, NGP data).

**Fix (DR1-applicable).** Model particles (still one per coarse cell) and data are CIC-assigned to a mesh $m\times$ finer, the kept modes ($|n_i| < r/2$) are divided by the fine CIC window, truncated to the coarse grid and sampled at its nodes. Data power is then independent of $m$ to 0.2 %. The model converges at $m = 4$ ($m = 2$ is 5 % high at $k_{\rm Ny}$); a finer particle lattice changes nothing (0.3 %). DR1 cost: 160³ particles + one 640³ assignment mesh per shell, not 640³ particles.

**Noise.** With the truncated data the noise amplitude cannot be sampled freely: the white-noise field has more dof than the kept data modes and $\ln A \to -27$ (σ collapse). The expected shot noise of deconvolved truncated CIC is $\prod_i [1 - \tfrac23\sin^2(\pi m_i/n)]/{\rm sinc}^4(m_i/n) \approx 1$ at $m=4$, so the Gaussian variance is fixed at Poisson $\bar n V_{\rm cell}$ (as in Stage 2).

**Pilot on halos** (redshift space, cold start, $b_\nabla$ free, $p = 1$): realisation 0: $f_{\rm NL} = -91 \pm 68$, $b_1 = 1.98$, $b_\nabla = 51 \pm 27$; realisation 1: $+50 \pm 65$, $b_1 = 1.94$, $b_\nabla = 53 \pm 28$. The ensemble runs measure the mean bias.

Figure: `test/figures/stage3_gridding.png`
