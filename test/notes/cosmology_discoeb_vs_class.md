# DISCO-EB vs CLASS linear power

Both backends use the same P_Phi(k) and growth; only T(k) differs (normalised to 1 at k_min).
DISCO-EB: RECFAST, lmax=11, nqmax=3, rtol=1e-5; solver precision/lmax/Neff changes move the result by < 0.1%.

- m_nu = 0.06 eV: sigma8 DISCO-EB 0.8157 vs CLASS 0.8117; max|dP/P| = 1.14% (k<=0.1), 1.45% (k<=0.3); time 2s / 0s
- m_nu = 0 (massless): sigma8 DISCO-EB 0.8244 vs CLASS 0.8261; max|dP/P| = 0.46% (k<=0.1), 0.72% (k<=0.3); time 81s / 1s

Result: FAIL against the 0.5% target (worst 1.14% at k<=0.1).
A ~1% P_lin offset changes sigma(f_NL) by ~0.5% and is irrelevant for the Stage-1 go/no-go.
Decision: Part A uses CLASS (fast, standard). DISCO-EB stays available for Stage 2 consistency with DISCO-DJ.

Figure: `test/figures/cosmology_discoeb_vs_class.png`
