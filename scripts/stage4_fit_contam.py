"""Stage 4 diagnostic: how much of the DR1 per-bin imaging contamination can each template model absorb?
Weighted linear least squares of the contamination field (g - 1) on the template set over each tracer's footprint
(weights = expected counts), at the analysis grid. Reports residual rms and the residual's large-scale angular
power (l <= 20 on the shell-averaged map), the part that biases f_NL."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stage1_config import build
from geometry import dr1_angular_mask
from forward_model import ForwardModel
from templates import grid_template_compact, dense_from_compact, wsys_contamination_desi, grid_directions
res = int(sys.argv[1]) if len(sys.argv) > 1 else 80
cosmo, geoms, shells, nz = build()
fm = ForwardModel(cosmo, shells, L=10000., res=res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
lab0, comp0 = grid_template_compact(fm, fm.tracers, intercepts=False)
labI, compI = grid_template_compact(fm, fm.tracers, intercepts=True)
M0 = dense_from_compact(fm, lab0, comp0); MI = dense_from_compact(fm, labI, compI)
from numpy.polynomial.legendre import legval
for t in fm.tracers:
    g = wsys_contamination_desi(fm, t).ravel() - 1
    w = sum(np.asarray(fm.nbar_cell[i]).ravel() for i, s in enumerate(fm.shells) if s.tracer == t)
    fp = w > 0
    k0 = [k for k, (tt, _) in enumerate(lab0) if tt == t]; kI = [k for k, (tt, _) in enumerate(labI) if tt == t]
    base = {"DESI (no intercept)": M0[k0], "DESI + intercepts": MI[kI]}
    # + quadratic: squares of each feature per group
    Q = np.array([M0[k] ** 2 * (M0[k] != 0) for k in k0])
    Q = Q - np.array([np.where(M0[k] != 0, np.average(Q[j][M0[k] != 0], weights=w[M0[k] != 0]) if np.any(M0[k] != 0) else 0, 0) for j, k in enumerate(k0)])
    base["DESI + intercepts + squares"] = np.concatenate([MI[kI], Q])
    # + per-region radial P1, P2 (n(z) shape per region)
    G = compI[t]["G"]; chi = fm.chi_grid.ravel(); R = []
    for i, s in enumerate(fm.shells):
        if s.tracer != t: continue
        u = np.clip(2 * (chi - s.chi_lo) / (s.chi_hi - s.chi_lo) - 1, -1, 1)
        inshell = np.asarray(fm.nbar_cell[i]).ravel() > 0
        for gid in np.unique(G[inshell & (G > 0)]):
            sel = inshell & (G == gid)
            for l in (1, 2):
                c = np.zeros(l + 1); c[l] = 1; R.append(np.where(sel, legval(u, c), 0.0))
    base["DESI + intercepts + region radial P1,P2"] = np.concatenate([MI[kI], np.array(R)])
    base["all of the above"] = np.concatenate([MI[kI], Q, np.array(R)])
    print(f"== {t}: contamination rms {np.sqrt(np.average(g[fp]**2, weights=w[fp])):.4f}")
    for name, X in base.items():
        X = X[:, fp]; ww = w[fp]; y = g[fp]
        A = (X * ww) @ X.T; b = (X * ww) @ y
        coef = np.linalg.lstsq(A, b, rcond=None)[0]; r = y - coef @ X
        print(f"   {name:42s} n={len(X):4d}  residual rms {np.sqrt(np.average(r**2, weights=ww)):.4f}  ({100*np.sqrt(np.average(r**2, weights=ww)/np.average(y**2, weights=ww)):.0f}% of rms)")
