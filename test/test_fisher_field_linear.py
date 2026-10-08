"""Operator validation (GPU): Schur-complement Fisher on the strictly linear model vs the closed-form Tier 1.
Expected: Schur == Tier 1 with laplace=True weighting (per mode dP dP / 4P(P+N)); the exact Gaussian-field
Fisher (laplace=False) differs by (P+N)/(2P) per mode. Uses the DR1 footprint, QSO+LRG shells, 80^3 (fast)."""
from _common import *
import numpy as np, time, json, jax, jax.numpy as jnp
from stage1_config import build
from geometry import dr1_angular_mask
from forward_model import ForwardModel, ParamLayout
from fisher_closed import ShellGeometry, Tier1
from fisher_field import FieldFisher, sigma_of

res = int(os.environ.get("RES", 80)); L = 10000.0; nreal = int(os.environ.get("NREAL", 2))
cosmo, geoms, shells, nz = build()
BOX = os.environ.get("BOX", "0") == "1"      # BOX=1: window-free periodic box, one shell per tracer, uniform nbar
if BOX:
    from forward_model import default_shell
    from stage1_config import B1_OF_Z, P_UMF
    shells = [default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]) for t in ("QSO", "LRG")]
    nbar = {"QSO": 2.5e-5, "LRG": 3.0e-4}
    nz = {t: (lambda zz, n=nbar[t]: np.full_like(zz, n)) for t in nbar}
    fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs=nz, ang_mask=None, observer=[L / 2, L / 2, -200 * L])
    for i in range(len(shells)):
        fm.nbar_cell[i] = jnp.full((res,) * 3, nbar[shells[i].tracer] * fm.V_cell)
    fm.footprint = {t: np.ones((res,) * 3, bool) for t in fm.tracers}
    class _SG:                                  # whole box volume, uniform density, one z bin
        def tables(self, s): return np.array([1.0]), np.array([L ** 3]), np.array([nbar[s.tracer]])
    sg = _SG()
else:
    mask = dr1_angular_mask(64)
    fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs=nz, ang_mask=mask)
    sg = ShellGeometry(geoms)
lines = [f"res $={res}$, {'periodic box, no window' if BOX else 'DR1 footprint'}; voxels: " + ", ".join(f"{t}: {int(fm.footprint[t].sum())}" for t in fm.tracers),
         r"Linear model: $\langle N\rangle = \bar nV\,[1 + (b_1 + b_\phi f_{\rm NL}/T(k) + f\mu_z^2)\,D\,\delta_L]$; Schur $= F_{\theta\theta} - F_{\theta\varepsilon}F_{\varepsilon\varepsilon}^{-1}F_{\varepsilon\theta}$; Tier 1 weights: exact $\partial P\partial P/(P+N)^2$, Laplace $\partial P\partial P/(2P(P+N))$ per mode."]
rows = []
ref = {}
for tr in ("QSO", "LRG", "QSO+LRG"):
    sh = [s for s in shells if s.tracer in tr]; lay_t = ParamLayout(sh)
    t1 = Tier1(cosmo, sh, lay_t, sg, L=L, res=res, continuum=False)
    ref[tr] = (t1.fisher(laplace=True), t1.fisher(laplace=False))
lay = ParamLayout(shells)
for r in range(nreal):
    eps = jax.random.normal(jax.random.PRNGKey(100 + r), (res,) * 3, dtype=jnp.float64)
    row = {"real": r}
    for tr in ("QSO", "LRG", "QSO+LRG"):
        trs = [t for t in fm.tracers if t in tr]
        t0 = time.time()
        ff = FieldFisher(fm, lay, eps, lay.fiducial(), rsd=True, linear=True, verbose=False, tracers=trs)
        F_s = ff.fisher()
        row[tr] = dict(schur=sigma_of(F_s), tier1_laplace=sigma_of(ref[tr][0]), tier1_exact=sigma_of(ref[tr][1]),
                       time=time.time() - t0, cg_resid_max=float(max(ff.iters)))
        if tr == "QSO+LRG":
            F_last, F_lap, F_exact = F_s, ref[tr][0], ref[tr][1]
    rows.append(row)
    lines.append(f"- realisation {r}: $\\sigma(f_{{\\rm NL}})$: " + "; ".join(
        f"{tr}: Schur {row[tr]['schur']:.2f} vs Tier1-Laplace {row[tr]['tier1_laplace']:.2f} (exact Gaussian {row[tr]['tier1_exact']:.2f}) [{row[tr]['time']:.0f}s, CG {row[tr]['cg_resid_max']:.0e}]" for tr in ("QSO", "LRG", "QSO+LRG")))
# diagonal comparison figure
fig, ax = plt.subplots(figsize=(7, 3.4))
names = lay.names
d_lap = np.diag(F_lap); d_ex = np.diag(F_exact); d_sc = np.diag(F_last)
ok = d_lap > 0
x = np.arange(ok.sum())
ax.plot(x, d_sc[ok] / d_lap[ok], "o", label="Schur / Tier1-Laplace (expect 1)")
ax.plot(x, d_sc[ok] / d_ex[ok], "s", label="Schur / Tier1-exact")
ax.set_xticks(x); ax.set_xticklabels(np.array(names)[ok], rotation=60, fontsize=7); ax.axhline(1, color="k", lw=0.6)
ax.set_ylabel("ratio of Fisher diagonals"); ax.legend(fontsize=8); ax.set_title(f"Linear-model operator check, {res}^3, last realisation")
save_fig(fig, f"fisher_field_linear_check_res{res}{'_box' if BOX else ''}")
json.dump(rows, open(ROOT / "results" / f"fisher_field_linear_check_res{res}{'_box' if BOX else ''}.json", "w"), indent=1)
lines += ["", r"Interpretation: in the periodic box, Schur must equal Tier1-Laplace (same modes, no window) up to the realisation",
          r"scatter of the few largest-scale modes; with the DR1 footprint the window removes the $k \sim k_f$ modes that dominate the",
          r"$k_{\min} = k_f$ mode sum, so Schur is larger than the full-box closed form (compare with the $k_{\min} = 0.003$ closed form).",
          r"The Schur (joint-Gaussian) estimate is not the exact marginal: per mode it is $(P+N)/(2P)$ times the Gaussian-field Fisher."]
write_note(f"fisher_field_linear_check_res{res}{'_box' if BOX else ''}", "Schur-complement operator check on the linear model" + (" (periodic box)" if BOX else " (DR1 footprint)"), lines)
print("\n".join(lines))
