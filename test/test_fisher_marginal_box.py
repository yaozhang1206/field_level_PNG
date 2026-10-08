"""Validate the Hutchinson Gaussian-marginal Fisher on the linear model in a periodic box against the exact
Tier 1 mode sum (laplace=False). Also prints the Schur (profile) value and the Laplace-weighted mode sum."""
from _common import *
import numpy as np, time, json, jax, jax.numpy as jnp
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from fisher_closed import Tier1
from fisher_field import FieldFisher, sigma_of
res = int(os.environ.get("RES", 80)); L = 10000.0; probes = int(os.environ.get("PROBES", 8))
cosmo = Cosmology(backend="class")
shells = [default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]) for t in ("QSO", "LRG")]
nbar = {"QSO": 2.5e-5, "LRG": 3.0e-4}
nz = {t: (lambda zz, n=nbar[t]: np.full_like(zz, n)) for t in nbar}
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs=nz, ang_mask=None, observer=[L / 2, L / 2, -200 * L])
for i in range(len(shells)):
    fm.nbar_cell[i] = jnp.full((res,) * 3, nbar[shells[i].tracer] * fm.V_cell)
fm.footprint = {t: np.ones((res,) * 3, bool) for t in fm.tracers}
class _SG:
    def tables(self, s): return np.array([1.0]), np.array([L ** 3]), np.array([nbar[s.tracer]])
lines, rows = [f"periodic box {res}^3, uniform nbar QSO {nbar['QSO']:.1e}, LRG {nbar['LRG']:.1e}, {probes} probes"], {}
for tr in ("QSO", "LRG", "QSO+LRG"):
    sh = [s for s in shells if s.tracer in tr]; lay_t = ParamLayout(sh)
    t1 = Tier1(cosmo, sh, lay_t, _SG(), L=L, res=res, continuum=False)
    ex, lap = sigma_of(t1.fisher(laplace=False)), sigma_of(t1.fisher(laplace=True))
    lay = ParamLayout(shells); trs = [t for t in fm.tracers if t in tr]
    eps = jax.random.normal(jax.random.PRNGKey(7), (res,) * 3, dtype=jnp.float64)
    t0 = time.time()
    ff = FieldFisher(fm, lay, eps, lay.fiducial(), rsd=True, linear=True, verbose=False, tracers=trs)
    s_schur = sigma_of(ff.fisher()); Fm = ff.fisher_marginal(n_probes=probes); s_marg = sigma_of(Fm)
    rows[tr] = dict(exact=ex, laplace=lap, schur=s_schur, marginal=s_marg, probe_std=float(ff.F_marginal_probe_std), time=time.time() - t0)
    lines.append(f"- {tr}: exact Gaussian mode sum {ex:.2f} | Hutchinson marginal {s_marg:.2f} (probe-to-probe std {rows[tr]['probe_std']:.2f}) | Schur {s_schur:.2f} | Laplace-weighted mode sum {lap:.2f}   [{rows[tr]['time']:.0f}s]")
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.arange(3); names = list(rows)
for j, (k, lab, m) in enumerate([("exact", "exact mode sum", "k_"), ("marginal", "Hutchinson marginal", "o"), ("schur", "Schur (profile)", "s"), ("laplace", "Laplace-weighted mode sum", "^")]):
    ax.plot(x + 0.1 * j - 0.15, [rows[n][k] for n in names], m, label=lab, ms=7, mfc="none" if j else None)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel(r"$\sigma(f_{NL})$"); ax.legend(fontsize=7); ax.set_title(f"Linear model, periodic box {res}^3")
save_fig(fig, f"fisher_marginal_box_res{res}")
lines += ["", "Expect: Hutchinson marginal == exact mode sum (within probe noise); Schur == Laplace-weighted mode sum."]
write_note(f"fisher_marginal_box_res{res}", "Hutchinson Gaussian-marginal Fisher vs exact mode sum (periodic box)", lines)
json.dump(rows, open(ROOT / "results" / f"fisher_marginal_box_res{res}.json", "w"), indent=1)
print("\n".join(lines))
