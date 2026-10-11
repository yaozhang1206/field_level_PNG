"""Does the quadratic-coupling information exist in the posterior? Small periodic box with the production cell
size (L=2000, 16^3 -> 125 Mpc/h, k_Ny = 0.025), uniform nbar, Gaussian noise, f_NL,true = 0.
Compare sigma(f_NL) from NUTS on (a) the strictly linear model and (b) the linearised... no: the FULL 2LPT model
(b1, b2, bK free, bphi/bphidelta fixed), with the Fisher predictions: Tier 1 exact, Hutchinson (linearised 2LPT),
Schur (profile). Env: RES, L, NS, NW, TRACER (LRG|QSO|both)."""
from _common import *
import numpy as np, jax, jax.numpy as jnp, time, json, os
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from posterior import FieldPosterior
from fisher_field import FieldFisher, sigma_of
from fisher_closed import Tier1
from sampler import run_nuts, summarize_chain
res = int(os.environ.get("RES", 16)); L = float(os.environ.get("L", 2000)); cosmo = Cosmology(backend="class")
NS = int(os.environ.get("NS", 2000)); NW = int(os.environ.get("NW", 400)); TR = os.environ.get("TRACER", "LRG")
nbar_all = {"QSO": 2.5e-5, "LRG": 3e-4}
tracers = ["QSO", "LRG"] if TR == "both" else [TR]
shells = [default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]) for t in tracers]
nbar = {t: nbar_all[t] for t in tracers}
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda z, n=nbar[t]: np.full_like(z, n)) for t in nbar}, ang_mask=None, observer=[L/2, L/2, -200*L])
MASK = float(os.environ.get("MASK", 0))      # >0: keep only voxels within a sphere of radius MASK*L about the centre
x = (np.arange(res) + 0.5) / res - 0.5; X, Y, Z = np.meshgrid(x, x, x, indexing="ij"); inside = np.sqrt(X**2 + Y**2 + Z**2) < MASK if MASK > 0 else np.ones((res,)*3, bool)
for i, s in enumerate(shells): fm.nbar_cell[i] = jnp.asarray(np.where(inside, nbar[s.tracer] * fm.V_cell, 0.0))
fm.footprint = {t: inside.copy() for t in fm.tracers}
lay = ParamLayout(shells); th0 = lay.fiducial(0.0)
eps0 = jax.random.normal(jax.random.PRNGKey(21), (res,)*3, dtype=jnp.float64)
class _SG:
    def tables(self, s): return np.array([1.0]), np.array([L**3 * inside.mean()]), np.array([nbar[s.tracer]])
lines = [f"box $L={L:.0f}$, ${res}^3$ (cell {L/res:.0f} Mpc/h, $k_{{\\rm Ny}}={np.pi*res/L:.4f}$), tracers {tracers}, $\\bar n$ {nbar}, Gaussian noise, $f_{{\\rm NL,true}}=0$" + (f"; spherical window radius {MASK}L ({100*inside.mean():.0f}% of voxels)" if MASK > 0 else "")]
res_out = {}
t1 = Tier1(cosmo, shells, lay, _SG(), L=L, res=res, continuum=False)
F1 = t1.fisher(); s_tier1 = sigma_of(F1)
for model in ("linear", "full"):
    linear = model == "linear"
    ff = FieldFisher(fm, lay, eps0, th0, linear=linear, verbose=False); Fs = ff.fisher(); s_schur = sigma_of(Fs)
    Fm = ff.fisher_marginal(n_probes=24); s_marg = sigma_of(Fm)
    post = FieldPosterior(fm, lay, linear=linear, likelihood="gaussian")
    d = post.mock(eps0, th0, seed=5)
    free = np.array([0] + [1 + i * lay.per_shell for i in range(len(shells))]) if linear else np.array([0] + [1 + i * lay.per_shell + j for i in range(len(shells)) for j in range(3)])
    scale = np.zeros(lay.n); dg = np.diag(ff.F_tt); scale[free] = 1 / np.sqrt(np.maximum(dg[free], 1e-12))
    logdens, unpack = post.make_logdensity(d, theta_scale=scale, free=free)
    st0 = post.pack(eps0, th0, scale, free)
    out = run_nuts(logdens, st0, n_warmup=NW, n_samples=NS, seed=1, unpack=unpack, log=lambda *a: None)
    summ = summarize_chain(out["theta"], lay.names); s = summ["f_NL"]
    res_out[model] = dict(tier1_exact=s_tier1, hutchinson=s_marg, schur=s_schur, chain=s, chain_fnl=out["theta"][:, 0].tolist(),
                          mean_leapfrog=float(np.mean(out["n_steps"])), accept=float(np.nanmean(out["accept"])), time=out["time"])
    lines.append(f"- {model} model: NUTS $\\sigma(f_{{\\rm NL}})$ = **{s['std']:.3f}** (mean {s['mean']:.3f}, $\\tau$ {s['tau']:.1f}, ESS {s['ess']:.0f}, {out['time']:.0f}s, leapfrog {res_out[model]['mean_leapfrog']:.0f}/sample) | "
                 f"Fisher: Tier 1 exact {s_tier1:.3f}, Hutchinson(linearised) {s_marg:.3f}, Schur {s_schur:.3f}")
    print(lines[-1], flush=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, m in zip(axes, res_out):
    x = np.array(res_out[m]["chain_fnl"]); ax.plot(x, lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-res_out[m]["tier1_exact"], res_out[m]["tier1_exact"], color="g", alpha=0.15, label=f"Tier 1 exact {res_out[m]['tier1_exact']:.3g}")
    ax.axhspan(-res_out[m]["hutchinson"], res_out[m]["hutchinson"], color="r", alpha=0.12, label=f"Hutchinson linearised {res_out[m]['hutchinson']:.3g}")
    ax.set_title(f"{m} model: chain std {res_out[m]['chain']['std']:.3g}"); ax.set_xlabel("sample"); ax.set_ylabel("$f_{NL}$"); ax.legend(fontsize=7)
tag = f"nonlinear_box_L{L:.0f}_res{res}_{TR}" + (f"_mask{MASK:g}" if MASK > 0 else "")
save_fig(fig, f"sampler_{tag}")
json.dump(res_out, open(ROOT / "results" / f"sampler_{tag}.json", "w"))
write_note(f"sampler_{tag}", "Stage 2.1: posterior width, linear vs full 2LPT model (small box, production cell size)", lines +
           ["", "If the full-model chain width tracks Tier 1 rather than the linearised Fisher, the quadratic-coupling information is not realised in the posterior."])
