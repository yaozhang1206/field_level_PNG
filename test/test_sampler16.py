"""Stage 2.1 sampler validation at 16^3 (CPU): linear model, periodic box, Gaussian noise. Posterior sigma(f_NL)
from NUTS (b1 free, b2/bK/bphi fixed) must match the exact marginal from test_schur_dense16 (LRG 0.614, QSO 26.7)."""
from _common import *
import numpy as np, jax, jax.numpy as jnp, time, json, os
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from posterior import FieldPosterior
from fisher_field import FieldFisher, sigma_of
from sampler import run_nuts, summarize_chain
res = 16; L = 1e4; cosmo = Cosmology(backend="class")
NS = int(os.environ.get("NS", 1500)); NW = int(os.environ.get("NW", 300))
lines, results = [], {}
for t, n, exact in (("LRG", 3e-4, 0.614), ("QSO", 2.5e-5, 26.73)):
    sh = default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]); shells = [sh]
    fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda z, n=n: np.full_like(z, n))}, ang_mask=None, observer=[L/2, L/2, -200*L])
    fm.nbar_cell[0] = jnp.full((res,)*3, n * fm.V_cell); fm.footprint = {t: np.ones((res,)*3, bool)}
    lay = ParamLayout(shells); th0 = lay.fiducial(0.0)
    eps0 = jax.random.normal(jax.random.PRNGKey(11), (res,)*3, dtype=jnp.float64)
    post = FieldPosterior(fm, lay, linear=True, likelihood="gaussian")
    d = post.mock(eps0, th0, seed=5)
    ff = FieldFisher(fm, lay, eps0, th0, linear=True, verbose=False); ff.fisher()
    scale = np.zeros(lay.n); scale[[0, 1]] = 1 / np.sqrt(np.diag(ff.F_tt)[[0, 1]])     # whiten f_NL, b1 by F_tt
    free = np.array([0, 1])
    logdens, unpack = post.make_logdensity(d, theta_scale=scale, free=free)
    st0 = post.pack(eps0, th0, scale, free)
    t0 = time.time()
    out = run_nuts(logdens, st0, n_warmup=NW, n_samples=NS, seed=1, unpack=unpack, log=lambda *a: None)
    summ = summarize_chain(out["theta"], lay.names)
    s = summ["f_NL"]
    results[t] = dict(exact=exact, chain=s, time=out["time"], mean_leapfrog=float(np.mean(out["n_steps"])), accept=float(np.nanmean(out["accept"])))
    lines.append(f"- {t} ($\\bar n={n:.1e}$), {NS} samples after {NW} warmup: $\\sigma(f_{{\\rm NL}})$ chain {s['std']:.3f} (exact marginal {exact:.3f}), "
                 f"mean {s['mean']:.3f}, $\\tau_{{\\rm int}}$ {s['tau']:.1f}, ESS {s['ess']:.0f}; mean leapfrog/sample {results[t]['mean_leapfrog']:.0f}, accept {results[t]['accept']:.2f}; {out['time']:.0f}s")
    print(lines[-1], flush=True)
    results[t]["chain_fnl"] = out["theta"][:, 0].tolist()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, t in zip(axes, results):
    x = np.array(results[t]["chain_fnl"]); ax.plot(x, lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-results[t]["exact"], results[t]["exact"], color="g", alpha=0.15, label=f"exact $\\pm\\sigma$ = {results[t]['exact']:.3g}")
    ax.set_title(f"{t}: chain std {results[t]['chain']['std']:.3g}"); ax.set_xlabel("sample"); ax.set_ylabel("$f_{NL}$"); ax.legend(fontsize=7)
save_fig(fig, "sampler16_linear")
json.dump(results, open(ROOT / "results" / "sampler16_linear.json", "w"))
write_note("sampler16_linear", "Stage 2.1: NUTS on the $16^3$ linear box vs exact marginal", lines + ["", "Posterior width must match the exact Gaussian marginal; ESS per gradient sets the cost baseline."])
