"""Stage 3: field-level f_NL posterior on Quijote halo catalogues (model-mismatch test).

    python scripts/run_stage3_halos.py --sample fiducial_HR_z0.5_LRG --ids 0-49 --res 16 [--n_order 2] [--rsd]
        [--sampler nuts|mclmc] [--n_warmup 300] [--n_samples 1500] [--sample_bphi] [--p 1.0] [--likelihood gaussian]
        [--mmin M] [--mock]  -> results/stage3/<sample>_<tag>.json (one entry per realisation, appended/resumable)

Per realisation: the halos (real space, periodic 1 Gpc/h box) are CIC-assigned to a res^3 grid; the forward model
(DISCO-DJ nLPT + Lagrangian bias + local PNG, same code as Stages 1-2) is set up at the Quijote cosmology with one
shell at the catalogue redshift, uniform nbar = N / L^3; the posterior over (eps, f_NL, b1, b2, bK [, bphi, bphidelta])
is sampled from eps = 0 (warmup discards the burn-in). --mock replaces the halo data by a mock drawn from the model at
the sample's mean bias (calibration of the width and of any sampler bias). The ensemble mean of the posterior means
over realisations is the model-mismatch bias Delta f_NL (truth: 0 for fiducial, +-100 for LC_p/LC_m).
"""
import argparse
import json
import os
import sys
import time
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from quijote import quijote_cosmology, cic_mesh, linear_bias, L_QUIJOTE  # noqa: E402
from forward_model import ForwardModel, ParamLayout, default_shell  # noqa: E402
from posterior import FieldPosterior  # noqa: E402
from fisher_field import FieldFisher  # noqa: E402
from sampler import run_nuts, run_mclmc, summarize_chain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True)
ap.add_argument("--cuts_root", default="/work/hdd/bdne/yzhang116/quijote_cuts")
ap.add_argument("--ids", default="0-9")
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--n_order", type=int, default=2)
ap.add_argument("--rsd", action="store_true")
ap.add_argument("--sampler", default="nuts")
ap.add_argument("--n_warmup", type=int, default=300)
ap.add_argument("--n_samples", type=int, default=1500)
ap.add_argument("--max_doublings", type=int, default=7)
ap.add_argument("--sample_bphi", action="store_true")
ap.add_argument("--bphi_prior", type=float, default=0.4)
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--likelihood", default="gaussian")
ap.add_argument("--mmin", type=float, default=None, help="extra mass cut applied to the saved cut")
ap.add_argument("--mock", action="store_true")
ap.add_argument("--fnl_mock", type=float, default=0.0)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            x, y = part.split("-"); out += list(range(int(x), int(y) + 1))
        else:
            out.append(int(part))
    return out


sdir = os.path.join(a.cuts_root, a.sample)
summary = json.load(open(os.path.join(sdir, "summary.json")))
z = float(summary["z"]); tracer = summary.get("tracer", "LRG")
tag = f"res{a.res}_lpt{a.n_order}" + ("_rsd" if a.rsd else "") + (f"_M{a.mmin:.0e}" if a.mmin else "") + \
      ("_bphi" if a.sample_bphi else "") + f"_p{a.p:g}" + (f"_{a.sampler}" if a.sampler != "nuts" else "") + \
      (f"_mock{a.fnl_mock:g}" if a.mock else "") + (f"_{a.tag}" if a.tag else "")
os.makedirs(os.path.join(ROOT, "results", "stage3"), exist_ok=True)
out_path = os.path.join(ROOT, "results", "stage3", f"{a.sample}_{tag}.json")
results = json.load(open(out_path)) if os.path.exists(out_path) else {"args": vars(a), "sample": summary | {"realisations": None}, "runs": {}}
ids = [i for i in parse_ids(a.ids) if str(i) not in results["runs"]]
print(f"[stage3] {a.sample} z={z} -> {out_path}; {len(ids)} realisations to run", flush=True)

# ---- model at the Quijote cosmology, one shell, periodic box, uniform nbar ----------------------------------------
cosmo = quijote_cosmology()
L, res = L_QUIJOTE, a.res
# mean bias of the sample (fiducial for the parameter scaling and the mock); refined per realisation below
b1_mean = float(np.mean([r["b1"] for r in summary["realisations"].values()]))
shell = default_shell(tracer, z, z, b1_mean, a.p)
shells = [shell]
nbar_mean = float(np.mean([r["n"] for r in summary["realisations"].values()]))
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={tracer: lambda zz: np.full_like(zz, nbar_mean)},
                  ang_mask=None, n_order=a.n_order, observer=[L / 2, L / 2, -200 * L])
lay = ParamLayout(shells, sample_bphi=a.sample_bphi)
th_fid = lay.fiducial(0.0)
ones = np.ones((res,) * 3, bool)
fm.footprint = {tracer: ones}


def set_nbar(n):
    fm.nbar_cell[0] = jnp.asarray(np.full((res,) * 3, n * fm.V_cell))


set_nbar(nbar_mean)
prior_sigma = None
if a.sample_bphi:
    prior_sigma = np.full(lay.n, np.inf)
    prior_sigma[4] = a.bphi_prior * abs(shell.bphi); prior_sigma[5] = a.bphi_prior * abs(shell.bphidelta)
post = FieldPosterior(fm, lay, rsd=a.rsd, linear=False, likelihood=a.likelihood,
                      theta_prior_sigma=None if prior_sigma is None else np.where(np.isinf(prior_sigma), 1e30, prior_sigma))
# parameter scaling from the Fisher block at a random fiducial field
eps_fid = jax.random.normal(jax.random.PRNGKey(123), (res,) * 3, dtype=jnp.float64)
ff = FieldFisher(fm, lay, eps_fid, th_fid, rsd=a.rsd, linear=False, verbose=False)
ff.fisher(); F_tt = np.asarray(ff.F_tt)
free = np.arange(lay.n)
scale = 1.0 / np.sqrt(np.maximum(np.diag(F_tt), 1e-12))
print(f"[stage3] b1_mean {b1_mean:.3f}, nbar {nbar_mean:.2e}, cell {L / res:.1f} Mpc/h; theta scale {np.round(scale, 3)}", flush=True)

for i in ids:
    t0 = time.time()
    d = np.load(os.path.join(sdir, f"{i}.npz"))
    pos, mass = d["pos"].astype(np.float64), d["mass"]
    if a.mmin:
        pos = pos[mass > a.mmin]
    N = len(pos)
    nbar = N / L ** 3
    set_nbar(nbar)
    post.var = jnp.full(res ** 3, nbar * fm.V_cell); post.invvar = 1.0 / post.var
    if a.mock:
        eps_true = jax.random.normal(jax.random.PRNGKey(1000 + i), (res,) * 3, dtype=jnp.float64)
        data = post.mock(eps_true, lay.fiducial(a.fnl_mock), seed=2000 + i)
    else:
        data = jnp.asarray(cic_mesh(pos, L, res).ravel())
    logdens, unpack = post.make_logdensity(data, theta_scale=scale, free=free, theta_fid=th_fid)
    st0 = post.pack(jnp.zeros((res,) * 3), th_fid, scale, free, th_fid)
    if a.sampler == "mclmc":
        out = run_mclmc(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed + i, unpack=unpack,
                        log=lambda *x: None, step_factor=0.5)
    else:
        out = run_nuts(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed + i, unpack=unpack,
                       max_doublings=a.max_doublings, log=lambda *x: None)
    summ = summarize_chain(out["theta"], lay.names)
    b1_meas = linear_bias(pos, cosmo, z, L)[0] if not a.mock else b1_mean
    results["runs"][str(i)] = {"N": N, "nbar": nbar, "b1_pk": float(b1_meas), "summary": summ,
                               "fnl_chain": np.asarray(out["theta"][:, 0]).astype(np.float32).tolist(),
                               "time": time.time() - t0, "mean_leapfrog": float(np.mean(out.get("n_steps", [0])))}
    json.dump(results, open(out_path, "w"))
    s = summ["f_NL"]
    print(f"[stage3] {i}: N={N} b1(Pk)={b1_meas:.3f} | f_NL = {s['mean']:+.1f} +- {s['std']:.1f} (tau {s['tau']:.1f}, ESS {s['ess']:.0f}) "
          f"b1 = {summ[lay.names[1]]['mean']:.3f} +- {summ[lay.names[1]]['std']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

means = np.array([r["summary"]["f_NL"]["mean"] for r in results["runs"].values()])
stds = np.array([r["summary"]["f_NL"]["std"] for r in results["runs"].values()])
print(f"[stage3] {len(means)} realisations: <f_NL> = {means.mean():+.2f} +- {means.std(ddof=1) / np.sqrt(len(means)):.2f} "
      f"(scatter of means {means.std(ddof=1):.1f}, mean posterior std {stds.mean():.1f})")
