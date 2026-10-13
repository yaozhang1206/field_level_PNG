"""Stage 3: field-level f_NL posterior on Quijote halo catalogues (model-mismatch test of the DR1 model).

    python scripts/run_stage3_halos.py --sample fiducial_HR_z0.5_LRG --ids 0-49 --res 16 --oversample 4 --lap \
        --rsd --sample_noise [--sample_bphi] [--p 1.0] [--n_order 2] [--mock] [--tag T]
        -> results/stage3/<sample>_<tag>.json (one entry per realisation, appended / resumable)

Per realisation (periodic 1 Gpc/h box, one shell at the catalogue redshift, uniform nbar = N / L^3):
  data   halo counts on the res^3 grid: NGP aligned with the model's block-averaged fine lattice (oversample > 1),
         or CIC (oversample = 1, the Stage 1-2 lattice model); redshift space along z with the halo velocities (--rsd).
  model  the DR1 forward model (DISCO-DJ nLPT, Lagrangian bias b1, b2, bK, local PNG with b_phi, b_phidelta from the
         DR1 p value, Kaiser RSD) + optional Laplacian bias b_nabla (--lap); Gaussian likelihood with variance
         A * nbar V_cell, ln A sampled with a flat prior (--sample_noise) or A = 1.
  sampler cold start (eps = 0.1 x a random field, f_NL = +50, b1 from the halo P(k), b2/bK from b1, b_nabla = 0),
         phased NUTS warm-up: (1) eps, ln A; (2) + bias; (3) + f_NL, then sampling.
The ensemble mean of the posterior means is the model-mismatch bias (truth 0 for fiducial, +-100 for LC_p / LC_m).
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
from quijote import quijote_cosmology, cic_mesh, ngp_mesh, redshift_space, linear_bias, L_QUIJOTE  # noqa: E402
from forward_model import ForwardModel, ParamLayout, default_shell  # noqa: E402
from posterior import FieldPosterior  # noqa: E402
from fisher_field import FieldFisher  # noqa: E402
from sampler import run_nuts, summarize_chain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True)
ap.add_argument("--cuts_root", default="/work/hdd/bdne/yzhang116/quijote_cuts")
ap.add_argument("--ids", default="0-9")
ap.add_argument("--res", type=int, default=16)
ap.add_argument("--oversample", type=int, default=4)
ap.add_argument("--n_order", type=int, default=2)
ap.add_argument("--rsd", action="store_true")
ap.add_argument("--lap", action="store_true")
ap.add_argument("--sample_noise", action="store_true")
ap.add_argument("--sample_bphi", action="store_true")
ap.add_argument("--bphi_prior", type=float, default=0.4)
ap.add_argument("--p", type=float, default=None, help="UMF p (default: DR1 value for the tracer, LRG 1.0, QSO 1.6)")
ap.add_argument("--n_warmup", type=int, default=300)
ap.add_argument("--n_phase", type=int, default=150, help="warm-up steps of each of phases 1 and 2")
ap.add_argument("--n_samples", type=int, default=1500)
ap.add_argument("--max_doublings", type=int, default=7)
ap.add_argument("--fnl_start", type=float, default=50.0)
ap.add_argument("--mmin", type=float, default=None, help="extra mass cut applied to the saved cut")
ap.add_argument("--mock", action="store_true", help="replace the halos by a mock drawn from the model (calibration)")
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
z = float(summary["z"])
tracer = summary.get("tracer", "LRG")
tracer = tracer if tracer in ("LRG", "QSO") else "LRG"
p = a.p if a.p is not None else {"LRG": 1.0, "QSO": 1.6}[tracer]
tag = (f"res{a.res}_os{a.oversample}_lpt{a.n_order}" + ("_rsd" if a.rsd else "") + ("_lap" if a.lap else "")
       + ("_noise" if a.sample_noise else "") + ("_bphi" if a.sample_bphi else "") + f"_p{p:g}"
       + (f"_M{a.mmin:.0e}" if a.mmin else "") + (f"_mock{a.fnl_mock:g}" if a.mock else "") + (f"_{a.tag}" if a.tag else ""))
os.makedirs(os.path.join(ROOT, "results", "stage3"), exist_ok=True)
out_path = os.path.join(ROOT, "results", "stage3", f"{a.sample}_{tag}.json")
results = json.load(open(out_path)) if os.path.exists(out_path) else {"args": vars(a), "p": p, "runs": {}}
ids = [i for i in parse_ids(a.ids) if str(i) not in results["runs"]]
print(f"[stage3] {a.sample} z={z} tracer={tracer} p={p} -> {out_path}; {len(ids)} realisations to run", flush=True)

# ---- model ----------------------------------------------------------------------------------------------------------
cosmo = quijote_cosmology()
L, res = L_QUIJOTE, a.res
b1_mean = float(np.mean([r["b1"] for r in summary["realisations"].values()]))
nbar_mean = float(np.mean([r["n"] for r in summary["realisations"].values()]))
shell = default_shell(tracer, z, z, b1_mean, p)
fm = ForwardModel(cosmo, [shell], L=L, res=res, nz_funcs={tracer: lambda zz: np.full_like(zz, nbar_mean)},
                  ang_mask=None, n_order=a.n_order, observer=[L / 2, L / 2, -200 * L], oversample=a.oversample)
fm.footprint = {tracer: np.ones((res,) * 3, bool)}
fm.nbar_cell[0] = jnp.full((res,) * 3, nbar_mean * fm.V_cell)
lay = ParamLayout([shell], sample_bphi=a.sample_bphi, lap=a.lap)
prior_sigma = None
if a.sample_bphi:
    prior_sigma = np.full(lay.n, 1e30)
    prior_sigma[lay.index("bphi", 0)] = a.bphi_prior * abs(shell.bphi)
    prior_sigma[lay.index("bphid", 0)] = a.bphi_prior * max(abs(shell.bphidelta), 1e-3)
post = FieldPosterior(fm, lay, rsd=a.rsd, linear=False, likelihood="gaussian", theta_prior_sigma=prior_sigma)

# parameter scaling from the Fisher block at a random field (sampler preconditioning only)
th_fid = lay.fiducial(0.0)
eps_fid = jax.random.normal(jax.random.PRNGKey(123), (res,) * 3, dtype=jnp.float64)
ff = FieldFisher(fm, lay, eps_fid, th_fid, rsd=a.rsd, linear=False, verbose=False); ff.fisher()
scale = 1.0 / np.sqrt(np.maximum(np.diag(np.asarray(ff.F_tt)), 1e-12))
print(f"[stage3] b1_mean {b1_mean:.3f}, nbar {nbar_mean:.2e}, cell {L / res:.1f} Mpc/h, oversample {a.oversample}; "
      f"theta {lay.names} scale {np.round(scale, 4)}", flush=True)
all_idx = np.arange(lay.n)
bias_idx = all_idx[1:]


def phase(d, eps, theta, lnA, free, n_warm, n_samp, seed):
    """One NUTS run over (eps, theta[free], ln A); returns final (eps, theta, lnA) and the run output."""
    logd, unpack = post.make_logdensity(d, theta_scale=scale, free=free, theta_fid=theta, sample_noise=a.sample_noise)
    st0 = post.pack(eps, theta, scale, free, theta, log_noise=lnA if a.sample_noise else None)
    out = run_nuts(logd, st0, n_warmup=n_warm, n_samples=n_samp, seed=seed, unpack=unpack,
                   max_doublings=a.max_doublings, log=lambda *x: None)
    u = unpack(out["final_state"].position)
    return u[0], u[1], (float(u[2]) if a.sample_noise else 0.0), out


for i in ids:
    t0 = time.time()
    d = np.load(os.path.join(sdir, f"{i}.npz"))
    pos, mass, vel = d["pos"].astype(np.float64), d["mass"], d["vel"].astype(np.float64)
    if a.mmin:
        sel = mass > a.mmin; pos, vel = pos[sel], vel[sel]
    N = len(pos); nbar = N / L ** 3
    fm.nbar_cell[0] = jnp.full((res,) * 3, nbar * fm.V_cell)
    post.var = jnp.full(res ** 3, nbar * fm.V_cell); post.invvar = 1.0 / post.var
    b1_pk = float(linear_bias(pos, cosmo, z, L)[0])         # real-space P(k) bias: the starting value (a P(k) fit in DR1)
    if a.mock:
        eps_true = jax.random.normal(jax.random.PRNGKey(1000 + i), (res,) * 3, dtype=jnp.float64)
        th_true = lay.fiducial(a.fnl_mock).at[1].set(b1_pk)
        data = post.mock(eps_true, th_true, seed=2000 + i)
    else:
        if a.rsd:
            pos = redshift_space(pos, vel, z, cosmo, L, axis=2)
        grid = ngp_mesh(pos, L, res, shift=L / (2 * a.oversample * res)) if a.oversample > 1 else cic_mesh(pos, L, res)
        data = jnp.asarray(grid.ravel())
    # cold start
    s0 = default_shell(tracer, z, z, b1_pk, p)
    th = lay.fiducial(a.fnl_start)
    th = th.at[lay.index("b1", 0)].set(b1_pk).at[lay.index("b2", 0)].set(s0.b2).at[lay.index("bK", 0)].set(s0.bK)
    if a.sample_bphi:
        th = th.at[lay.index("bphi", 0)].set(shell.bphi).at[lay.index("bphid", 0)].set(shell.bphidelta)
    eps = 0.1 * jax.random.normal(jax.random.PRNGKey(5000 + i), (res,) * 3, dtype=jnp.float64)
    lnA = 0.0
    eps, th, lnA, _ = phase(data, eps, th, lnA, np.array([], dtype=int), a.n_phase, 20, a.seed + i)
    eps, th, lnA, _ = phase(data, eps, th, lnA, bias_idx, a.n_phase, 20, a.seed + i + 1)
    eps, th, lnA, out = phase(data, eps, th, lnA, all_idx, a.n_warmup, a.n_samples, a.seed + i + 2)
    names = lay.names + (["lnA"] if a.sample_noise else [])
    summ = summarize_chain(out["theta"], names)
    results["runs"][str(i)] = {"N": N, "nbar": nbar, "b1_pk": b1_pk, "summary": summ,
                               "fnl_chain": np.asarray(out["theta"][:, 0]).astype(np.float32).tolist(),
                               "time": time.time() - t0, "mean_leapfrog": float(np.mean(out["n_steps"])),
                               "accept": float(np.nanmean(out["accept"]))}
    json.dump(results, open(out_path, "w"))
    s = summ["f_NL"]
    extra = "  ".join(f"{n.split('_')[0]} {summ[n]['mean']:.3g}±{summ[n]['std']:.2g}" for n in names[1:])
    print(f"[stage3] {i}: N={N} b1(Pk)={b1_pk:.3f} | f_NL = {s['mean']:+.1f} +- {s['std']:.1f} (tau {s['tau']:.1f}, "
          f"ESS {s['ess']:.0f}) | {extra}  [{time.time() - t0:.0f}s]", flush=True)

means = np.array([r["summary"]["f_NL"]["mean"] for r in results["runs"].values()])
stds = np.array([r["summary"]["f_NL"]["std"] for r in results["runs"].values()])
if len(means) > 1:
    print(f"[stage3] {len(means)} realisations: <f_NL> = {means.mean():+.2f} +- {means.std(ddof=1) / np.sqrt(len(means)):.2f} "
          f"(scatter of means {means.std(ddof=1):.1f}, mean posterior std {stds.mean():.1f})")
