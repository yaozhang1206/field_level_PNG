"""Stage 2 production sampler on the DR1 configuration.
Usage: python scripts/run_sampler.py --res 80 [--linear] [--tracers QSO+LRG] [--sampler nuts|hmc|mclmc]
       --n_warmup 300 --n_samples 2000 [--leapfrog 32] [--fnl_true 0] [--seed 0] [--tag name]"""
import sys, os, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "test"))
import numpy as np, jax, jax.numpy as jnp
from stage1_config import build
from geometry import dr1_angular_mask
from forward_model import ForwardModel, ParamLayout
from posterior import FieldPosterior
from fisher_field import FieldFisher
from sampler import run_nuts, run_nuts_da, run_nuts_window_loop, run_hmc, run_mclmc, summarize_chain

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=80); ap.add_argument("--linear", action="store_true")
ap.add_argument("--tracers", default="QSO+LRG"); ap.add_argument("--sampler", default="nuts")
ap.add_argument("--n_warmup", type=int, default=300); ap.add_argument("--n_samples", type=int, default=2000)
ap.add_argument("--leapfrog", type=int, default=32); ap.add_argument("--fnl_true", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--likelihood", default="gaussian")
ap.add_argument("--max_doublings", type=int, default=8); ap.add_argument("--thin", type=int, default=1)
ap.add_argument("--bias_prior", type=float, default=0.0, help="Gaussian prior width on b2 and bK about fiducial (0 = flat)")
ap.add_argument("--resume", action="store_true", help="continue the chain in results/chain_<tag>.npz (position, step size, mass) without warmup")
ap.add_argument("--adapt", default="auto", help="window (BlackJAX run, stores warmup) | loop (same adaptation, Python loop, no storage) | dual (identity mass) | auto (loop for res>=160)")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build()
lay = ParamLayout(shells)
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
trs = [t for t in fm.tracers if t in a.tracers]
tag = a.tag or f"{a.sampler}_res{a.res}_{a.tracers.replace('+','')}{'_linear' if a.linear else ''}{'_bprior%g' % a.bias_prior if a.bias_prior > 0 else ''}_fnl{a.fnl_true:g}_s{a.seed}"
th_true = lay.fiducial(a.fnl_true)
eps_true = jax.random.normal(jax.random.PRNGKey(1000 + a.seed), (a.res,) * 3, dtype=jnp.float64)
prior_sigma = None
if a.bias_prior > 0:
    prior_sigma = np.full(lay.n, 1e6); prior_sigma[[i for i, n in enumerate(lay.names) if n.startswith(("b2_", "bK_"))]] = a.bias_prior
post = FieldPosterior(fm, lay, tracers=trs, linear=a.linear, likelihood=a.likelihood, theta_prior_sigma=prior_sigma)
d = post.mock(eps_true, th_true, seed=a.seed)
# whitening from the Stage-1 Fisher block at the truth
ff = FieldFisher(fm, lay, eps_true, th_true, linear=a.linear, tracers=trs, verbose=False); ff.fisher()
if a.linear:
    free = np.array([0] + [1 + i * lay.per_shell for i in range(len(shells))])
else:
    free = np.array([0] + [1 + i * lay.per_shell + j for i in range(len(shells)) for j in range(3)])
scale = np.zeros(lay.n); dg = np.diag(ff.F_tt); scale[free] = 1 / np.sqrt(np.maximum(dg[free], 1e-12))
logdens, unpack = post.make_logdensity(d, theta_scale=scale, free=free)
st0 = post.pack(eps_true, th_true, scale, free)       # start at the truth (we measure mixing, not burn-in)
outfile = ROOT / "results" / f"chain_{tag}.npz"
print(f"[{tag}] state dim {st0.size}, free params {[lay.names[i] for i in free]}, Schur sigma(f_NL) {ff.sigma():.2f}", flush=True)
t0 = time.time(); g = jax.grad(logdens)(st0); g.block_until_ready(); t1 = time.time(); g = jax.grad(logdens)(st0); g.block_until_ready()
print(f"[{tag}] gradient: {t1-t0:.1f}s compile, {time.time()-t1:.3f}s per evaluation", flush=True)
log = lambda *x: print(f"[{tag}]", *x, flush=True)
prev = None
if a.resume and outfile.exists():
    prev = dict(np.load(outfile, allow_pickle=True))
    if "position" not in prev:
        raise SystemExit("cannot resume: no saved position in " + str(outfile))
    print(f"[{tag}] resuming from {len(prev['theta'])} samples", flush=True)
def checkpoint(theta, accept, nsteps, position=None):
    if prev is not None:
        theta = np.concatenate([prev["theta"], theta]); accept = np.concatenate([prev["accept"], accept]); nsteps = np.concatenate([prev["n_steps"], nsteps])
    summ = summarize_chain(theta, lay.names)
    np.savez(outfile, theta=theta, names=np.array(lay.names), accept=accept, n_steps=nsteps, time=time.time() - t1, partial=True,
             schur_sigma=ff.sigma(), args=json.dumps(vars(a)), position=position if position is not None else np.zeros(0),
             step_size=run_params.get("step_size", 0.0), inverse_mass_matrix=run_params.get("imm", np.zeros(0)))
    json.dump(dict(tag=tag, args=vars(a), summary=summ, partial=True, n_done=int(len(theta)), schur_sigma=ff.sigma()),
              open(ROOT / "results" / f"chain_{tag}.json", "w"), indent=1)
    log(f"  checkpoint {len(theta)} samples: f_NL std {summ['f_NL']['std']:.2f}, tau {summ['f_NL']['tau']:.1f}")
adapt = a.adapt if a.adapt != "auto" else ("loop" if a.res >= 160 else "window")
run_params = {}
if a.sampler == "nuts" and adapt == "loop":
    resume = None
    if prev is not None:
        resume = dict(position=prev["position"], step_size=float(prev["step_size"]), inverse_mass_matrix=prev["inverse_mass_matrix"])
        run_params = dict(step_size=resume["step_size"], imm=resume["inverse_mass_matrix"])
    out = run_nuts_window_loop(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed + (len(prev["theta"]) if prev is not None else 0),
                               unpack=unpack, log=log, max_doublings=a.max_doublings, thin=a.thin, checkpoint=checkpoint, resume=resume)
    run_params = dict(step_size=out["step_size"], imm=out["inverse_mass_matrix"])
elif a.sampler == "nuts" and adapt == "dual":
    out = run_nuts_da(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, max_doublings=a.max_doublings, thin=a.thin, checkpoint=checkpoint)
elif a.sampler == "nuts":
    out = run_nuts(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, max_doublings=a.max_doublings, thin=a.thin, checkpoint=checkpoint)
elif a.sampler == "hmc":
    out = run_hmc(logdens, st0, n_leapfrog=a.leapfrog, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, thin=a.thin, checkpoint=checkpoint)
else:
    out = run_mclmc(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, thin=a.thin)
theta_all = out["theta"] if prev is None else np.concatenate([prev["theta"], out["theta"]])
accept_all = out.get("accept", np.array([])) if prev is None else np.concatenate([prev["accept"], out.get("accept", np.array([]))])
nsteps_all = out.get("n_steps", np.array([])) if prev is None else np.concatenate([prev["n_steps"], out.get("n_steps", np.array([]))])
summ = summarize_chain(theta_all, lay.names)
np.savez(outfile, theta=theta_all, names=np.array(lay.names), accept=accept_all, n_steps=nsteps_all,
         time=out["time"], grad_time=time.time() - t1, schur_sigma=ff.sigma(), args=json.dumps(vars(a)),
         position=np.asarray(out["final_state"].position), step_size=run_params.get("step_size", 0.0), inverse_mass_matrix=run_params.get("imm", np.zeros(0)))
json.dump(dict(tag=tag, args=vars(a), summary=summ, time=out["time"], schur_sigma=ff.sigma(), n_total=int(len(theta_all)),
               mean_leapfrog=float(np.mean(out["n_steps"])) if len(out.get("n_steps", [])) else None), open(ROOT / "results" / f"chain_{tag}.json", "w"), indent=1)
print(f"[{tag}] done {out['time']:.0f}s; total samples {len(theta_all)}; f_NL: {summ['f_NL']}", flush=True)
