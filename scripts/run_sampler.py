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
from sampler import run_nuts, run_hmc, run_mclmc, summarize_chain

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=80); ap.add_argument("--linear", action="store_true")
ap.add_argument("--tracers", default="QSO+LRG"); ap.add_argument("--sampler", default="nuts")
ap.add_argument("--n_warmup", type=int, default=300); ap.add_argument("--n_samples", type=int, default=2000)
ap.add_argument("--leapfrog", type=int, default=32); ap.add_argument("--fnl_true", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--likelihood", default="gaussian")
ap.add_argument("--max_doublings", type=int, default=8); ap.add_argument("--thin", type=int, default=1)
ap.add_argument("--bias_prior", type=float, default=0.0, help="Gaussian prior width on b2 and bK about fiducial (0 = flat)")
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
def checkpoint(theta, accept, nsteps):
    summ = summarize_chain(theta, lay.names)
    np.savez(outfile, theta=theta, names=np.array(lay.names), accept=accept, n_steps=nsteps, time=time.time() - t1, partial=True,
             schur_sigma=ff.sigma(), args=json.dumps(vars(a)))
    json.dump(dict(tag=tag, args=vars(a), summary=summ, partial=True, n_done=int(len(theta)), schur_sigma=ff.sigma()),
              open(ROOT / "results" / f"chain_{tag}.json", "w"), indent=1)
    log(f"  checkpoint {len(theta)} samples: f_NL std {summ['f_NL']['std']:.2f}, tau {summ['f_NL']['tau']:.1f}")
if a.sampler == "nuts":
    out = run_nuts(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, max_doublings=a.max_doublings, thin=a.thin, checkpoint=checkpoint)
elif a.sampler == "hmc":
    out = run_hmc(logdens, st0, n_leapfrog=a.leapfrog, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, thin=a.thin, checkpoint=checkpoint)
else:
    out = run_mclmc(logdens, st0, n_warmup=a.n_warmup, n_samples=a.n_samples, seed=a.seed, unpack=unpack, log=log, thin=a.thin)
summ = summarize_chain(out["theta"], lay.names)
np.savez(outfile, theta=out["theta"], names=np.array(lay.names), accept=out.get("accept", np.array([])), n_steps=out.get("n_steps", np.array([])),
         time=out["time"], grad_time=time.time() - t1, schur_sigma=ff.sigma(), args=json.dumps(vars(a)))
json.dump(dict(tag=tag, args=vars(a), summary=summ, time=out["time"], schur_sigma=ff.sigma(),
               mean_leapfrog=float(np.mean(out["n_steps"])) if len(out.get("n_steps", [])) else None), open(ROOT / "results" / f"chain_{tag}.json", "w"), indent=1)
print(f"[{tag}] done {out['time']:.0f}s; f_NL: {summ['f_NL']}", flush=True)
