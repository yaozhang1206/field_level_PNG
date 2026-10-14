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
ap.add_argument("--sample_bphi", action="store_true", help="free b_phi, b_phidelta per shell with a 40% Gaussian prior")
ap.add_argument("--mclmc_step_factor", type=float, default=1.0); ap.add_argument("--mclmc_L", type=float, default=None); ap.add_argument("--mclmc_step", type=float, default=None)
ap.add_argument("--resume", action="store_true", help="continue the chain in results/chain_<tag>.npz (position, step size, mass) without warmup")
ap.add_argument("--adapt", default="auto", help="window (BlackJAX run, stores warmup) | loop (same adaptation, Python loop, no storage) | dual (identity mass) | auto (loop for res>=160)")
ap.add_argument("--templates", default="none", help="imaging template amplitudes sampled with flat priors: none | desi (DESI DR1 PNG scheme: per tracer, z-bin, region, feature) | imaging3 | imaging9 (first Stage 4 setup)")
ap.add_argument("--contam", default="none", help="mock contamination: none | linear (random template amplitudes) | wsys (DR1 1/WEIGHT_SYS per DESI z-bin) | wsys_allz (first setup: one map per tracer)")
ap.add_argument("--intercepts", action="store_true", help="DESI scheme: add a free constant per tracer x shell x region (region renormalisation)")
ap.add_argument("--quadratic", default="", help="tracers (e.g. QSO or QSO+LRG) whose templates include all squares and pairwise products per group")
ap.add_argument("--contam_amp", type=float, default=0.02, help="rms of the true template amplitudes for --contam linear")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build()
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
trs = [t for t in fm.tracers if t in a.tracers]
if a.templates != "none" or a.contam == "linear":
    from templates import grid_template_maps, grid_template_compact
    if a.templates in ("desi", "none"):
        tlabels, tcomp = grid_template_compact(fm, trs, intercepts=a.intercepts, quadratic=tuple(a.quadratic.split("+")) if a.quadratic else ())
        lay = ParamLayout(shells, sample_bphi=a.sample_bphi, templates=tlabels)
        fm.set_templates_compact(lay, tcomp)
    else:
        tlabels, tmaps = grid_template_maps(fm, trs, a.templates)
        lay = ParamLayout(shells, sample_bphi=a.sample_bphi, templates=tlabels)
        fm.set_templates(lay, tmaps)
    print(f"[templates] {a.templates}{' + intercepts' if a.intercepts else ''}: {len(tlabels)} amplitudes", flush=True)
else:
    lay = ParamLayout(shells, sample_bphi=a.sample_bphi)
tag = a.tag or f"{a.sampler}_res{a.res}_{a.tracers.replace('+','')}{'_linear' if a.linear else ''}{'_bprior%g' % a.bias_prior if a.bias_prior > 0 else ''}{'_bphi' if a.sample_bphi else ''}{'_sf%g' % a.mclmc_step_factor if a.mclmc_step_factor != 1 else ''}{'_T' + a.templates if a.templates != 'none' else ''}{'_icpt' if a.intercepts else ''}{'_quad' + a.quadratic.replace('+','') if a.quadratic else ''}{'_C' + a.contam if a.contam != 'none' else ''}_fnl{a.fnl_true:g}_s{a.seed}"
th_true = lay.fiducial(a.fnl_true)
if a.contam == "linear":
    a_true = np.random.default_rng(500 + a.seed).normal(0, a.contam_amp, len(lay.templates))
    th_true = th_true.at[lay.t0:].set(a_true)
eps_true = jax.random.normal(jax.random.PRNGKey(1000 + a.seed), (a.res,) * 3, dtype=jnp.float64)
prior_sigma = None
if a.bias_prior > 0 or a.sample_bphi:
    prior_sigma = np.full(lay.n, 1e6)
    if a.bias_prior > 0:
        prior_sigma[[i for i, n in enumerate(lay.names) if n.startswith(("b2_", "bK_"))]] = a.bias_prior
    if a.sample_bphi:
        P = lay.prior(0.4); idx = np.where(np.diag(P) > 0)[0]; prior_sigma[idx] = 1.0 / np.sqrt(np.diag(P)[idx])
post = FieldPosterior(fm, lay, tracers=trs, linear=a.linear, likelihood=a.likelihood, theta_prior_sigma=prior_sigma)
d = post.mock(eps_true, th_true, seed=a.seed)
if a.contam in ("wsys", "wsys_allz"):
    from templates import wsys_contamination, wsys_contamination_desi
    cmap = wsys_contamination_desi if a.contam == "wsys" else wsys_contamination
    mu = post.model(eps_true, th_true)
    cfac = jnp.concatenate([jnp.asarray(cmap(fm, t).ravel()[np.asarray(post.idx[t])]) for t in trs])
    noise = d - mu
    d = mu * cfac + noise
    print(f"[contam] wsys modulation rms {float(jnp.std(cfac)):.3f} (min {float(cfac.min()):.3f}, max {float(cfac.max()):.3f})", flush=True)
# whitening from the Stage-1 Fisher block at the truth
ff = FieldFisher(fm, lay, eps_true, th_true, linear=a.linear, tracers=trs, verbose=False); ff.fisher()
if a.linear:
    free = np.array([0] + [1 + i * lay.per_shell for i in range(len(shells))] + ([1 + i * lay.per_shell + 3 for i in range(len(shells))] if a.sample_bphi else []))
else:
    free = np.array([0] + [1 + i * lay.per_shell + j for i in range(len(shells)) for j in range(lay.per_shell)])
if a.templates != "none":
    free = np.concatenate([free, np.arange(lay.t0, lay.n)])
scale = np.zeros(lay.n); dg = np.diag(ff.F_tt) + (0 if prior_sigma is None else 1 / prior_sigma ** 2); scale[free] = 1 / np.sqrt(np.maximum(dg[free], 1e-12))
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
             step_size=run_params.get("step_size", 0.0), inverse_mass_matrix=run_params.get("imm", np.zeros(0)),
             L=run_params.get("L", 0.0))
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
    resume = None
    if prev is not None:
        resume = dict(position=prev["position"], L=float(prev["L"]), step_size=float(prev["step_size"]), inverse_mass_matrix=prev["inverse_mass_matrix"])
    n_left = a.n_samples - (len(prev["theta"]) if prev is not None else 0)
    if n_left <= 0:
        print(f"[{tag}] already complete", flush=True); n_left = 0
    out = run_mclmc(logdens, st0, n_warmup=a.n_warmup, n_samples=n_left, seed=a.seed + (len(prev["theta"]) if prev is not None else 0),
                    unpack=unpack, log=log, thin=a.thin, L_override=a.mclmc_L, step_override=a.mclmc_step, step_factor=a.mclmc_step_factor,
                    checkpoint=checkpoint, resume=resume, params_out=run_params, checkpoint_every=2000)
    run_params = dict(step_size=out["step_size"], imm=out["inverse_mass_matrix"], L=out["L"])
theta_all = out["theta"] if prev is None else np.concatenate([prev["theta"], out["theta"]])
accept_all = out.get("accept", np.array([])) if prev is None else np.concatenate([prev["accept"], out.get("accept", np.array([]))])
nsteps_all = out.get("n_steps", np.array([])) if prev is None else np.concatenate([prev["n_steps"], out.get("n_steps", np.array([]))])
summ = summarize_chain(theta_all, lay.names)
np.savez(outfile, theta=theta_all, names=np.array(lay.names), accept=accept_all, n_steps=nsteps_all,
         time=out["time"], grad_time=time.time() - t1, schur_sigma=ff.sigma(), args=json.dumps(vars(a)),
         position=np.asarray(out["final_state"].position), step_size=run_params.get("step_size", 0.0), inverse_mass_matrix=run_params.get("imm", np.zeros(0)),
         L=run_params.get("L", 0.0))
json.dump(dict(tag=tag, args=vars(a), summary=summ, time=out["time"], schur_sigma=ff.sigma(), n_total=int(len(theta_all)),
               mean_leapfrog=float(np.mean(out["n_steps"])) if len(out.get("n_steps", [])) else None), open(ROOT / "results" / f"chain_{tag}.json", "w"), indent=1)
print(f"[{tag}] done {out['time']:.0f}s; total samples {len(theta_all)}; f_NL: {summ['f_NL']}", flush=True)
