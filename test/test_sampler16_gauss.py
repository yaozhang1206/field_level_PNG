"""Sampler mechanics check with an exactly Gaussian target: linear model, theta fixed, eps free (4096 dims).
Exact posterior: precision A = J^T N^-1 J + I, mean m = A^-1 J^T N^-1 d (dense at 16^3). Compare NUTS chain
means/variances of eps against (m, diag A^-1) and the chain of a few linear projections. Then the multi-mock
f_NL width test: 6 mocks, posterior std averaged, vs the Fisher."""
from _common import *
import numpy as np, jax, jax.numpy as jnp, time, json, os
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from posterior import FieldPosterior
from fisher_field import FieldFisher, sigma_of
from sampler import run_nuts, summarize_chain, tau_int
res = 16; L = 1e4; cosmo = Cosmology(backend="class"); t = "LRG"; n = 3e-4
sh = default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]); shells = [sh]
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda z: np.full_like(z, n))}, ang_mask=None, observer=[L/2, L/2, -200*L])
fm.nbar_cell[0] = jnp.full((res,)*3, n * fm.V_cell); fm.footprint = {t: np.ones((res,)*3, bool)}
lay = ParamLayout(shells); th0 = lay.fiducial(0.0)
eps0 = jax.random.normal(jax.random.PRNGKey(11), (res,)*3, dtype=jnp.float64)
post = FieldPosterior(fm, lay, linear=True)
d = post.mock(eps0, th0, seed=5)
lines = []
# ---- (1) exact Gaussian target: eps only
model = lambda e: post.model(e, th0)
J = np.asarray(jax.jacfwd(model)(eps0).reshape(res**3, res**3)); iv = np.asarray(post.invvar)
A = J.T @ (iv[:, None] * J) + np.eye(res**3); Ainv = np.linalg.inv(A)
m = Ainv @ (J.T @ (iv * np.asarray(d)))
logdens, unpack = post.make_logdensity(d, free=np.array([], dtype=int))
st0 = post.pack(eps0, th0, free=np.array([], dtype=int))
out = run_nuts(logdens, st0, n_warmup=300, n_samples=3000, seed=2, unpack=unpack, log=lambda *a: None,
               record=lambda e, th: np.asarray(e).ravel())
E = np.array(out["records"])                          # (n_samples, 4096)
mean_err = (E.mean(0) - m) / np.sqrt(np.diag(Ainv)); var_ratio = E.var(0) / np.diag(Ainv)
rng = np.random.default_rng(0); P = rng.normal(size=(5, res**3)); P /= np.linalg.norm(P, axis=1, keepdims=True)
proj = E @ P.T; proj_exact_var = np.einsum("ij,jk,ik->i", P, Ainv, P); proj_exact_mean = P @ m
taus = [tau_int(proj[:, i]) for i in range(5)]
lines += [f"- Gaussian target (eps only, 4096 dims), 3000 NUTS samples: mean error / posterior std: rms {np.sqrt(np.mean(mean_err**2)):.3f} (expect ~{1/np.sqrt(3000/np.mean(taus)):.3f} from ESS);",
          f"  variance ratio chain/exact: mean {var_ratio.mean():.3f}, 5-95% [{np.percentile(var_ratio,5):.3f}, {np.percentile(var_ratio,95):.3f}]; random projections: var ratio {np.round(proj.var(0)/proj_exact_var, 3).tolist()}, tau {np.round(taus,1).tolist()}; leapfrog/sample {np.mean(out['n_steps']):.0f}, {out['time']:.0f}s"]
print("\n".join(lines[-2:]), flush=True)
# ---- (2) multi-mock f_NL width
ff = FieldFisher(fm, lay, eps0, th0, linear=True, verbose=False); ff.fisher(); Fex = ff.fisher_marginal(n_probes=16, seed=3)
s_fisher = sigma_of(Fex)
stds, means = [], []
for seed in range(6):
    e = jax.random.normal(jax.random.PRNGKey(100 + seed), (res,)*3, dtype=jnp.float64)
    dd = post.mock(e, th0, seed=200 + seed)
    scale = np.zeros(lay.n); scale[[0, 1]] = 1 / np.sqrt(np.diag(ff.F_tt)[[0, 1]]); free = np.array([0, 1])
    ld, up = post.make_logdensity(dd, theta_scale=scale, free=free)
    o = run_nuts(ld, post.pack(e, th0, scale, free), n_warmup=300, n_samples=1500, seed=seed, unpack=up, log=lambda *a: None)
    s = summarize_chain(o["theta"], lay.names)["f_NL"]; stds.append(s["std"]); means.append(s["mean"])
lines.append(f"- f_NL width over 6 mocks (LRG, b1 free): chain std per mock {np.round(stds,3).tolist()}, mean {np.mean(stds):.3f} ± {np.std(stds)/np.sqrt(6):.3f}; "
             f"exact-marginal Fisher {s_fisher:.3f}; posterior means {np.round(means,2).tolist()} (truth 0; rms {np.sqrt(np.mean(np.square(means))):.3f})")
print(lines[-1], flush=True)
write_note("sampler16_gauss", "Sampler mechanics (exact Gaussian target) and multi-mock $f_{\\rm NL}$ width at $16^3$", lines + ["", "No figure (numerical table)."])
