"""Stage 3 diagnostic: compare the gridded halo field of one realisation with the forward model at the measured bias
(mean, variance, P(k) on the analysis grid), and evaluate the log-posterior along f_NL and b1 at a Wiener-like eps.
    python scripts/stage3_diag_pk.py --sample fiducial_HR_z0.5_LRG --id 0 --res 16
"""
import argparse, os, sys, json
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import jax, jax.numpy as jnp
from quijote import quijote_cosmology, cic_mesh, L_QUIJOTE
from forward_model import ForwardModel, ParamLayout, default_shell
from posterior import FieldPosterior

ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--id", type=int, default=0)
ap.add_argument("--res", type=int, default=16); ap.add_argument("--p", type=float, default=1.0); a = ap.parse_args()
L, res = L_QUIJOTE, a.res
sdir = f"/work/hdd/bdne/yzhang116/quijote_cuts/{a.sample}"; summ = json.load(open(f"{sdir}/summary.json")); z = summ["z"]
b1 = summ["realisations"][str(a.id)]["b1"]
d = np.load(f"{sdir}/{a.id}.npz"); pos = d["pos"].astype(float); N = len(pos); nbar = N / L ** 3
c = quijote_cosmology()
sh = default_shell("LRG", z, z, b1, a.p)
fm = ForwardModel(c, [sh], L=L, res=res, nz_funcs={"LRG": lambda zz: np.full_like(zz, nbar)}, observer=[L / 2, L / 2, -200 * L])
fm.footprint = {"LRG": np.ones((res,) * 3, bool)}; fm.nbar_cell[0] = jnp.full((res,) * 3, nbar * fm.V_cell)
lay = ParamLayout([sh]); th = lay.fiducial(0.0)
post = FieldPosterior(fm, lay, rsd=False)
data = cic_mesh(pos, L, res)
print(f"data: mean {data.mean():.2f} (expected {nbar * fm.V_cell:.2f}), var/mean {data.var() / data.mean():.2f}, min {data.min():.0f}")


def pk(field, nb=6):
    f = np.fft.rfftn(field / field.mean() - 1) / res ** 3; kf = 2 * np.pi / L
    kx = np.fft.fftfreq(res, 1 / res) * kf; kz = np.fft.rfftfreq(res, 1 / res) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij"); K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2); P = np.abs(f) ** 2 * L ** 3
    edges = np.linspace(kf, np.pi * res / L, nb + 1); idx = np.digitize(K.ravel(), edges) - 1; ok = (idx >= 0) & (idx < nb)
    return (np.array([K.ravel()[ok][idx[ok] == j].mean() for j in range(nb)]),
            np.array([P.ravel()[ok][idx[ok] == j].mean() for j in range(nb)]))


k, Pd = pk(data)
mus = [np.asarray(post.model(jax.random.normal(jax.random.PRNGKey(s), (res,) * 3, dtype=jnp.float64), th)).reshape((res,) * 3) for s in range(8)]
Pm = np.mean([pk(m)[1] for m in mus], axis=0)
Plin = np.asarray(c.P_lin(k, z))
print("model fields: mean %.2f, min %.2f" % (np.mean([m.mean() for m in mus]), np.min([m.min() for m in mus])))
print("   k       P_data - 1/n     <P_model>(8 eps)    b1^2 P_lin")
for j in range(len(k)):
    print(f"{k[j]:.4f}  {Pd[j] - 1 / nbar:12.0f}  {Pm[j]:14.0f}  {b1 ** 2 * Plin[j]:12.0f}")
# log posterior along f_NL and b1 with eps optimised (MAP in eps) at each theta: short L-BFGS
import jax.scipy.optimize as jso
dvec = jnp.asarray(data.ravel())
def neg_lp(eps_flat, theta):
    return -post.log_post(eps_flat.reshape((res,) * 3), theta, dvec)
g = jax.jit(jax.value_and_grad(neg_lp))
def map_eps(theta, eps0=None, n=300, lr=None):
    import optax
    eps = jnp.zeros(res ** 3) if eps0 is None else eps0
    opt = optax.adam(0.05); st = opt.init(eps)
    for _ in range(n):
        v, gr = g(eps, theta); up, st = opt.update(gr, st); eps = optax.apply_updates(eps, up)
    return eps, float(v)
try:
    eps_map, v0 = map_eps(th, n=400)
    print(f"MAP eps at fiducial theta: -log post {v0:.1f}; |eps| rms {float(jnp.std(eps_map)):.3f}")
    for fnl in [-200, -100, -50, 0, 50, 100, 200, 400]:
        t = th.at[0].set(fnl); e, v = map_eps(t, eps_map, n=150); print(f"  f_NL={fnl:+5d}: profile -log post {v:.1f}")
    for b in [1.0, 1.3, 1.6, b1, 2.2, 2.6]:
        t = th.at[1].set(b); e, v = map_eps(t, eps_map, n=150); print(f"  b1={b:.2f}: profile -log post {v:.1f}")
except Exception as ex:
    print("profile step skipped:", ex)
