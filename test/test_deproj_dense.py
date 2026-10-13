"""Stage 4 check: exact template-deprojection Fisher loss (FieldFisher.deproj_loss, CG operators + Woodbury)
against a dense brute-force calculation on a small periodic box with random smooth "templates".
Dense: C = J_e J_e^T + N, dC_i = dJ_i J_e^T + J_e dJ_i^T (autodiff), C_c = C + Lambda T^T T with Lambda -> inf
(explicit projection), F = 1/2 Tr[C^-1 dC_i C^-1 dC_j]."""
from _common import *
import numpy as np, jax, jax.numpy as jnp
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from fisher_field import FieldFisher, sigma_of

res, L = 8, 2000.0
cosmo = Cosmology(backend="class")
t, nb = "LRG", 3e-4
sh = default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]); shells = [sh]
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda zz: np.full_like(zz, nb))}, ang_mask=None,
                  observer=[L / 2, L / 2, -200 * L])
fm.nbar_cell[0] = jnp.full((res,) * 3, nb * fm.V_cell); fm.footprint = {t: np.ones((res,) * 3, bool)}
lay = ParamLayout(shells); th = lay.fiducial(0.0); n = lay.n
eps = jax.random.normal(jax.random.PRNGKey(3), (res,) * 3, dtype=jnp.float64)
ff = FieldFisher(fm, lay, eps, th, rsd=True, linear=False, verbose=False)
N = res ** 3
# smooth random templates (low-k Gaussian fields) times the mean counts
rng = np.random.default_rng(0)
k = np.fft.fftfreq(res) * res
K = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
def smooth():
    f = np.fft.fftn(rng.normal(size=(res,) * 3)) * (K <= 1.5)
    x = np.real(np.fft.ifftn(f)); return (x - x.mean()) / x.std()
T = np.array([smooth().ravel() * nb * fm.V_cell for _ in range(4)])
configs = {"none": [], "T0": [0], "T01": [0, 1], "T0123": [0, 1, 2, 3]}
dF = ff.deproj_loss(T, configs, tol=1e-10, maxiter=2000)

# dense brute force
model = lambda e, th_: fm(e, th_, lay, True)[t].ravel()
fm_fwd = __import__("copy").copy(fm); fm_fwd.requires_jacfwd = True
model_f = lambda e, th_: fm_fwd(e, th_, lay, True)[t].ravel()
Je = np.asarray(jax.jacfwd(model_f, argnums=0)(eps, th).reshape(N, N))
dJ = np.asarray(jax.jacfwd(lambda th_: jax.jacfwd(model_f, argnums=0)(eps, th_).reshape(N, N))(th))   # (N, N, n)
var = np.asarray(1.0 / ff.invvar)
C = Je @ Je.T + np.diag(var)
dC = [dJ[:, :, i] @ Je.T + Je @ dJ[:, :, i].T for i in range(n)]
def Fdense(Cinv):
    return np.array([[0.5 * np.trace(Cinv @ dC[i] @ Cinv @ dC[j]) for j in range(n)] for i in range(n)])
Ci = np.linalg.inv(C)
F0 = Fdense(Ci)
lines = [f"Periodic box $L = {L:.0f}$, ${res}^3$, LRG-like, full 2LPT model linearised at a random field; 4 smooth random templates.", "",
         "| deprojected | $\\Delta F_{f f}$ (CG + Woodbury) | $\\Delta F_{f f}$ (dense) | max rel. diff (all $ij$) | $\\sigma(f_{\\rm NL})$ dense |", "|---|---|---|---|---|"]
ok = True
for c, rows in configs.items():
    if rows:
        Tc = T[rows]; G = Tc @ Ci @ Tc.T
        Cc = Ci - Ci @ Tc.T @ np.linalg.solve(G, Tc @ Ci)
    else:
        Cc = Ci
    dFd = F0 - Fdense(Cc)
    rel = np.max(np.abs(dF[c] - dFd)) / max(np.max(np.abs(dFd)), 1e-30) if rows else np.max(np.abs(dF[c]))
    ok &= rel < 1e-4
    lines.append(f"| {c} | {dF[c][0, 0]:.4e} | {dFd[0, 0]:.4e} | {rel:.1e} | {sigma_of(F0 - dFd):.3f} |")
lines += ["", f"**{'PASS' if ok else 'FAIL'}** (criterion: max relative difference < 1e-4)."]
fig, ax = plt.subplots(figsize=(5, 3.5))
xs = np.arange(len(configs))
ax.plot(xs, [dF[c][0, 0] for c in configs], "o", label="CG + Woodbury")
ax.plot(xs, [(F0 - Fdense(Ci if not r else Ci - Ci @ T[r].T @ np.linalg.solve(T[r] @ Ci @ T[r].T, T[r] @ Ci)))[0, 0] for c, r in configs.items()], "x", ms=10, label="dense")
ax.set_xticks(xs); ax.set_xticklabels(list(configs)); ax.set_ylabel("$\\Delta F_{f_{\\rm NL} f_{\\rm NL}}$"); ax.legend()
save_fig(fig, "deproj_dense")
write_note("deproj_dense", "Stage 4: exact template-deprojection Fisher loss vs dense brute force", lines)
print("\n".join(lines))
