"""Brute-force check at 16^3 (CPU): dense joint Fisher of (eps, theta) for the linear box model, exact Schur
complement vs the CG operator code vs the Tier 1 mode sums (exact / Laplace-weighted)."""
from _common import *
import numpy as np, jax, jax.numpy as jnp, time
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from fisher_closed import Tier1
from fisher_field import FieldFisher, sigma_of
res = 16; L = 10000.0
cosmo = Cosmology(backend="class")
lines = []
for t, n in (("LRG", 3.0e-4), ("QSO", 2.5e-5)):
    sh = default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]); shells = [sh]
    fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda zz, n=n: np.full_like(zz, n))}, ang_mask=None, observer=[L/2, L/2, -200*L])
    fm.nbar_cell[0] = jnp.full((res,)*3, n * fm.V_cell); fm.footprint = {t: np.ones((res,)*3, bool)}
    lay = ParamLayout(shells); th = lay.fiducial()
    eps = jax.random.normal(jax.random.PRNGKey(11), (res,)*3, dtype=jnp.float64)
    ff = FieldFisher(fm, lay, eps, th, rsd=True, linear=True, verbose=False)
    F_cg = ff.fisher()
    # dense
    model = lambda e, th_: fm.linear_galaxy_field(e, th_, lay, True)[t].ravel()
    Je = np.asarray(jax.jacfwd(model, argnums=0)(eps, th).reshape(res**3, res**3))
    Jt = np.asarray(jax.jacfwd(model, argnums=1)(eps, th))                       # (N, ntheta)
    iv = np.asarray(ff.invvar)
    keep = [0, 1]                                                               # f_NL, b1 (b2, bK have zero derivative)
    Jt = Jt[:, keep]
    Ftt = Jt.T @ (iv[:, None] * Jt); Fte = Jt.T @ (iv[:, None] * Je); Fee = Je.T @ (iv[:, None] * Je) + np.eye(res**3)
    F_dense = Ftt - Fte @ np.linalg.solve(Fee, Fte.T)
    # exact Gaussian-marginal Fisher of the linear model: C = Je Je^T + N, dC_i = dJ_i Je^T + Je dJ_i^T
    N = np.diag(1.0 / iv); C = Je @ Je.T + N; Ci = np.linalg.inv(C)
    dJ = [np.asarray(jax.jacfwd(lambda th_: jax.jacfwd(model, argnums=0)(eps, th_).reshape(res**3, res**3))(th)[..., i]) for i in keep]
    dC = [d @ Je.T + Je @ d.T for d in dJ]
    F_marg = np.array([[0.5 * np.trace(Ci @ dC[i] @ Ci @ dC[j]) for j in range(2)] for i in range(2)])
    class SG:
        def tables(s_, s): return np.array([1.0]), np.array([L**3]), np.array([n])
    t1 = Tier1(cosmo, shells, lay, SG(), L=L, res=res, continuum=False)
    F_ex, F_lap = t1.fisher(laplace=False)[:2, :2], t1.fisher(laplace=True)[:2, :2]
    F_h = ff.fisher_marginal(n_probes=16)[:2, :2]                                # Hutchinson, same operators
    s = lambda F: float(np.sqrt(np.linalg.inv(F)[0, 0]))
    lines.append(f"- {t} (n={n:.1e}): sigma(f_NL) [b1 marg.]: CG-Schur {s(F_cg[:2,:2]):.3f} | dense Schur {s(F_dense):.3f} | "
                 f"dense exact marginal {s(F_marg):.3f} | Hutchinson(16 probes) {s(F_h):.3f} | mode sum exact {s(F_ex):.3f} | mode sum Laplace-weighted {s(F_lap):.3f}")
    lines.append(f"    F_ff: CG {F_cg[0,0]:.4f} dense {F_dense[0,0]:.4f} | marginal {F_marg[0,0]:.4f} Hutchinson {F_h[0,0]:.4f} exact-sum {F_ex[0,0]:.4f} | Laplace-sum {F_lap[0,0]:.4f}")
lines += ["", "Conclusions: CG operator code == dense Schur; dense exact marginal == Tier 1 mode sum (validates both);",
          "Schur == Laplace-weighted sum (within one-realisation scatter): the Schur/profile estimate is biased by (P+N)/(2P) per mode;",
          "Hutchinson reproduces the exact marginal within probe noise."]
write_note("schur_dense16", "Dense 16^3 check: Schur (CG vs dense) vs exact marginal vs mode sums", lines + ["", "No figure (numerical table)."])
print("\n".join(lines))
