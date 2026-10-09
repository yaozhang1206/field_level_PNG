"""Stage 2.0 checks at 16^3 (CPU): (1) grad of the log-posterior vs finite differences in eps and theta;
(2) with noise-free data d = M(truth), -d^2 ln p / d theta^2 at the truth equals F_theta,theta = J^T N^-1 J
exactly (Gaussian likelihood); (3) Poisson mock + likelihood run and gradient check."""
from _common import *
import numpy as np, jax, jax.numpy as jnp, time
from cosmology import Cosmology
from stage1_config import B1_OF_Z, P_UMF
from forward_model import ForwardModel, ParamLayout, default_shell
from posterior import FieldPosterior
from fisher_field import FieldFisher
res = 16; L = 1e4; cosmo = Cosmology(backend="class")
shells = [default_shell(t, 1.0, 1.0, float(B1_OF_Z[t](1.0)), P_UMF[t]) for t in ("QSO", "LRG")]
nbar = {"QSO": 2.5e-5, "LRG": 3e-4}
fm = ForwardModel(cosmo, shells, L=L, res=res, nz_funcs={t: (lambda z, n=nbar[t]: np.full_like(z, n)) for t in nbar}, ang_mask=None)
for i, s in enumerate(shells): fm.nbar_cell[i] = jnp.full((res,) * 3, nbar[s.tracer] * fm.V_cell)
fm.footprint = {t: np.ones((res,) * 3, bool) for t in fm.tracers}
lay = ParamLayout(shells); th0 = lay.fiducial(0.0)
eps0 = jax.random.normal(jax.random.PRNGKey(3), (res,) * 3, dtype=jnp.float64)
lines = []
for lik in ("gaussian", "poisson"):
    post = FieldPosterior(fm, lay, likelihood=lik)
    d = post.mock(eps0, th0, seed=1)
    lp = lambda e, t: post.log_post(e, t, d)
    g_eps, g_th = jax.grad(lp, argnums=(0, 1))(eps0, th0)
    # finite differences along random directions
    key = jax.random.PRNGKey(7); v = jax.random.normal(key, eps0.shape, dtype=jnp.float64); v /= jnp.linalg.norm(v)
    u = jnp.zeros(lay.n).at[0].set(1.0)                      # f_NL direction
    h = 1e-4
    fd_eps = (lp(eps0 + h * v, th0) - lp(eps0 - h * v, th0)) / (2 * h)
    fd_th = (lp(eps0, th0 + h * u) - lp(eps0, th0 - h * u)) / (2 * h)
    ad_eps, ad_th = float(jnp.vdot(g_eps, v)), float(jnp.vdot(g_th, u))
    lines.append(f"- {lik}: mock counts mean {float(d.mean()):.2f}; grad check eps: AD {ad_eps:.6e} vs FD {float(fd_eps):.6e} (rel {abs(ad_eps-float(fd_eps))/abs(float(fd_eps)):.1e}); "
                 f"f_NL: AD {ad_th:.6e} vs FD {float(fd_th):.6e} (rel {abs(ad_th-float(fd_th))/abs(float(fd_th)):.1e})")
    if lik == "gaussian":
        d0 = post.model(eps0, th0)                           # noise-free data
        H = -jax.hessian(lambda t: post.log_post(eps0, t, d0))(th0)
        ff = FieldFisher(fm, lay, eps0, th0, verbose=False); ff.fisher()
        Ftt = ff.F_tt
        keep = np.where(np.diag(Ftt) > 1e-12)[0]
        rel = np.abs(np.asarray(H)[np.ix_(keep, keep)] - Ftt[np.ix_(keep, keep)]) / np.abs(Ftt[np.ix_(keep, keep)]).max()
        lines.append(f"- Gaussian, noise-free data: max |Hessian - F_tt| / max|F_tt| = {rel.max():.2e} (expect ~1e-8)")
        # whitened-theta logdensity (BlackJAX interface)
        scale = 1.0 / np.sqrt(np.maximum(np.diag(Ftt), 1e-12))
        logdens, unpack = post.make_logdensity(d, theta_scale=scale)
        st = post.pack(eps0, th0, scale); t0 = time.time(); val = float(logdens(st)); g = jax.grad(logdens)(st); t1 = time.time() - t0
        lines.append(f"- whitened logdensity on flat state (n={st.size}): value {val:.3f}, grad norm {float(jnp.linalg.norm(g)):.3e}, {t1:.1f}s incl. compile")
write_note("posterior16", "Stage 2.0: log-posterior gradient and Hessian checks ($16^3$)", lines + ["", "No figure (numerical table)."])
print("\n".join(lines))
