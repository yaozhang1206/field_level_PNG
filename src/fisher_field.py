"""Field-level Fisher by Schur complement over the white-noise field (Stage 1 Part B).

Joint Fisher of (eps, theta) for Gaussian noise with variance sigma^2(x) = <N(x)> (1/sigma^2 = 0 off footprint):
    F = J^T N^-1 J + P,   J = dM/d(eps, theta),   P = I on eps (white-noise prior) + theta prior
    F_marg = F_tt - F_te F_ee^-1 F_et,   sigma(f_NL) = sqrt[(F_marg^-1)_00]
F_ee = J_e^T N^-1 J_e + I is never formed; (F_ee)^-1 F_et is obtained by conjugate gradient with the matvec
    v -> vjp_eps( jvp_eps(v) / sigma^2 ) + v.

NOTE (documented in the Stage-1 memo): this is the curvature of the *joint* (eps, theta) Gaussian approximation,
i.e. the profile-likelihood curvature. For a linear model it differs from the exact marginal (Gaussian-field)
Fisher mode by mode by the factor (P + N) / (2 P): the Schur result is tighter in the shot-noise dominated
regime and looser (by up to 2 in F) in the sample-variance dominated regime. The closed-form Tier 1 in
fisher_closed.py can be evaluated with either weighting (laplace=True reproduces the Schur expectation).
"""
from __future__ import annotations

import time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.sparse.linalg import cg  # noqa: E402

from forward_model import ForwardModel, ParamLayout  # noqa: E402


_OPS_CACHE = {}


def _get_ops(fm, layout, tracers, linear, rsd, idx):
    """Jitted operators compiled once per (model, layout, tracer set): all take (eps, theta[, invvar]) as
    arguments so that new realisations / fiducials never trigger recompilation (compiled 2LPT+scatter
    programs are large and accumulate in host memory otherwise)."""
    key = (id(fm), id(layout), tuple(tracers), bool(linear), bool(rsd))
    if key in _OPS_CACHE:
        return _OPS_CACHE[key]
    import copy
    fm_fwd = copy.copy(fm); fm_fwd.requires_jacfwd = True
    fm_bwd = copy.copy(fm); fm_bwd.requires_jacfwd = False

    def make_model(f):
        def model(eps, theta):
            out = f.linear_galaxy_field(eps, theta, layout, rsd) if linear else f(eps, theta, layout, rsd)
            return jnp.concatenate([out[t].ravel()[idx[t]] for t in tracers])
        return model
    m_fwd, m_bwd = make_model(fm_fwd), make_model(fm_bwd)
    n = layout.n
    ops = {}
    ops["model"] = jax.jit(m_bwd)
    ops["jvp"] = jax.jit(lambda eps, th, v: jax.jvp(lambda e: m_fwd(e, th), (eps,), (v,))[1])
    ops["vjp"] = jax.jit(lambda eps, th, w: jax.vjp(lambda e: m_bwd(e, th), eps)[1](w)[0])
    ops["jvp_theta"] = jax.jit(lambda eps, th, e_i: jax.jvp(lambda t: m_fwd(eps, t), (th,), (e_i,))[1])
    # second-order: dJ_i v and dJ_i^T w along a theta direction e_i (array argument -> one program)
    ops["dJ_v"] = jax.jit(lambda eps, th, v, e_i: jax.jvp(
        lambda t: jax.jvp(lambda e: m_fwd(e, t), (eps,), (v,))[1], (th,), (e_i,))[1])
    ops["dJT_w"] = jax.jit(lambda eps, th, w, e_i: jax.jvp(
        lambda t: jax.vjp(lambda e: m_bwd(e, t), eps)[1](w)[0], (th,), (e_i,))[1])

    def matvec_schur(eps, th, invvar, v):          # (J^T N^-1 J + I) v
        return ops["vjp"](eps, th, invvar * ops["jvp"](eps, th, v)) + v
    def matvec_C(eps, th, var, w):                 # (J J^T + N) w
        return ops["jvp"](eps, th, ops["vjp"](eps, th, w)) + var * w
    ops["solve_schur"] = jax.jit(lambda eps, th, invvar, b, tol, maxiter: cg(
        lambda v: matvec_schur(eps, th, invvar, v), b, tol=tol, maxiter=maxiter)[0], static_argnums=5)
    ops["solve_C"] = jax.jit(lambda eps, th, var, b, tol, maxiter: cg(
        lambda w: matvec_C(eps, th, var, w), b, tol=tol, maxiter=maxiter)[0], static_argnums=5)
    ops["matvec_schur"] = jax.jit(matvec_schur)
    ops["dC"] = jax.jit(lambda eps, th, w, e_i: ops["dJ_v"](eps, th, ops["vjp"](eps, th, w), e_i)
                        + ops["jvp"](eps, th, ops["dJT_w"](eps, th, w, e_i)))
    _OPS_CACHE[key] = ops
    return ops


class FieldFisher:
    def __init__(self, fm: ForwardModel, layout: ParamLayout, eps_fid, theta_fid, prior=None, rsd=True,
                 linear=False, noise_floor=1e-3, verbose=True, tracers=None):
        """tracers: subset of fm.tracers whose voxels enter the likelihood (default all)."""
        self.fm, self.layout, self.rsd, self.linear, self.verbose = fm, layout, rsd, linear, verbose
        self.tracers = list(tracers) if tracers else fm.tracers
        self.idx = {t: jnp.asarray(np.flatnonzero(fm.footprint[t].ravel())) for t in self.tracers}
        self.n_data = int(sum(len(v) for v in self.idx.values()))
        self.eps_fid = jnp.asarray(eps_fid, dtype=jnp.float64)
        self.theta_fid = jnp.asarray(theta_fid, dtype=jnp.float64)
        self.prior = np.zeros((layout.n, layout.n)) if prior is None else np.asarray(prior)
        var = []
        for t in self.tracers:
            nb = sum(np.asarray(fm.nbar_cell[i]) for i, s in enumerate(fm.shells) if s.tracer == t).ravel()
            var.append(np.maximum(nb[np.asarray(self.idx[t])], noise_floor))
        self.invvar = jnp.asarray(1.0 / np.concatenate(var))
        self.ops = _get_ops(fm, layout, self.tracers, linear, rsd, self.idx)
        self.model = lambda eps, th: self.ops["model"](eps, th)
        self.jvp_eps = lambda v: self.ops["jvp"](self.eps_fid, self.theta_fid, v)
        self.vjp_eps = lambda w: self.ops["vjp"](self.eps_fid, self.theta_fid, w)
        self.matvec = lambda v: self.ops["matvec_schur"](self.eps_fid, self.theta_fid, self.invvar, v)

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def jac_theta(self):
        """dM/dtheta on the footprint: (n_data, n_theta), by n_theta forward-mode JVPs."""
        n = self.layout.n
        cols = [self.ops["jvp_theta"](self.eps_fid, self.theta_fid, jnp.zeros(n).at[i].set(1.0)) for i in range(n)]
        return jnp.stack(cols, axis=1)

    def fisher(self, tol=1e-6, maxiter=400):
        t0 = time.time()
        J = self.jac_theta()
        self.log(f"  jac_theta done ({J.shape}) {time.time()-t0:.0f}s")
        F_tt = np.asarray(J.T @ (self.invvar[:, None] * J)) + self.prior
        n = self.layout.n
        F_te = jnp.stack([self.vjp_eps(self.invvar * J[:, i]) for i in range(n)])    # (n_theta, res^3)
        corr = np.zeros((n, n))
        self.iters = []
        for i in range(n):
            rhs = F_te[i]
            v = self.ops["solve_schur"](self.eps_fid, self.theta_fid, self.invvar, rhs, tol, maxiter)
            rel = float(jnp.linalg.norm(rhs - self.matvec(v)) / jnp.linalg.norm(rhs))
            self.iters.append(rel)
            corr[i, :] = np.asarray(jnp.tensordot(F_te, v, axes=v.ndim))
            self.log(f"  CG {i+1}/{n} ({self.layout.names[i]}): residual {rel:.1e}  {time.time()-t0:.0f}s")
        corr = 0.5 * (corr + corr.T)
        self.F_tt, self.F_marg = F_tt, F_tt - corr
        return self.F_marg

    # ------------------------------------------------------------------------------------------
    # Exact Gaussian-marginal Fisher of the model linearised in eps (Hutchinson trace estimator)
    # ------------------------------------------------------------------------------------------
    def fisher_marginal(self, n_probes=8, tol=1e-6, maxiter=400, seed=0, probe_callback=None, start_probe=0, Fs_prev=None):
        """F_ij = 1/2 Tr[C^-1 dC_i C^-1 dC_j] with C = J_e J_e^T + N (data space), dC_i = dJ_i J^T + J dJ_i^T,
        dJ_i v = d/dtheta_i [J_e(theta) v] (second-order autodiff). Estimated with Rademacher probes z:
            F_ij ~ 1/2 <(C^-1 z)^T dC_i (C^-1 dC_j z)>_z.
        For a linear model this is the exact marginal (Gaussian-field) Fisher including the survey window;
        it differs from the Schur (profile) estimate by (P+N)/(2P) per mode."""
        import resource
        t0 = time.time()
        n = self.layout.n
        eps, th, var = self.eps_fid, self.theta_fid, 1.0 / self.invvar
        E = [jnp.zeros(n).at[i].set(1.0) for i in range(n)]
        dC = lambda w, i: self.ops["dC"](eps, th, w, E[i])
        solve = lambda b: self.ops["solve_C"](eps, th, var, b, tol, maxiter)
        key = jax.random.PRNGKey(seed)
        Fs = list(Fs_prev) if Fs_prev else []
        F = sum(Fs) if Fs else np.zeros((n, n))
        for p_ in range(n_probes):
            key, sub = jax.random.split(key)
            if p_ < start_probe:
                continue
            z = jax.random.rademacher(sub, (self.n_data,), dtype=jnp.float64)
            Cinv_z = solve(z)
            W = jnp.stack([dC(Cinv_z, i) for i in range(n)])         # (n, n_data): dC_i C^-1 z
            Fp = np.zeros((n, n))
            for j in range(n):
                yj = solve(dC(z, j))                                  # C^-1 dC_j z
                Fp[:, j] = 0.5 * np.asarray(W @ yj)
            del W, Cinv_z, z
            Fp = 0.5 * (Fp + Fp.T)
            Fs.append(Fp); F = F + Fp
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            self.log(f"  probe {p_+1}/{n_probes}: sigma(f_NL) running = {sigma_of(F / len(Fs) + self.prior):.2f}  {time.time()-t0:.0f}s  maxRSS {rss:.1f} GB")
            if probe_callback is not None:
                probe_callback(Fs)
        self.F_marginal = F / len(Fs) + self.prior
        self.F_marginal_probes = Fs
        self.F_marginal_probe_std = np.std([sigma_of(Fp + self.prior) if Fp[0, 0] > 0 else np.nan for Fp in Fs])
        return self.F_marginal

    def sigma(self, idx=0, F=None):
        F = self.F_marg if F is None else F
        keep = np.where(np.diag(F) > 0)[0]
        j = list(keep).index(idx)
        return float(np.sqrt(np.linalg.inv(F[np.ix_(keep, keep)])[j, j]))


def sigma_of(F, idx=0):
    keep = np.where(np.diag(F) > 0)[0]
    j = list(keep).index(idx)
    return float(np.sqrt(np.linalg.inv(F[np.ix_(keep, keep)])[j, j]))
