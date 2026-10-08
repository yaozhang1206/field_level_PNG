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
        # noise: sigma^2 = unperturbed expected counts per voxel (sum of shells of that tracer)
        var = []
        for t in self.tracers:
            nb = sum(np.asarray(fm.nbar_cell[i]) for i, s in enumerate(fm.shells) if s.tracer == t).ravel()
            var.append(np.maximum(nb[np.asarray(self.idx[t])], noise_floor))
        self.invvar = jnp.asarray(1.0 / np.concatenate(var))

        def model(eps, theta):
            out = fm.linear_galaxy_field(eps, theta, layout, rsd) if linear else fm(eps, theta, layout, rsd)
            return jnp.concatenate([out[t].ravel()[self.idx[t]] for t in self.tracers])

        self.model = jax.jit(model)
        self._f_eps = lambda e: self.model(e, self.theta_fid)
        self._f_th = lambda th: self.model(self.eps_fid, th)
        self.jvp_eps = jax.jit(lambda v: jax.jvp(self._f_eps, (self.eps_fid,), (v,))[1])
        _, vjp_fn = jax.vjp(self._f_eps, self.eps_fid)
        self.vjp_eps = jax.jit(lambda w: vjp_fn(w)[0])
        self.matvec = jax.jit(lambda v: self.vjp_eps(self.invvar * self.jvp_eps(v)) + v)

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def jac_theta(self):
        """dM/dtheta on the footprint: (n_data, n_theta), by n_theta forward-mode JVPs."""
        cols = []
        n = self.layout.n
        for i in range(n):
            e = jnp.zeros(n).at[i].set(1.0)
            cols.append(jax.jvp(self._f_th, (self.theta_fid,), (e,))[1])
        return jnp.stack(cols, axis=1)

    def fisher(self, tol=1e-6, maxiter=400):
        t0 = time.time()
        J = self.jac_theta()
        self.log(f"  jac_theta done ({J.shape}) {time.time()-t0:.0f}s")
        F_tt = np.asarray(J.T @ (self.invvar[:, None] * J)) + self.prior
        n = self.layout.n
        F_te = []
        for i in range(n):
            F_te.append(self.vjp_eps(self.invvar * J[:, i]))
        F_te = jnp.stack(F_te)                                   # (n_theta, res^3)
        corr = np.zeros((n, n))
        self.iters = []
        for i in range(n):
            rhs = F_te[i]
            v, info = cg(self.matvec, rhs, tol=tol, maxiter=maxiter)
            r = rhs - self.matvec(v)
            rel = float(jnp.linalg.norm(r) / jnp.linalg.norm(rhs))
            self.iters.append(rel)
            for j in range(n):
                corr[i, j] = float(jnp.vdot(F_te[j], v))
            self.log(f"  CG {i+1}/{n} ({self.layout.names[i]}): residual {rel:.1e}  {time.time()-t0:.0f}s")
        corr = 0.5 * (corr + corr.T)
        self.F_tt, self.F_marg = F_tt, F_tt - corr
        return self.F_marg

    def sigma(self, idx=0, F=None):
        F = self.F_marg if F is None else F
        keep = np.where(np.diag(F) > 0)[0]
        j = list(keep).index(idx)
        return float(np.sqrt(np.linalg.inv(F[np.ix_(keep, keep)])[j, j]))


def sigma_of(F, idx=0):
    keep = np.where(np.diag(F) > 0)[0]
    j = list(keep).index(idx)
    return float(np.sqrt(np.linalg.inv(F[np.ix_(keep, keep)])[j, j]))
