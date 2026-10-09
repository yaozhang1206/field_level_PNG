"""Stage 2: mock data and log-posterior for field-level inference of f_NL.

    ln p(eps, theta | d) = -1/2 sum_x [d(x) - M(eps,theta)(x)]^2 / sigma^2(x)  - 1/2 |eps|^2  + ln p(theta)
    (Gaussian noise, sigma^2 = nbar V_cell, zero weight off footprint)   or   Poisson:  sum_x [d ln mu - mu]

The model is evaluated with the reverse-mode view of the forward model (DISCO-DJ scatter custom_vjp), so
jax.grad / jax.value_and_grad work on (eps, theta). Parameters are kept in their natural units; a whitening
transform for theta (from the Stage 1 Fisher) is provided for samplers.

Mean guard: the Lagrangian bias expansion can give 1 + delta_g < 0 in a few voxels at 2LPT. For the Poisson
likelihood the mean is softly floored, mu -> mu_floor + softplus(mu - mu_floor), which is smooth and leaves
mu >> mu_floor unchanged; for the Gaussian likelihood no floor is needed (sigma^2 is the unperturbed mean).
"""
from __future__ import annotations

import copy
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from forward_model import ForwardModel, ParamLayout  # noqa: E402


class FieldPosterior:
    def __init__(self, fm: ForwardModel, layout: ParamLayout, tracers=None, rsd=True, linear=False,
                 likelihood="gaussian", noise_floor=1e-3, mu_floor=1e-2, theta_prior_sigma=None):
        """theta_prior_sigma: optional array of Gaussian prior widths on theta (None -> flat)."""
        self.fm, self.layout, self.rsd, self.linear = fm, layout, rsd, linear
        self.tracers = list(tracers) if tracers else fm.tracers
        self.likelihood = likelihood
        self.idx = {t: jnp.asarray(np.flatnonzero(fm.footprint[t].ravel())) for t in self.tracers}
        self.n_data = int(sum(len(v) for v in self.idx.values()))
        var = []
        for t in self.tracers:
            nb = sum(np.asarray(fm.nbar_cell[i]) for i, s in enumerate(fm.shells) if s.tracer == t).ravel()
            var.append(np.maximum(nb[np.asarray(self.idx[t])], noise_floor))
        self.var = jnp.asarray(np.concatenate(var))
        self.invvar = 1.0 / self.var
        self.mu_floor = mu_floor
        self.theta_prior_sigma = None if theta_prior_sigma is None else jnp.asarray(theta_prior_sigma)
        fm_bwd = copy.copy(fm); fm_bwd.requires_jacfwd = False
        fm_fwd = copy.copy(fm); fm_fwd.requires_jacfwd = True

        def make_model(f):
            def model(eps, theta):
                out = f.linear_galaxy_field(eps, theta, layout, rsd) if linear else f(eps, theta, layout, rsd)
                return jnp.concatenate([out[t].ravel()[self.idx[t]] for t in self.tracers])
            return model
        self.model = jax.jit(make_model(fm_bwd))          # reverse-mode view (grad)
        self.model_fwd = jax.jit(make_model(fm_fwd))      # forward-mode view (jvp)

    # ---- data -------------------------------------------------------------------------------
    def mock(self, eps_true, theta_true, seed=0):
        """Draw mock data from the model at (eps_true, theta_true)."""
        mu = self.model(eps_true, theta_true)
        key = jax.random.PRNGKey(seed)
        if self.likelihood == "poisson":
            return jax.random.poisson(key, jnp.maximum(mu, 0.0)).astype(jnp.float64)
        return mu + jnp.sqrt(self.var) * jax.random.normal(key, mu.shape, dtype=jnp.float64)

    # ---- log posterior ----------------------------------------------------------------------
    def log_like(self, eps, theta, d):
        mu = self.model(eps, theta)
        if self.likelihood == "poisson":
            mu = self.mu_floor + jax.nn.softplus(mu - self.mu_floor)
            return jnp.sum(d * jnp.log(mu) - mu)
        return -0.5 * jnp.sum((d - mu) ** 2 * self.invvar)

    def log_prior(self, eps, theta):
        lp = -0.5 * jnp.sum(eps ** 2)
        if self.theta_prior_sigma is not None:
            lp = lp - 0.5 * jnp.sum(((theta - self.layout.fiducial()) / self.theta_prior_sigma) ** 2)
        return lp

    def log_post(self, eps, theta, d):
        return self.log_like(eps, theta, d) + self.log_prior(eps, theta)

    def make_logdensity(self, d, theta_scale=None, free=None, theta_fid=None):
        """Return (logdensity(state), unpack) with a flat state vector [eps.ravel(), theta_w[free]] where
        theta[free] = theta_fid[free] + theta_scale[free] * theta_w (whitened), other parameters fixed."""
        n_eps = self.fm.res ** 3; shape = (self.fm.res,) * 3
        th0 = self.layout.fiducial() if theta_fid is None else jnp.asarray(theta_fid)
        free = jnp.arange(self.layout.n) if free is None else jnp.asarray(free)
        scale = jnp.ones_like(th0) if theta_scale is None else jnp.asarray(theta_scale)
        def unpack(state):
            theta = th0.at[free].add(scale[free] * state[n_eps:])
            return state[:n_eps].reshape(shape), theta
        def logdensity(state):
            eps, theta = unpack(state)
            return self.log_post(eps, theta, d)
        return jax.jit(logdensity), unpack

    def pack(self, eps, theta, theta_scale=None, free=None, theta_fid=None):
        th0 = self.layout.fiducial() if theta_fid is None else jnp.asarray(theta_fid)
        free = jnp.arange(self.layout.n) if free is None else jnp.asarray(free)
        scale = jnp.ones_like(th0) if theta_scale is None else jnp.asarray(theta_scale)
        return jnp.concatenate([jnp.ravel(eps), ((theta - th0) / scale)[free]])
