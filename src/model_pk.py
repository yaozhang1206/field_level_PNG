"""Galaxy power-spectrum model of Chaussidon et al. 2024 (arXiv:2411.17623), Eq. 2.4:

    P_g(k, mu) = [ b1 + b_phi f_NL / T_Phi->delta(k, z) + f mu^2 ]^2 / [1 + (k mu Sigma_s)^2 / 2]^2 * P_lin(k, z) + s_n0

with b_phi = 2 delta_c (b1 - p). Free parameters theta = (f_NL, b1, s_n0, Sigma_s).
Legendre multipoles P_ell(k) = (2 ell + 1)/2 * int_{-1}^{1} P(k, mu) L_ell(mu) dmu by Gauss-Legendre.

Everything is jax so dP_ell/dtheta comes from autodiff.
Units: k in h/Mpc, P in (Mpc/h)^3, Sigma_s in Mpc/h.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from cosmology import Cosmology, DELTA_C, bias_phi

PARAM_NAMES = ("f_NL", "b1", "s_n0", "Sigma_s")

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def legendre(ell, mu):
    if ell == 0:
        return jnp.ones_like(mu)
    if ell == 2:
        return 0.5 * (3 * mu ** 2 - 1)
    if ell == 4:
        return (35 * mu ** 4 - 30 * mu ** 2 + 3) / 8.0
    raise ValueError(ell)


class PkModel:
    """P_ell(k; theta) for one tracer at one effective redshift."""

    def __init__(self, cosmo: Cosmology, z_eff: float, p: float, ells=(0, 2), delta_c: float = DELTA_C):
        self.cosmo = cosmo
        self.z = float(z_eff)
        self.p = float(p)
        self.ells = tuple(ells)
        self.delta_c = delta_c
        self.f = float(cosmo.f(self.z))

    def pk_mu(self, k, mu, theta, bphi_fnl=None):
        """P_g(k, mu). theta = (f_NL, b1, s_n0, Sigma_s).
        If bphi_fnl is given it replaces b_phi * f_NL (assumption-free parametrisation)."""
        f_NL, b1, s_n0, Sigma_s = theta
        k = jnp.asarray(k)[:, None]
        mu = jnp.asarray(mu)[None, :]
        Tk = self.cosmo.T_Phi_delta(k, self.z)
        if bphi_fnl is None:
            bphi_fnl = bias_phi(b1, self.p, self.delta_c) * f_NL
        kaiser = b1 + bphi_fnl / Tk + self.f * mu ** 2
        damp = 1.0 / (1.0 + (k * mu * Sigma_s) ** 2 / 2.0) ** 2
        return kaiser ** 2 * damp * self.cosmo.P_lin(k, self.z) + s_n0

    def multipoles(self, k, theta, bphi_fnl=None):
        """Return array of shape (n_ell, n_k)."""
        mu = jnp.asarray(_GL_X)
        w = jnp.asarray(_GL_W)
        P = self.pk_mu(k, mu, theta, bphi_fnl)
        out = []
        for ell in self.ells:
            out.append((2 * ell + 1) / 2.0 * jnp.sum(P * legendre(ell, mu)[None, :] * w[None, :], axis=1))
        return jnp.stack(out)

    def data_vector(self, k_ells, theta, bphi_fnl=None):
        """Concatenated [P_ell(k_ell) for ell in ells] with per-ell k arrays (different Delta k)."""
        return jnp.concatenate([self.multipoles(k, theta, bphi_fnl)[i] for i, k in enumerate(k_ells)])

    def jacobian(self, k_ells, theta, bphi_fnl=None):
        """d data_vector / d theta, shape (n_data, n_theta). If bphi_fnl is not None the first
        parameter is b_phi f_NL instead of f_NL."""
        if bphi_fnl is None:
            fn = lambda th: self.data_vector(k_ells, th)
            return jax.jacfwd(fn)(jnp.asarray(theta, dtype=jnp.float64))
        theta = jnp.asarray(theta, dtype=jnp.float64)
        fn = lambda th: self.data_vector(k_ells, jnp.concatenate([jnp.zeros(1), th[1:]]), bphi_fnl=th[0])
        return jax.jacfwd(fn)(theta.at[0].set(bphi_fnl))
