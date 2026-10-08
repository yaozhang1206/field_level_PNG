"""Gaussian Fisher forecast for the DESI DR1 P(k) multipole analysis (Part A of Stage 1).

Covariance (Gaussian, FKP-weighted, Feldman et al. 1994 / Grieb et al. 2016), per k-bin of width Dk:

    Cov[P_l(k), P_l'(k)] = (2l+1)(2l'+1) * 2 (2 pi^2) / (k^2 Dk) * int_0^1 dmu L_l(mu) L_l'(mu) * S(k, mu)
    S(k, mu) = int dV n^4 w^4 [P_g(k, mu) + 1/n]^2 / [int dV n^2 w^2]^2 ,   w = 1 / (1 + n P0)

For constant n this reduces to 2 (2l+1)(2l'+1)/N_modes * int_0^1 dmu L L' [P + 1/n]^2 with
N_modes = V k^2 Dk / (2 pi^2), i.e. Var[P_0] = 2 [P + 1/n]^2 / N_modes.

Fisher: F_ij = sum_k dP^T(k) C^-1(k) dP(k) with dP = d(P_0, P_2)/dtheta from jax autodiff.

Two modes:
  * single-z:   P_g evaluated at z_eff with one (b1, f, D), n(z) integrated in S(k, mu)  [FKP analysis]
  * z-shells:   Fisher summed over thin redshift shells, each with its own b1(z), f(z), D(z) and shot
                noise; f_NL, s_n0, Sigma_s shared, b1 per shell marginalised through the b1(z) template
                amplitude. This is the information an optimal redshift weighting (OQE) can approach.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from cosmology import Cosmology, bias_phi
from model_pk import PkModel, legendre, _GL_X, _GL_W
from geometry import Geometry

PARAMS = ("f_NL", "b1", "s_n0", "Sigma_s")


def k_bins(kmin, kmax, dk=0.001):
    edges = np.arange(kmin, kmax + 1e-9, dk)
    return 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)


class FisherPk:
    def __init__(self, cosmo: Cosmology, geom: Geometry, z_eff: float, b1: float, p: float, P0: float,
                 kmin: float, kmax: float, dk: float = 0.001, ells=(0, 2), Sigma_s_fid: float = 3.0,
                 s_n0_fid: float = 0.0, nz_kind: str = "nx"):
        self.cosmo, self.geom = cosmo, geom
        self.model = PkModel(cosmo, z_eff, p, ells=ells)
        self.theta_fid = np.array([0.0, b1, s_n0_fid, Sigma_s_fid])
        self.p, self.P0, self.ells = p, P0, tuple(ells)
        self.k, self.dk = k_bins(kmin, kmax, dk)
        # radial integration tables. nz_kind='full': weights use n_NX, volume is the geometric
        # footprint and shot noise uses the observed density c_bar n_NX (DESI completeness weighting).
        if nz_kind == "full":
            self.n = jnp.asarray(geom.nz("nx")); self.n_shot = jnp.asarray(geom.nz_obs())
            self.dV = jnp.asarray(geom.dV("full"))
        else:
            self.n = jnp.asarray(geom.nz(nz_kind)); self.n_shot = self.n
            self.dV = jnp.asarray(geom.dV(nz_kind))
        self.w = 1.0 / (1.0 + self.n * P0)
        self.A = jnp.sum(self.n ** 2 * self.w ** 2 * self.dV)       # FKP normalisation
        self.mu = jnp.asarray(0.5 * (_GL_X + 1.0))                  # Gauss-Legendre on [0, 1]
        self.wmu = jnp.asarray(0.5 * _GL_W)

    # ---- covariance ----------------------------------------------------------------------------
    def _S(self, k, theta):
        """S(k, mu) for all k, mu: shape (nk, nmu)."""
        P = self.model.pk_mu(k, self.mu, theta)                     # (nk, nmu)
        n, w, dV = self.n[None, None, :], self.w[None, None, :], self.dV[None, None, :]
        ns = self.n_shot[None, None, :]
        num = jnp.sum(n ** 4 * w ** 4 * (P[..., None] + 1.0 / ns) ** 2 * dV, axis=-1)
        return num / self.A ** 2

    def covariance(self, theta=None):
        """Block-diagonal covariance: array (nk, nell, nell)."""
        theta = self.theta_fid if theta is None else theta
        k = jnp.asarray(self.k)
        S = self._S(k, theta)
        L = jnp.stack([legendre(l, self.mu) for l in self.ells])    # (nell, nmu)
        pref = 2.0 * (2 * jnp.pi ** 2) / (k ** 2 * jnp.asarray(self.dk))
        I = jnp.einsum("am,bm,km,m->kab", L, L, S, self.wmu)
        fac = jnp.array([(2 * l + 1) for l in self.ells], dtype=jnp.float64)
        return pref[:, None, None] * fac[None, :, None] * fac[None, None, :] * I

    # ---- fisher ----------------------------------------------------------------------------------
    def jacobian(self, theta=None, bphi_fnl=False):
        """dP_l(k)/dtheta: array (nk, nell, ntheta). If bphi_fnl, parameter 0 is b_phi f_NL."""
        theta = jnp.asarray(self.theta_fid if theta is None else theta, dtype=jnp.float64)
        k = jnp.asarray(self.k)
        if not bphi_fnl:
            fn = lambda th: self.model.multipoles(k, th)
            J = jax.jacfwd(fn)(theta)                               # (nell, nk, ntheta)
        else:
            fn = lambda th: self.model.multipoles(k, jnp.concatenate([jnp.zeros(1), th[1:]]), bphi_fnl=th[0])
            J = jax.jacfwd(fn)(theta)
        return jnp.transpose(J, (1, 0, 2))

    def fisher(self, theta=None, bphi_fnl=False, kmask=None):
        C = self.covariance(theta)
        J = self.jacobian(theta, bphi_fnl)
        if kmask is not None:
            C, J = C[kmask], J[kmask]
        Cinv = jnp.linalg.inv(C)
        F = jnp.einsum("kai,kab,kbj->ij", J, Cinv, J)
        return np.asarray(F)

    def info_density(self, theta=None):
        """Per-k Fisher element for f_NL after marginalising the rest: dF/dln k approximations.
        Returns per-bin unmarginalised F_fnl,fnl(k) and the cumulative marginalised sigma(k_max)."""
        C = self.covariance(theta); J = self.jacobian(theta)
        Cinv = jnp.linalg.inv(C)
        Fk = np.asarray(jnp.einsum("kai,kab,kbj->kij", J, Cinv, J))  # (nk, nt, nt)
        return Fk


def sigma_from_fisher(F, idx=0, fixed=()):
    """Marginalised sigma of parameter idx; `fixed` lists parameter indices held fixed."""
    keep = [i for i in range(F.shape[0]) if i not in fixed]
    Fk = F[np.ix_(keep, keep)]
    j = keep.index(idx)
    return float(np.sqrt(np.linalg.inv(Fk)[j, j]))


def add_prior(F, idx, sigma):
    F = F.copy(); F[idx, idx] += 1.0 / sigma ** 2; return F


# ---------------------------------------------------------------------------------------------------
# Redshift-shell version (optimal-weighting limit)
# ---------------------------------------------------------------------------------------------------
class FisherPkShells:
    """Sum of single-shell Fishers with per-shell b1(z) = a (1+z)^2 + b; the free bias parameter is a
    global amplitude A_b multiplying the template (b1 -> A_b * b1(z)), so the parameter vector is
    (f_NL, A_b, s_n0, Sigma_s) — comparable to the single-z analysis."""

    def __init__(self, cosmo, geom, b1_of_z, p, P0, kmin, kmax, dk=0.001, ells=(0, 2),
                 Sigma_s_fid=3.0, nshell=8, nz_kind="nx"):
        self.shells = []
        edges = np.linspace(geom.zmin, geom.zmax, nshell + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (geom.z >= lo) & (geom.z < hi + 1e-12)
            if sel.sum() == 0:
                continue
            sub = _SubGeom(geom, sel, nz_kind)
            nk = "nx" if nz_kind == "full" else nz_kind
            zc = float(np.sum(geom.z[sel] * geom.dV(nz_kind)[sel] * geom.nz(nk)[sel] ** 2)
                       / np.sum(geom.dV(nz_kind)[sel] * geom.nz(nk)[sel] ** 2))
            f = FisherPk(cosmo, sub, zc, float(b1_of_z(zc)), p, P0, kmin, kmax, dk, ells, Sigma_s_fid, nz_kind=nz_kind)
            self.shells.append((zc, f))

    def fisher(self):
        F = 0.0
        for zc, f in self.shells:
            Fs = f.fisher()
            # chain rule: d/dA_b = b1(z) d/db1 at A_b = 1
            b1 = f.theta_fid[1]
            T = np.diag([1.0, b1, 1.0, 1.0])
            F = F + T @ Fs @ T
        return F


class _SubGeom:
    """View of a Geometry restricted to a redshift selection (for shell Fishers)."""
    def __init__(self, g, sel, kind):
        self._nz = {"nx": g.nz("nx")[sel], "count": g.nz("count")[sel]}
        self._dV = {"nx": g.dV("nx")[sel], "count": g.dV("count")[sel], "full": g.dV("full")[sel]}
        self._nobs = g.nz_obs()[sel]
        self.z, self.zmin, self.zmax = g.z[sel], g.z[sel].min(), g.z[sel].max()
    def nz(self, kind="nx"): return self._nz[kind]
    def nz_obs(self): return self._nobs
    def dV(self, kind="nx"): return self._dV[kind]
