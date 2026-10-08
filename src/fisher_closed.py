"""Closed-form Fisher forecasts for the field-level analysis (Stage 1 Part B cross-checks).

Tier 1 (Gaussian two-point, mode-by-mode):
    F_ij = sum_shells  1/2 int dV(z) int d^3k/(2pi)^3  dP_g/dtheta_i dP_g/dtheta_j / [P_g(k,mu) + 1/n_obs(z)]^2
with P_g(k, mu) = [b1 + bphi f_NL / T(k,z) + f mu^2]^2 P_lin(k, z)  (no damping; tree level).
The k integral runs over the modes of the simulation grid (cube |k_i| <= k_Ny) or a sphere k < k_max.

Tier 2 (tree-level galaxy bispectrum, real space, Gaussian covariance):
    B_g = c1 c2 c3 B_PNG + sum_cyc c1 c2 [2 b1 F2(k1,k2) + b2 + 2 bK S2(k1,k2) + bphidelta f_NL (1/T1 + 1/T2)] P1 P2
    c_i = b1 + bphi f_NL / T(k_i),   B_PNG = 2 f_NL T1 T2 T3 (PPhi1 PPhi2 + PPhi2 PPhi3 + PPhi3 PPhi1)
    Var[B] = s_B (2pi)^6 P~1 P~2 P~3 / (V 8 pi^2 k1 k2 k3 Dk^3),  P~ = b1^2 P_lin + 1/n_obs
    F_ij = sum_shells sum_triangles dB_i dB_j / Var[B]      (summed over the shell's n(z) bins)
RSD is not included in Tier 2 (real-space bispectrum); Tier 1 includes Kaiser RSD.

Parameter layout follows forward_model.ParamLayout: [f_NL] + per shell [b1, b2, bK] (+ [bphi, bphidelta]).
Fiducial f_NL = 0.
"""
from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cosmology import Cosmology  # noqa: E402
from forward_model import ParamLayout, Shell  # noqa: E402


class ShellGeometry:
    """Per-shell radial tables: dV(z) (full footprint) and observed density n_obs(z)."""

    def __init__(self, geoms: dict):
        self.geoms = geoms  # {tracer: Geometry}

    def tables(self, s: Shell):
        g = self.geoms[s.tracer]
        sel = (g.z >= s.zlo) & (g.z < s.zhi)
        return g.z[sel], g.dV("full")[sel], g.nz_obs()[sel]


def _shell_params(theta, layout: ParamLayout, i):
    f_NL, b1, b2, bK, bphi, bphid = layout.unpack(theta, i)
    return jnp.stack([f_NL, b1, b2, bK, bphi, bphid])


def _embed(F6, layout: ParamLayout, i):
    """Embed a 6x6 shell Fisher (f_NL, b1, b2, bK, bphi, bphid) into the global layout."""
    idx = [0] + [1 + i * layout.per_shell + j for j in range(layout.per_shell)]
    F = np.zeros((layout.n, layout.n))
    sub = F6 if layout.sample_bphi else F6[:4, :4]
    F[np.ix_(idx, idx)] += sub
    return F


# ----------------------------------------------------------------------------------------------
# Tier 1
# ----------------------------------------------------------------------------------------------
def grid_modes(L, res, sphere_kmax=None):
    """Independent modes of an res^3 rfft grid (half-space), returns (k, mu, weight)."""
    kf = 2 * np.pi / L
    n = np.fft.fftfreq(res, 1.0 / res)
    kx, ky = np.meshgrid(n * kf, n * kf, indexing="ij")
    kz = np.arange(res // 2 + 1) * kf
    KX, KY, KZ = np.meshgrid(n * kf, n * kf, kz, indexing="ij")
    k = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    w = np.ones_like(k); w[..., 0] = 0.5; w[..., -1] = 0.5          # kz=0 and Nyquist planes counted half
    sel = k > 0
    if sphere_kmax is not None:
        sel &= k < sphere_kmax
    mu = np.where(k > 0, KZ / np.where(k > 0, k, 1), 0.0)
    return k[sel], mu[sel], w[sel]


class Tier1:
    def __init__(self, cosmo: Cosmology, shells, layout: ParamLayout, sgeom: ShellGeometry,
                 L=10000.0, res=80, sphere_kmax=None, kmin=None, nk=200, continuum=True):
        """continuum=True: integrate d^3k analytically over k in [kmin, kmax] (kmax = k_Ny or sphere_kmax)
        with the mode density V/(2pi)^3; continuum=False: sum the actual grid modes."""
        self.cosmo, self.shells, self.layout, self.sg = cosmo, shells, layout, sgeom
        self.L, self.res = L, res
        kny = np.pi * res / L
        self.kmax = sphere_kmax or kny
        self.kmin = kmin or 2 * np.pi / L
        self.continuum = continuum
        if continuum:
            self.k = np.logspace(np.log10(self.kmin), np.log10(self.kmax), nk)
            x, wmu = np.polynomial.legendre.leggauss(16)
            self.mu, self.wmu = 0.5 * (x + 1), 0.5 * wmu
        else:
            self.kg, self.mug, self.wg = grid_modes(L, res, sphere_kmax)

    def P_g(self, k, mu, par, z):
        f_NL, b1, b2, bK, bphi, bphid = par
        T = self.cosmo.T_Phi_delta(k, z)
        f = self.cosmo.f(z)
        return (b1 + bphi * f_NL / T + f * mu ** 2) ** 2 * self.cosmo.P_lin(k, z)

    def fisher(self, theta=None, laplace=False):
        """Per independent (complex, half-space) mode: laplace=False: exact Gaussian-field Fisher dP dP/(P+N)^2
        (= 1/2 dlnP dlnP per real dof x 2 dof); laplace=True: expectation of the joint-(eps,theta) Schur
        complement, dP dP / (2 P (P+N)). Ratio Laplace/exact = (P+N)/(2P)."""
        theta = self.layout.fiducial() if theta is None else theta
        F = np.zeros((self.layout.n, self.layout.n))
        weight = (lambda P, N: 1.0 / (2.0 * P * (P + N))) if laplace else (lambda P, N: 1.0 / (P + N) ** 2)
        for i, s in enumerate(self.shells):
            z, dV, nobs = self.sg.tables(s)
            zc = 0.5 * (s.zlo + s.zhi)
            par = _shell_params(theta, self.layout, i)
            if self.continuum:
                k = jnp.asarray(self.k); mu = jnp.asarray(self.mu)
                K, MU = jnp.meshgrid(k, mu, indexing="ij")
                dP = jax.jacfwd(lambda p: self.P_g(K.ravel(), MU.ravel(), p, zc))(par)   # (nk*nmu, 6)
                P = self.P_g(K.ravel(), MU.ravel(), par, zc)
                # inverse-variance weight integrated over the shell's n(z)
                inv = jnp.sum(jnp.asarray(dV)[None, :] * weight(P[:, None], 1.0 / jnp.asarray(nobs)[None, :]), axis=1)
                # d^3k/(2pi)^3 with 1/2 for independent modes: 2pi k^2 dk dmu(-1..1) / (2pi)^3 * 1/2 = k^2 dk dmu(0..1) / (4 pi^2)
                wk = (K ** 2 * jnp.asarray(np.gradient(self.k))[:, None] * jnp.asarray(self.wmu)[None, :]).ravel() / (4 * np.pi ** 2)
                F6 = jnp.einsum("m,m,mi,mj->ij", wk, inv, dP, dP)
            else:
                k = jnp.asarray(self.kg); mu = jnp.asarray(self.mug); w = jnp.asarray(self.wg)
                dP = jax.jacfwd(lambda p: self.P_g(k, mu, p, zc))(par)
                P = self.P_g(k, mu, par, zc)
                Vbox = self.L ** 3
                inv = jnp.sum(jnp.asarray(dV)[None, :] * weight(P[:, None], 1.0 / jnp.asarray(nobs)[None, :]), axis=1) / Vbox
                F6 = jnp.einsum("m,m,mi,mj->ij", w, inv, dP, dP)    # each grid mode: 1 mode per V/(2pi)^3 * (2pi/L)^3 = 1/Vbox
            F += _embed(np.asarray(F6), self.layout, i)
        return F


# ----------------------------------------------------------------------------------------------
# Tier 2
# ----------------------------------------------------------------------------------------------
def triangles(kmin, kmax, dk):
    """All triangle bins k1 >= k2 >= k3 >= kmin with k1 <= kmax and k3 >= k1 - k2 (bin centres)."""
    ks = np.arange(kmin + dk / 2, kmax, dk)
    out = []
    for i, k1 in enumerate(ks):
        for j, k2 in enumerate(ks[:i + 1]):
            for l, k3 in enumerate(ks[:j + 1]):
                if k3 >= k1 - k2 - 1e-12:
                    sB = 6 if (i == j == l) else (2 if (i == j or j == l) else 1)
                    out.append((k1, k2, k3, sB))
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


class Tier2:
    def __init__(self, cosmo: Cosmology, shells, layout: ParamLayout, sgeom: ShellGeometry,
                 kmax=0.025, kmin=None, dk=0.002, L=10000.0):
        self.cosmo, self.shells, self.layout, self.sg = cosmo, shells, layout, sgeom
        self.kmin = kmin or 2 * np.pi / L
        self.kmax, self.dk = kmax, dk
        self.k1, self.k2, self.k3, self.sB = triangles(self.kmin, kmax, dk)

    @staticmethod
    def F2(k1, k2, k3):
        c = (k3 ** 2 - k1 ** 2 - k2 ** 2) / (2 * k1 * k2)      # cos(angle between k1 and k2)
        return 5.0 / 7.0 + 0.5 * c * (k1 / k2 + k2 / k1) + 2.0 / 7.0 * c ** 2

    @staticmethod
    def S2(k1, k2, k3):
        c = (k3 ** 2 - k1 ** 2 - k2 ** 2) / (2 * k1 * k2)
        return c ** 2 - 1.0 / 3.0

    def B_g(self, par, z):
        f_NL, b1, b2, bK, bphi, bphid = par
        k1, k2, k3 = jnp.asarray(self.k1), jnp.asarray(self.k2), jnp.asarray(self.k3)
        P = [self.cosmo.P_lin(k, z) for k in (k1, k2, k3)]
        T = [self.cosmo.T_Phi_delta(k, z) for k in (k1, k2, k3)]
        PPhi = [self.cosmo.P_Phi(k) for k in (k1, k2, k3)]
        c = [b1 + bphi * f_NL / Ti for Ti in T]
        B_png = 2 * f_NL * T[0] * T[1] * T[2] * (PPhi[0] * PPhi[1] + PPhi[1] * PPhi[2] + PPhi[2] * PPhi[0])
        B = c[0] * c[1] * c[2] * B_png
        for (a, b, cc) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            ka, kb, kc = (k1, k2, k3)[a], (k1, k2, k3)[b], (k1, k2, k3)[cc]
            kern = 2 * b1 * self.F2(ka, kb, kc) + b2 + 2 * bK * self.S2(ka, kb, kc) + bphid * f_NL * (1 / T[a] + 1 / T[b])
            B = B + c[a] * c[b] * kern * P[a] * P[b]
        return B

    def fisher(self, theta=None):
        theta = self.layout.fiducial() if theta is None else theta
        F = np.zeros((self.layout.n, self.layout.n))
        k1, k2, k3, sB = (jnp.asarray(x) for x in (self.k1, self.k2, self.k3, self.sB))
        for i, s in enumerate(self.shells):
            z, dV, nobs = self.sg.tables(s)
            zc = 0.5 * (s.zlo + s.zhi)
            par = _shell_params(theta, self.layout, i)
            dB = jax.jacfwd(lambda p: self.B_g(p, zc))(par)          # (ntri, 6)
            b1 = par[1]
            Pt = [b1 ** 2 * self.cosmo.P_lin(k, zc) for k in (k1, k2, k3)]
            inv = 0.0
            for dVz, n in zip(dV, nobs):
                inv = inv + dVz / ((Pt[0] + 1 / n) * (Pt[1] + 1 / n) * (Pt[2] + 1 / n))
            wt = k1 * k2 * k3 * self.dk ** 3 / (8 * np.pi ** 4 * sB) * inv
            F6 = jnp.einsum("t,ti,tj->ij", wt, dB, dB)
            F += _embed(np.asarray(F6), self.layout, i)
        return F


def sigma(F, idx=0, prior=None):
    """Marginalised sigma of parameter idx. Parameters with no information and no prior (zero row) are
    dropped (they do not enter the model at this order and so do not degrade the constraint)."""
    Ft = F + (prior if prior is not None else 0)
    keep = np.where(np.diag(Ft) > 0)[0]
    if idx not in keep:
        return float("inf")
    j = list(keep).index(idx)
    return float(np.sqrt(np.linalg.inv(Ft[np.ix_(keep, keep)])[j, j]))
