"""Field-level forward model for DESI DR1 QSO/LRG with local PNG (Stage 1 Part B, shared with Stage 2).

    eps (white noise, res^3)  ->  delta_L(k) = sqrt(P_lin(k,0) (res/L)^3) eps(k)
    phi_g(k) = delta_L(k) / T_Phi->delta(k, z=0)              (primordial potential, Gaussian)
    Phi_NG   = phi_g + f_NL (phi_g^2 - <phi_g^2>)              (local PNG, real space)
    delta_NG(k) = T_Phi->delta(k, 0) Phi_NG(k)                 (linear density at z=0 incl. PNG)
    DISCO-DJ nLPT (default 2LPT) from delta_NG  ->  psi_n(q)
    per redshift shell s with scale factor a_s:
        x(q)   = q + sum_n D(a_s)^n psi_n(q)                   (periodic box, observer at the centre)
        x_s    = x + f(a_s) [ (sum_n n D^n psi_n) . rhat ] rhat   (RSD along the local line of sight)
        w(q)   = 1 + b1L delta_L + b2L/2 (delta_L^2 - s2) + bKL (K_L^2 - <K^2>)
                   + bphi f_NL phi_g + bphidL f_NL delta_L phi_g                 (Lagrangian weights, at a_s)
        rho_s(x) = CIC-scatter of w(q) at x_s(q)                 (= 1 + delta_g in redshift space)
        <N_s>(x) = nbar_s(chi(x)) V_cell W_ang(x) rho_s(x)        (selection: radial range + angular mask)
    Data field for tracer t: M_t = sum_{s in t} <N_s>.  Noise: Gaussian, sigma^2 = M_fid (1/sigma^2 = 0 off footprint).

Lagrangian <-> Eulerian bias (local-Lagrangian + PNG, Desjacques/Barreira conventions):
    b1 = 1 + b1L ;  b2 = b2L + 8/21 b1L ;  bK = bKL - 2/7 b1L ;  bphi same ;  bphidelta = bphidL + bphi
so the Eulerian Euclid parameters (b1, b2, bK, bphi, bphidelta) are converted on input. In the linear limit the
galaxy field is  delta_g = [b1 + bphi f_NL / T(k) + f mu^2] delta  (Kaiser + scale-dependent bias).

All field operations are jax and differentiable w.r.t. eps and the parameter vector theta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from cosmology import Cosmology, bias_phi, DELTA_C  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Parameter bookkeeping
# ----------------------------------------------------------------------------------------------
@dataclass
class Shell:
    tracer: str
    zlo: float
    zhi: float
    b1: float
    b2: float
    bK: float
    bphi: float
    bphidelta: float
    p: float
    # filled by the model
    a: float = 0.0
    chi_lo: float = 0.0
    chi_hi: float = 0.0


def lazeyras_b2(b1):
    """b2(b1) fit of Lazeyras et al. 2016 (Eulerian)."""
    return 0.412 - 2.143 * b1 + 0.929 * b1 ** 2 + 0.008 * b1 ** 3


def euclid_bphidelta(b1, b2, p, delta_c=DELTA_C):
    bphi = bias_phi(b1, p, delta_c)
    return bphi - b1 + 1 + delta_c * (b2 - (8.0 / 21.0) * (b1 - 1))


def default_shell(tracer, zlo, zhi, b1, p):
    b2 = lazeyras_b2(b1)
    bK = -(2.0 / 7.0) * (b1 - 1)
    return Shell(tracer, zlo, zhi, b1, b2, bK, bias_phi(b1, p), euclid_bphidelta(b1, b2, p), p)


# parameter layout: theta = [f_NL] + per shell [b1, b2, bK] (+ [bphi, bphidelta] if sample_bphi)
class ParamLayout:
    def __init__(self, shells, sample_bphi=False):
        self.shells = shells
        self.sample_bphi = sample_bphi
        self.per_shell = 5 if sample_bphi else 3
        self.names = ["f_NL"]
        for i, s in enumerate(shells):
            tag = f"{s.tracer}{i}"
            self.names += [f"b1_{tag}", f"b2_{tag}", f"bK_{tag}"]
            if sample_bphi:
                self.names += [f"bphi_{tag}", f"bphid_{tag}"]
        self.n = len(self.names)

    def fiducial(self, f_NL=0.0):
        th = [f_NL]
        for s in self.shells:
            th += [s.b1, s.b2, s.bK] + ([s.bphi, s.bphidelta] if self.sample_bphi else [])
        return jnp.asarray(th, dtype=jnp.float64)

    def unpack(self, theta, i):
        """Return (f_NL, b1, b2, bK, bphi, bphidelta) for shell i."""
        o = 1 + i * self.per_shell
        s = self.shells[i]
        if self.sample_bphi:
            return theta[0], theta[o], theta[o + 1], theta[o + 2], theta[o + 3], theta[o + 4]
        return theta[0], theta[o], theta[o + 1], theta[o + 2], s.bphi, s.bphidelta

    def prior(self, sigma_frac_bphi=0.4):
        """Diagonal prior Fisher: 40 % Gaussian on bphi, bphidelta when sampled; flat otherwise."""
        P = np.zeros((self.n, self.n))
        if self.sample_bphi:
            for i, s in enumerate(self.shells):
                o = 1 + i * self.per_shell
                P[o + 3, o + 3] = 1.0 / (sigma_frac_bphi * s.bphi) ** 2
                P[o + 4, o + 4] = 1.0 / (sigma_frac_bphi * max(abs(s.bphidelta), 1e-3)) ** 2
        return P


# ----------------------------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------------------------
class ForwardModel:
    def __init__(self, cosmo: Cosmology, shells: list[Shell], L: float = 10000.0, res: int = 80,
                 nz_funcs: dict | None = None, ang_mask=None, mask_nside: int = 64, n_order: int = 2,
                 worder: int = 2, precision: str = "double", observer=None):
        """
        nz_funcs: {tracer: callable z -> observed density (h/Mpc)^3}
        ang_mask: HEALPix map (nside=mask_nside, RING) of fractional coverage in [0,1]
        """
        from discodj import DiscoDJ
        self.cosmo, self.shells, self.L, self.res = cosmo, shells, float(L), int(res)
        self.n_order, self.worder = n_order, worder
        self.dtype = jnp.float64 if precision == "double" else jnp.float32
        cp = cosmo.p
        self.dj = DiscoDJ(dim=3, res=res, boxsize=L, precision=precision,
                          cosmo=dict(Omega_c=cp.Omega_c, Omega_b=cp.Omega_b, h=cp.h, sigma8=cosmo.sigma8(0.0),
                                     n_s=cp.n_s, Omega_k=0.0, w0=-1.0, wa=0.0)).with_timetables()
        # tables
        kg = np.logspace(-5, 1.5, 2000)
        self.dj = self.dj.update(pk_table={"k": jnp.asarray(kg), "Pk": jnp.asarray(np.asarray(cosmo.P_lin(kg, 0.0)))})
        kd = self.dj.k_dict
        self.kmesh = jnp.asarray(kd["|k|"])
        self.kvecs = [jnp.asarray(k) for k in kd["k_vecs"]]
        k_safe = jnp.where(self.kmesh == 0, 1.0, self.kmesh)
        self.Pk_mesh = jnp.where(self.kmesh == 0, 0.0, jnp.asarray(np.interp(np.asarray(k_safe), kg, np.asarray(cosmo.P_lin(kg, 0.0)))))
        self.T_mesh = jnp.where(self.kmesh == 0, 0.0, jnp.asarray(np.asarray(cosmo.T_Phi_delta(k_safe, 0.0))))
        self.invT_mesh = jnp.where(self.kmesh == 0, 0.0, 1.0 / jnp.where(self.kmesh == 0, 1.0, self.T_mesh))
        self.norm = (res / L) ** 3
        self.V_cell = (L / res) ** 3
        self.inv_lap = jnp.where(self.kmesh == 0, 0.0, -1.0 / k_safe ** 2)
        # Lagrangian grid and observer
        self.q = jnp.asarray(self.dj.q, dtype=self.dtype)                    # (res,res,res,3)
        self.observer = jnp.asarray(observer if observer is not None else [L / 2] * 3, dtype=self.dtype)
        # shells: scale factor, chi range
        for s in self.shells:
            zc = 0.5 * (s.zlo + s.zhi)
            s.a = 1.0 / (1.0 + zc)
            s.chi_lo, s.chi_hi = float(cosmo.chi(s.zlo)), float(cosmo.chi(s.zhi))
        # growth at shell centres
        self.D = {i: float(cosmo.D(1 / s.a - 1)) for i, s in enumerate(shells)}
        self.f = {i: float(cosmo.f(1 / s.a - 1)) for i, s in enumerate(shells)}
        # selection on the Eulerian grid: chi, z, angular coverage per voxel
        self._build_selection(nz_funcs, ang_mask, mask_nside)
        self.tracers = sorted(set(s.tracer for s in shells))

    # ---- selection ---------------------------------------------------------------------------
    def _build_selection(self, nz_funcs, ang_mask, nside):
        import healpy as hp
        x = (np.arange(self.res) + 0.5) * self.L / self.res
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        r = np.stack([X, Y, Z], -1) - np.asarray(self.observer)
        chi = np.linalg.norm(r, axis=-1)
        zz = np.linspace(0, 6, 3000); chis = self.cosmo.chi(zz)
        z_of_chi = np.interp(chi, chis, zz)
        if ang_mask is not None:
            pix = hp.vec2pix(nside, r[..., 0], r[..., 1], r[..., 2])
            cov = np.asarray(ang_mask)[pix]
        else:
            cov = np.ones_like(chi)
        self.chi_grid, self.z_grid, self.cov_grid = chi, z_of_chi, cov
        # per-shell expected density per voxel (static), and the tracer-level footprint mask
        self.nbar_cell = []
        for s in self.shells:
            inside = (chi >= s.chi_lo) & (chi < s.chi_hi) & (cov > 0)
            n = nz_funcs[s.tracer](z_of_chi) if nz_funcs else np.full_like(chi, 1e-4)
            self.nbar_cell.append(jnp.asarray(np.where(inside, n * self.V_cell * cov, 0.0), dtype=self.dtype))
        self.footprint = {t: np.any([np.asarray(self.nbar_cell[i]) > 0 for i, s in enumerate(self.shells) if s.tracer == t], axis=0)
                          for t in sorted(set(s.tracer for s in self.shells))}

    # ---- linear fields -------------------------------------------------------------------------
    def linear_fields(self, eps, f_NL):
        """From white noise: (delta_NG(k) [rfft], delta_L(x), phi_g(x))."""
        eps = jnp.asarray(eps, dtype=self.dtype)
        feps = jnp.fft.rfftn(eps)
        fdelta = feps * jnp.sqrt(self.Pk_mesh * self.norm)
        fphi = fdelta * self.invT_mesh
        phi = jnp.fft.irfftn(fphi, s=eps.shape)
        phi_ng = phi + f_NL * (phi ** 2 - jnp.mean(phi ** 2))
        fdelta_ng = jnp.fft.rfftn(phi_ng) * self.T_mesh
        delta_L = jnp.fft.irfftn(fdelta_ng, s=eps.shape)
        return fdelta_ng, delta_L, phi

    def tidal_K2(self, fdelta):
        """K^2 = K_ij K_ij with K_ij = (k_i k_j / k^2 - delta_ij/3) delta."""
        k2 = jnp.where(self.kmesh == 0, 1.0, self.kmesh ** 2)
        shape = (self.res,) * 3
        K2 = 0.0
        for i in range(3):
            for j in range(3):
                kern = self.kvecs[i] * self.kvecs[j] / k2 - (1.0 / 3.0 if i == j else 0.0)
                Kij = jnp.fft.irfftn(kern * fdelta, s=shape)
                K2 = K2 + Kij ** 2
        return K2

    # ---- LPT ---------------------------------------------------------------------------------
    def lpt(self, fdelta_ng):
        fphi = fdelta_ng * self.inv_lap
        dj = self.dj.update(ics={"fphi": fphi}).with_lpt(n_order=self.n_order)
        psi = dj._lpt.psi
        return [psi[f"psi_{n}"] for n in range(1, self.n_order + 1)]

    def positions(self, psis, i, rsd=True):
        D, f = self.D[i], self.f[i]
        disp = sum(D ** (n + 1) * p for n, p in enumerate(psis))
        x = self.q + disp
        if rsd:
            vdisp = f * sum((n + 1) * D ** (n + 1) * p for n, p in enumerate(psis))
            r = x - self.observer
            rhat = r / jnp.maximum(jnp.linalg.norm(r, axis=-1, keepdims=True), 1e-3)
            x = x + jnp.sum(vdisp * rhat, axis=-1, keepdims=True) * rhat
        return jnp.mod(x, self.L)

    # ---- galaxy field ---------------------------------------------------------------------------
    def weights(self, delta_L, phi, K2, D, f_NL, b1, b2, bK, bphi, bphid):
        """Lagrangian weights at growth factor D: delta_L, K2 are the z=0 linear fields, phi is primordial."""
        b1L = b1 - 1.0
        b2L = b2 - (8.0 / 21.0) * b1L
        bKL = bK + (2.0 / 7.0) * b1L
        bphidL = bphid - bphi
        dL = D * delta_L
        K2D = D ** 2 * K2
        return (1.0 + b1L * dL + 0.5 * b2L * (dL ** 2 - jnp.mean(dL ** 2)) + bKL * (K2D - jnp.mean(K2D))
                + bphi * f_NL * phi + bphidL * f_NL * dL * phi)

    def scatter(self, pos, w):
        from discodj.core.scatter_and_gather import scatter
        mesh = jnp.zeros((self.res,) * 3, dtype=self.dtype)
        n = self.res ** 3
        return scatter(mesh, pos.reshape(n, 3), n_part_tot=n, res=self.res, boxsize=self.L,
                       dtype_num=64 if self.dtype == jnp.float64 else 32, weight=w.reshape(n, 1),
                       worder=self.worder)

    def __call__(self, eps, theta, layout: ParamLayout, rsd=True):
        """Return {tracer: expected counts field (res^3)}."""
        f_NL = theta[0]
        fdelta_ng, delta_L, phi = self.linear_fields(eps, f_NL)
        K2 = self.tidal_K2(fdelta_ng)
        psis = self.lpt(fdelta_ng)
        out = {t: jnp.zeros((self.res,) * 3, dtype=self.dtype) for t in self.tracers}
        for i, s in enumerate(self.shells):
            _, b1, b2, bK, bphi, bphid = layout.unpack(theta, i)
            w = self.weights(delta_L, phi, K2, self.D[i], f_NL, b1, b2, bK, bphi, bphid)
            rho = self.scatter(self.positions(psis, i, rsd), w)
            out[s.tracer] = out[s.tracer] + self.nbar_cell[i] * rho
        return out

    def linear_galaxy_field(self, eps, theta, layout: ParamLayout, rsd=True):
        """Strictly linear model for operator validation: <N> = nbar V [1 + (b1 + bphi f_NL/T(k) + f mu_z^2) D delta_L(k)]
        (plane-parallel Kaiser along z; no displacement, no b2/bK/bphidelta)."""
        f_NL = theta[0]
        eps = jnp.asarray(eps, dtype=self.dtype)
        fdelta = jnp.fft.rfftn(eps) * jnp.sqrt(self.Pk_mesh * self.norm)
        k2 = jnp.where(self.kmesh == 0, 1.0, self.kmesh ** 2)
        mu2 = jnp.where(self.kmesh == 0, 0.0, self.kvecs[2] ** 2 / k2)
        out = {t: jnp.zeros((self.res,) * 3, dtype=self.dtype) for t in self.tracers}
        for i, s in enumerate(self.shells):
            _, b1, b2, bK, bphi, bphid = layout.unpack(theta, i)
            # (b1 + f mu^2) D delta_L0 + bphi f_NL phi_g, with phi_g = delta_L0 / T(k, z=0) (no growth factor)
            kern = (b1 + (self.f[i] * mu2 if rsd else 0.0)) * self.D[i] + bphi * f_NL * self.invT_mesh
            dg = jnp.fft.irfftn(kern * fdelta, s=(self.res,) * 3)
            out[s.tracer] = out[s.tracer] + self.nbar_cell[i] * (1.0 + dg)
        return out

    def matter_field(self, eps, f_NL=0.0, i=0, rsd=False):
        """1 + delta_m (2LPT) for shell i's scale factor; for tests."""
        fdelta_ng, _, _ = self.linear_fields(eps, f_NL)
        psis = self.lpt(fdelta_ng)
        return self.scatter(self.positions(psis, i, rsd), jnp.ones((self.res,) * 3, dtype=self.dtype))

    # ---- diagnostics -----------------------------------------------------------------------------
    def power_spectrum(self, field, nbins=20, kmin=None, kmax=None, mu_bins=None):
        """Simple shell-averaged P(k) of a real field in the box (LOS = z axis if mu_bins)."""
        f = jnp.fft.rfftn(field - jnp.mean(field)) / self.res ** 3
        P = jnp.abs(f) ** 2 * self.L ** 3
        k = np.asarray(self.kmesh)
        kmin = kmin or 2 * np.pi / self.L
        kmax = kmax or np.pi * self.res / self.L
        edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        idx = np.digitize(k, edges) - 1
        P = np.asarray(P)
        w = np.ones_like(k); w[..., 0] = 0.5; w[..., -1] = 0.5 if self.res % 2 == 0 else 1.0  # rfft half-plane weights
        valid = (idx >= 0) & (idx < nbins)
        if mu_bins is None:
            num = np.bincount(idx[valid], weights=(P * w)[valid], minlength=nbins)
            den = np.bincount(idx[valid], weights=w[valid], minlength=nbins)
            kk = np.bincount(idx[valid], weights=(k * w)[valid], minlength=nbins)
            return kk / np.maximum(den, 1e-30), num / np.maximum(den, 1e-30)
        kz = np.broadcast_to(np.asarray(self.kvecs[2]), k.shape)
        mu = np.where(k > 0, kz / np.where(k > 0, k, 1), 0)
        out = []
        for lo, hi in zip(mu_bins[:-1], mu_bins[1:]):
            sel = valid & (np.abs(mu) >= lo) & (np.abs(mu) < hi)
            num = np.bincount(idx[sel], weights=(P * w)[sel], minlength=nbins)
            den = np.bincount(idx[sel], weights=w[sel], minlength=nbins)
            out.append(num / np.maximum(den, 1e-30))
        kk = np.bincount(idx[valid], weights=(k * w)[valid], minlength=nbins) / np.maximum(np.bincount(idx[valid], weights=w[valid], minlength=nbins), 1e-30)
        return kk, out
