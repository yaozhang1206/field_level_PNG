"""Linear cosmology for the DESI f_NL Fisher analysis.

Conventions (Slosar et al. 2008 / Chaussidon et al. 2024, "LSS convention"):

    delta_m(k, z) = T_Phi->delta(k, z) * Phi(k)
    T_Phi->delta(k, z) = (2/3) k^2 T(k) D_md(z) c^2 / (Omega_m H0^2)

with  T(k) -> 1 as k -> 0,  D_md(z) the linear growth factor normalised to D_md = a
in matter domination, Phi the primordial potential during matter domination
(Phi = 3/5 zeta), and

    P_Phi(k) = (9/25) (2 pi^2 / k^3) A_s (k / k_pivot)^(n_s - 1),   k_pivot = 0.05 Mpc^-1.

Then P_lin(k, z) = T_Phi->delta(k, z)^2 P_Phi(k) and the scale-dependent bias is
    Delta b(k, z) = b_phi f_NL / T_Phi->delta(k, z).

Units: k in h/Mpc, P in (Mpc/h)^3 throughout. All arrays are jax arrays in float64 so the
whole P(k) model is differentiable with jax.

Transfer function backends:
  * "discoeb" (default): DISCO-EB Boltzmann solver via discodj.cosmology.transfer_functions.disco_eb
  * "class":              classy (CLASS). Used as the cross-check (<= 0.5 % on P_lin).
  * "eh":                 Eisenstein-Hu no-wiggle-with-baryons fit (fast fallback for tests only).
Transfer functions are cached on disk (results/cache) because the Boltzmann solve is slow.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

C_KM_S = 299792.458           # km/s
C_OVER_H0 = 2997.92458        # Mpc/h  (c / (100 km/s/Mpc))
K_PIVOT_MPC = 0.05            # Mpc^-1
DELTA_C = 1.686

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "results" / "cache"


@dataclass(frozen=True)
class CosmoParams:
    """Planck 2018 (TT,TE,EE+lowE+lensing) base-LCDM, as fixed in Chaussidon et al. 2024."""
    Omega_m: float = 0.3153
    Omega_b: float = 0.0493
    h: float = 0.6736
    A_s: float = 2.1e-9
    n_s: float = 0.9649
    m_nu: float = 0.06        # eV, one massive species (as in DISCO-EB's internal setting)
    T_cmb: float = 2.72548

    @property
    def Omega_c(self):
        return self.Omega_m - self.Omega_b

    @property
    def Omega_L(self):
        return 1.0 - self.Omega_m

    def key(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


PLANCK18 = CosmoParams()


# ----------------------------------------------------------------------------------------------
# Transfer functions
# ----------------------------------------------------------------------------------------------
def _transfer_discoeb(cp: CosmoParams, k: np.ndarray, nmodes: int = 256) -> np.ndarray:
    """T(k) from DISCO-EB (normalised to 1 at k -> 0), k in h/Mpc.

    Calls discoeb directly (the discodj wrapper targets an older discoeb API). Uses the total
    matter perturbation delta_m (idx=4) at a=1; the overall amplitude is irrelevant because we
    normalise T(kmin)=1 and fix the amplitude with A_s.
    """
    from discoeb.background import evolve_background
    from discoeb.perturbations import evolve_perturbations, get_power

    Tnu = (4.0 / 11.0) ** (1.0 / 3.0)
    N_nu_mass, Neff = 1.0, 3.046
    param = dict(Omegam=cp.Omega_m, Omegab=cp.Omega_b, w_DE_0=-0.99, w_DE_a=0.0, cs2_DE=0.99,
                 Omegak=0.0, A_s=cp.A_s, n_s=cp.n_s, H0=100 * cp.h, Tcmb=cp.T_cmb, YHe=0.248,
                 Neff=Neff - N_nu_mass * (Tnu / ((4.0 / 11.0) ** (1.0 / 3.0))) ** 4,
                 Nmnu=N_nu_mass, mnu=cp.m_nu, k_p=K_PIVOT_MPC)
    param = evolve_background(param=param, thermo_module="RECFAST")
    kmin_mpc, kmax_mpc = 1e-5, float(k[-1]) * cp.h
    y, kmodes, param = evolve_perturbations(param=param, kmin=kmin_mpc, kmax=kmax_mpc, num_k=nmodes,
                                            aexp_out=jnp.array([1.0]), lmaxg=11, lmaxgp=11, lmaxr=11,
                                            lmaxnu=11, nqmax=3, max_steps=2048, rtol=1e-5, atol=1e-5)
    Pk = get_power(k=kmodes, y=y[:, 0, :], idx=4, param=param)      # Mpc^3, k in 1/Mpc
    k_h = np.asarray(kmodes) / cp.h
    T = np.sqrt(np.asarray(Pk) * cp.h ** 3 / k_h ** cp.n_s)
    T /= T[0]
    return np.exp(np.interp(np.log(k), np.log(k_h), np.log(T)))


def _class_instance(cp: CosmoParams, z_max=4.0, k_max_h=5.0):
    from classy import Class
    cosmo = Class()
    cosmo.set({
        "output": "mPk",
        "P_k_max_h/Mpc": k_max_h,
        "z_max_pk": z_max,
        "h": cp.h, "omega_b": cp.Omega_b * cp.h ** 2,
        "omega_cdm": (cp.Omega_m - cp.Omega_b) * cp.h ** 2 - cp.m_nu / 93.14,  # Omega_m includes nu
        "A_s": cp.A_s, "n_s": cp.n_s, "T_cmb": cp.T_cmb,
        "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": cp.m_nu,
        "k_pivot": K_PIVOT_MPC,
        "YHe": 0.248,
    })
    cosmo.compute()
    return cosmo


def _transfer_eh(cp: CosmoParams, k: np.ndarray) -> np.ndarray:
    from discodj.cosmology.cosmology import Cosmology as DJCosmology
    from discodj.cosmology.transfer_functions import eisenstein_hu
    dj = DJCosmology(Omega_c=cp.Omega_c, Omega_b=cp.Omega_b, h=cp.h, sigma8=0.8, n_s=cp.n_s,
                     dtype_num=64)
    T = eisenstein_hu(dj, jnp.asarray(k), Tcmb=cp.T_cmb)
    return np.asarray(T / T[0], dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# Growth
# ----------------------------------------------------------------------------------------------
def _growth_tables(cp: CosmoParams):
    """Return (a_grid, D_md(a), f(a)) with D_md -> a at early times. Uses DISCO-DJ's growth ODE
    (matter + Lambda, no radiation; fine for z < 4)."""
    from discodj.cosmology.cosmology import Cosmology as DJCosmology
    dj = DJCosmology(Omega_c=cp.Omega_c, Omega_b=cp.Omega_b, h=cp.h, sigma8=0.8, n_s=cp.n_s,
                     dtype_num=64)
    dj = dj.compute_timetables(None)
    a = np.logspace(-3, 0, 2000)
    D1 = np.asarray(dj.Dplus(jnp.asarray(a)))            # normalised D(a=1)=1
    D_unnormed_at_1 = float(dj.timetables["Dplus_unnormed_at_1"])  # D(a=1) in units where D->a early
    D_md = D1 * D_unnormed_at_1
    f = np.asarray(dj.growth_rate(jnp.asarray(a)))
    return a, D_md, f


# ----------------------------------------------------------------------------------------------
# Main object
# ----------------------------------------------------------------------------------------------
class Cosmology:
    """Linear cosmology with cached transfer function and growth tables.

    All public methods accept k in h/Mpc and return jax float64 arrays, differentiable in k.
    """

    def __init__(self, params: CosmoParams = PLANCK18, backend: str = "discoeb",
                 kmin: float = 1e-4, kmax: float = 5.0, nk: int = 600, use_cache: bool = True):
        self.p = params
        self.backend = backend
        self.k_grid = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.T_grid = self._get_transfer(use_cache)
        self._a, self._D_md, self._f = _growth_tables(params)
        self._lnk = jnp.log(self.k_grid)
        self._lnT = jnp.log(self.T_grid)

    # ---- cache -------------------------------------------------------------------------------
    def _get_transfer(self, use_cache):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"T_{self.backend}_{self.p.key()}_{len(self.k_grid)}_{self.k_grid[0]:.1e}_{self.k_grid[-1]:.1e}.npz"
        f = CACHE_DIR / tag
        if use_cache and f.exists():
            return np.load(f)["T"]
        if self.backend == "discoeb":
            T = _transfer_discoeb(self.p, self.k_grid)
        elif self.backend == "class":
            T = self._transfer_class(self.k_grid)
        elif self.backend == "eh":
            T = _transfer_eh(self.p, self.k_grid)
        else:
            raise ValueError(self.backend)
        np.savez(f, T=T, k=self.k_grid)
        return T

    def _transfer_class(self, k):
        """T(k) derived from CLASS P_lin(k, z=0) and our own normalisation, so that the
        'class' backend goes through exactly the same P_Phi / T_Phi->delta formulae."""
        cls = _class_instance(self.p)
        h = self.p.h
        Pk = np.array([cls.pk_lin(kk * h, 0.0) * h ** 3 for kk in k])   # (Mpc/h)^3
        a_, Dmd, _ = _growth_tables(self.p)
        D0 = Dmd[-1]
        PPhi = np.asarray(self.P_Phi(jnp.asarray(k)))
        TPhi = np.sqrt(Pk / PPhi)
        T = TPhi / ((2.0 / 3.0) * k ** 2 * D0 * C_OVER_H0 ** 2 / self.p.Omega_m)
        return T / T[0]

    # ---- primordial ---------------------------------------------------------------------------
    def P_Phi(self, k):
        """Primordial potential power spectrum (matter-domination Phi = 3/5 zeta), (Mpc/h)^3."""
        k = jnp.asarray(k)
        kp = K_PIVOT_MPC / self.p.h                      # h/Mpc
        return (9.0 / 25.0) * (2 * jnp.pi ** 2 / k ** 3) * self.p.A_s * (k / kp) ** (self.p.n_s - 1)

    # ---- transfer / growth --------------------------------------------------------------------
    def T(self, k):
        k = jnp.asarray(k)
        return jnp.exp(jnp.interp(jnp.log(k), self._lnk, self._lnT))

    def D_md(self, z):
        """Growth factor normalised to a in matter domination."""
        a = 1.0 / (1.0 + jnp.asarray(z))
        return jnp.interp(a, jnp.asarray(self._a), jnp.asarray(self._D_md))

    def D(self, z):
        """Growth factor normalised to D(z=0) = 1."""
        return self.D_md(z) / self._D_md[-1]

    def f(self, z):
        a = 1.0 / (1.0 + jnp.asarray(z))
        return jnp.interp(a, jnp.asarray(self._a), jnp.asarray(self._f))

    def T_Phi_delta(self, k, z):
        """T_Phi->delta(k, z): delta_m = T_Phi->delta * Phi.  Dimensionless."""
        k = jnp.asarray(k)
        return (2.0 / 3.0) * k ** 2 * self.T(k) * self.D_md(z) * C_OVER_H0 ** 2 / self.p.Omega_m

    def P_lin(self, k, z):
        """Linear matter power spectrum (Mpc/h)^3."""
        return self.T_Phi_delta(k, z) ** 2 * self.P_Phi(k)

    # ---- background -----------------------------------------------------------------------------
    def E(self, z):
        z = jnp.asarray(z)
        return jnp.sqrt(self.p.Omega_m * (1 + z) ** 3 + self.p.Omega_L)

    def chi(self, z):
        """Comoving distance in Mpc/h (flat LCDM, radiation ignored)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        zz = np.linspace(0, z.max(), 4000)
        integrand = 1.0 / np.asarray(self.E(zz))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zz))])
        out = C_OVER_H0 * np.interp(z, zz, cum)
        return out if out.size > 1 else float(out[0])

    def dV_dz(self, z, fsky=1.0):
        """Comoving volume per unit redshift, (Mpc/h)^3, times sky fraction."""
        chi = np.asarray(self.chi(z))
        return 4 * np.pi * fsky * chi ** 2 * C_OVER_H0 / np.asarray(self.E(z))

    # ---- diagnostics --------------------------------------------------------------------------
    def sigma8(self, z=0.0):
        k = jnp.asarray(self.k_grid)
        x = 8.0 * k
        W = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x ** 3
        integrand = k ** 3 * self.P_lin(k, z) * W ** 2 / (2 * jnp.pi ** 2)
        return float(jnp.sqrt(jnp.trapezoid(integrand, jnp.log(k))))


def bias_phi(b1, p=1.0, delta_c=DELTA_C):
    """Universal-mass-function relation b_phi = 2 delta_c (b1 - p)."""
    return 2.0 * delta_c * (b1 - p)


if __name__ == "__main__":
    import sys
    be = sys.argv[1] if len(sys.argv) > 1 else "discoeb"
    c = Cosmology(backend=be)
    print(f"backend={be}  sigma8(z=0)={c.sigma8():.4f}   D_md(0)={float(c.D_md(0.0)):.4f}  "
          f"f(0.733)={float(c.f(0.733)):.4f}  chi(3.1)={c.chi(3.1):.0f} Mpc/h")
