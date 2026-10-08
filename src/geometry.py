"""Survey geometry for the DESI DR1 LRG/QSO Fisher: n(z), sky area, effective volume.

Inputs: public DR1 LSS clustering catalogues (v1.5), $DESI_DR1_DIR/{TRACER}_{NGC,SGC}_clustering.dat.fits.
The NX column is the (completeness-scaled) comoving density n(z) in (h/Mpc)^3 used for the FKP weights
(catalogue FKP weights use the BAO P0 = 1e4 (LRG) / 6e3 (QSO); here we recompute with the PNG P0).

Two n(z) estimates are provided:
  * "nx":   mean of NX in redshift bins (what DESI used; includes completeness).
  * "count": N_w(z) / dV(z), with dV = Omega_sky * chi^2 dchi (requires the footprint area).
Footprint area from the galaxy catalogue itself: occupied HEALPix pixels (nside=128, ~0.21 deg^2).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from cosmology import Cosmology, C_OVER_H0

DEG2_SKY = 41252.96125


def catalog_path(tracer, cap):
    d = os.environ.get("DESI_DR1_DIR", "/work/hdd/bdne/yzhang116/desi_dr1")
    return Path(d) / f"{tracer}_{cap}_clustering.dat.fits"


def load_catalog(tracer, caps=("NGC", "SGC"), zmin=None, zmax=None):
    import fitsio
    cols = ["RA", "DEC", "Z", "NX", "WEIGHT"]
    parts = [fitsio.read(catalog_path(tracer, c), columns=cols) for c in caps]
    cat = np.concatenate(parts)
    if zmin is not None:
        cat = cat[cat["Z"] >= zmin]
    if zmax is not None:
        cat = cat[cat["Z"] <= zmax]
    return cat


def footprint_area_deg2(cat, nside=128):
    import healpy as hp
    pix = hp.ang2pix(nside, cat["RA"], cat["DEC"], lonlat=True)
    return len(np.unique(pix)) * hp.nside2pixarea(nside, degrees=True)


_DR1_AREA_CACHE = Path(__file__).resolve().parents[1] / "results" / "cache" / "dr1_footprint_area.json"


def footprint_area_from_randoms(files, nside=512, min_count=2):
    """Geometric footprint from random catalogues: area of HEALPix pixels holding >= min_count randoms.
    DR1 clustering randoms are completeness-subsampled (~2500 c deg^-2), so at nside=512 (0.013 deg^2)
    a pixel inside the footprint holds >~ 10 randoms even at c ~ 0.3."""
    import fitsio, healpy as hp
    cnt = np.zeros(hp.nside2npix(nside))
    for f in files:
        r = fitsio.read(f, columns=["RA", "DEC"])
        cnt += np.bincount(hp.ang2pix(nside, r["RA"], r["DEC"], lonlat=True), minlength=cnt.size)
    return float((cnt >= min_count).sum() * hp.nside2pixarea(nside, degrees=True))


def dr1_footprint_area(default=9437.0):
    """DR1 geometric footprint (deg^2). Cached from the QSO randoms when present; otherwise the
    LRG+QSO galaxy pixel footprint at nside=256 (9437 deg^2, an overestimate by a few %)."""
    import json
    if _DR1_AREA_CACHE.exists():
        return json.load(open(_DR1_AREA_CACHE))["area_full_deg2"]
    d = Path(os.environ.get("DESI_DR1_DIR", "/work/hdd/bdne/yzhang116/desi_dr1"))
    files = [d / f"QSO_{c}_0_clustering.ran.fits" for c in ("NGC", "SGC")]
    if all(f.exists() for f in files):
        A = footprint_area_from_randoms(files)
        _DR1_AREA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"area_full_deg2": A, "source": "QSO randoms, nside=512, >=2 randoms/pixel"}, open(_DR1_AREA_CACHE, "w"))
        return A
    return default


class Geometry:
    """Tabulated n(z), dV/dz and effective volume for one tracer."""

    def __init__(self, cosmo: Cosmology, tracer: str, zmin: float, zmax: float, nbins: int = 60,
                 cat=None, area_deg2=None, area_full_deg2=None):
        """area_full_deg2: geometric (tiled) footprint over which the density field is sampled.
        Sample variance is set by this volume; shot noise by the *observed* density c(x) n_NX, where
        c = area_eff / area_full is the mean completeness. Default: DR1 footprint from the randoms if
        available (see footprint_area_from_randoms), else the galaxy pixel footprint."""
        self.cosmo, self.tracer, self.zmin, self.zmax = cosmo, tracer, zmin, zmax
        cat = load_catalog(tracer, zmin=zmin, zmax=zmax) if cat is None else cat
        self.area_deg2 = footprint_area_deg2(cat) if area_deg2 is None else area_deg2
        self.area_full_deg2 = area_full_deg2 if area_full_deg2 is not None else dr1_footprint_area()
        self.fsky = self.area_deg2 / DEG2_SKY
        self.N_w = float(cat["WEIGHT"].sum())
        edges = np.linspace(zmin, zmax, nbins + 1)
        self.z = 0.5 * (edges[1:] + edges[:-1])
        self.dz = np.diff(edges)
        idx = np.clip(np.digitize(cat["Z"], edges) - 1, 0, nbins - 1)
        w = cat["WEIGHT"]
        self.Nw_z = np.bincount(idx, weights=w, minlength=nbins)
        self.nz_nx = np.bincount(idx, weights=w * cat["NX"], minlength=nbins) / np.maximum(self.Nw_z, 1e-300)
        dVdz_pix = cosmo.dV_dz(self.z, self.fsky)           # (Mpc/h)^3 per unit z, pixel footprint
        self.nz_count = self.Nw_z / (dVdz_pix * self.dz)
        # Effective area consistent with NX: the area over which n_NX(z) integrates to N_w.
        # (DR1 tiles are incomplete, so the pixel footprint overestimates the effective area;
        #  NX is the density per *effective* volume, as used for DESI's FKP weights.)
        self.area_eff_deg2 = self.N_w / float(np.sum(self.nz_nx * cosmo.dV_dz(self.z, 1.0 / DEG2_SKY) * self.dz))
        self.fsky_eff = self.area_eff_deg2 / DEG2_SKY
        self.dVdz = cosmo.dV_dz(self.z, self.fsky_eff)      # used with n = NX
        self.dVdz_pix = dVdz_pix                            # used with n = count
        self.cbar = self.area_eff_deg2 / self.area_full_deg2            # mean completeness
        self.dVdz_full = cosmo.dV_dz(self.z, self.area_full_deg2 / DEG2_SKY)
        self.chi = cosmo.chi(self.z)

    def nz(self, kind="nx"):
        return self.nz_nx if kind == "nx" else self.nz_count

    def dV(self, kind="nx"):
        """Volume elements matching nz(kind): 'nx' -> effective area, 'count' -> pixel area,
        'full' -> geometric footprint (use with nz('nx') for weights and nz_obs() for shot noise)."""
        if kind == "full":
            return self.dVdz_full * self.dz
        return (self.dVdz if kind == "nx" else self.dVdz_pix) * self.dz

    def nz_obs(self):
        """Observed (completeness-reduced) density c_bar * n_NX, for the shot noise over the full footprint."""
        return self.cbar * self.nz_nx

    def V_total(self, kind="nx"):
        return float(np.sum(self.dV(kind)))

    def V_eff(self, P0, kind="nx"):
        """V_eff = int [n P0 / (1 + n P0)]^2 dV  (FKP, Tegmark 1997)."""
        n = self.nz(kind)
        return float(np.sum((n * P0 / (1 + n * P0)) ** 2 * self.dV(kind)))

    def nbar_eff(self, P0, kind="nx"):
        """Effective density: FKP-weighted mean of n(z), used as the shot-noise density at z_eff."""
        n = self.nz(kind)
        w = (n * P0 / (1 + n * P0)) ** 2 * self.dV(kind)
        return float(np.sum(w * n) / np.sum(w))

    def z_eff(self, P0, kind="nx"):
        n = self.nz(kind)
        w = (n * P0 / (1 + n * P0)) ** 2 * self.dV(kind)
        return float(np.sum(w * self.z) / np.sum(w))

    def summary(self, P0, kind="nx"):
        return dict(tracer=self.tracer, zmin=self.zmin, zmax=self.zmax, area_pix_deg2=self.area_deg2,
                    area_eff_deg2=self.area_eff_deg2, area_full_deg2=self.area_full_deg2, cbar=self.cbar, N_w=self.N_w, V_total=self.V_total(kind), V_eff=self.V_eff(P0, kind),
                    nbar_eff=self.nbar_eff(P0, kind), z_eff=self.z_eff(P0, kind), P0=P0)


_DR1_MASK_CACHE = Path(__file__).resolve().parents[1] / "results" / "cache" / "dr1_angular_mask_nside{nside}.npy"


def dr1_angular_mask(nside=64):
    """Fractional angular coverage map (RING, nside) of the DR1 footprint from the QSO randoms:
    coverage = min(1, N_ran(pix) / N_full) with N_full the 90th percentile of occupied pixels."""
    import healpy as hp
    f = Path(str(_DR1_MASK_CACHE).format(nside=nside))
    if f.exists():
        return np.load(f)
    import fitsio
    d = Path(os.environ.get("DESI_DR1_DIR", "/work/hdd/bdne/yzhang116/desi_dr1"))
    cnt = np.zeros(hp.nside2npix(nside))
    for c in ("NGC", "SGC"):
        r = fitsio.read(d / f"QSO_{c}_0_clustering.ran.fits", columns=["RA", "DEC"])
        cnt += np.bincount(hp.ang2pix(nside, r["RA"], r["DEC"], lonlat=True), minlength=cnt.size)
    full = np.percentile(cnt[cnt > 0], 90)
    cov = np.clip(cnt / full, 0, 1)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, cov)
    return cov
