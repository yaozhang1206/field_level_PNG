"""Stage 3: Quijote halo catalogues -> DESI-like tracer samples by mass cut.

Quijote fiducial cosmology (Villaescusa-Navarro et al. 2020): Omega_m = 0.3175, Omega_b = 0.049, h = 0.6711,
n_s = 0.9624, sigma_8 = 0.834, massless neutrinos. A_s is fixed here by matching sigma_8 with the CLASS backend.

Catalogue format on disk (ASCII, header line): mass [Msun/h], x y z [Mpc/h], vx vy vz [km/s], Rs [kpc/h].
Compact cut catalogues are saved as .npz with pos (float32, Mpc/h), vel (float32), mass (float32), and the cut metadata.

The linear bias of a cut is measured from the halo auto-spectrum in the box:
    b_1^2 = [P_hh(k) - 1/n] / P_lin(k, z)   averaged over  k_lo < k < k_hi  (default 0.02-0.06 h/Mpc),
with P_hh from a CIC-assigned 128^3 mesh (deconvolved); and the mass threshold is found by bisection on b_1.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cosmology import CosmoParams, Cosmology  # noqa: E402

L_QUIJOTE = 1000.0  # Mpc/h


def quijote_cosmology(sigma8=0.834, backend="class"):
    """Cosmology object with the Quijote fiducial parameters, A_s set so that sigma_8(z=0) = 0.834."""
    base = CosmoParams(Omega_m=0.3175, Omega_b=0.049, h=0.6711, A_s=2.1e-9, n_s=0.9624, m_nu=0.0)
    c = Cosmology(base, backend=backend)
    s8 = c.sigma8(0.0)
    A_s = 2.1e-9 * (sigma8 / s8) ** 2
    return Cosmology(dataclasses.replace(base, A_s=A_s), backend=backend)


def read_catalogue(path, hosts_only=True):
    """Read either the compact ASCII format (mass x y z vx vy vz Rs) or a Rockstar out_X[_pid].list file.
    Rockstar: columns are taken from the header line (Mvir, X, Y, Z, VX, VY, VZ, PID); with hosts_only, keep PID == -1."""
    with open(path) as f:
        header = f.readline()
    if header.startswith("#ID") or "Mvir" in header:
        names = header.lstrip("#").split()
        col = {n: i for i, n in enumerate(names)}
        d = np.loadtxt(path, comments="#")
        if d.ndim == 1:
            d = d[None, :]
        if hosts_only and "PID" in col:
            d = d[d[:, col["PID"]] < 0]
        return {"mass": d[:, col["Mvir"]], "pos": d[:, [col["X"], col["Y"], col["Z"]]],
                "vel": d[:, [col["VX"], col["VY"], col["VZ"]]]}
    d = np.loadtxt(path, comments="#")
    return {"mass": d[:, 0], "pos": d[:, 1:4], "vel": d[:, 4:7]}


def ngp_mesh(pos, L, res, shift=0.0):
    """Counts in cells on a res^3 periodic mesh; cell j covers [j L/res - shift, (j+1) L/res - shift).
    The forward model with oversample = o block-averages fine CIC nodes 0..o-1 (node spacing L/(o res)), whose
    support is offset by half a fine spacing: use shift = L / (2 o res) to match it."""
    i = np.floor(((np.asarray(pos) + shift) / L * res) % res).astype(np.int64) % res
    return np.bincount((i[:, 0] * res + i[:, 1]) * res + i[:, 2], minlength=res ** 3).reshape((res,) * 3).astype(float)


def fourier_grid(pos, L, res, oversample=4):
    """CIC counts on the fine (oversample*res)^3 mesh, truncated to the res^3 grid's Fourier modes and sampled at the
    coarse nodes (same operation as ForwardModel.downsample 'fourier'), normalised to counts per coarse cell."""
    rf = res * oversample
    fine = cic_mesh(pos, L, rf)
    f = np.fft.rfftn(fine)
    keep = np.array([i for i in range(res) if i != res // 2]); src = np.where(keep < res // 2, keep, keep - res) % rf
    out = np.zeros((res, res, res // 2 + 1), dtype=complex)
    out[np.ix_(keep, keep, np.arange(res // 2))] = f[np.ix_(src, src, np.arange(res // 2))]
    return np.fft.irfftn(out, s=(res,) * 3)          # counts per coarse cell (fine counts summed: x (rf/res)^3 (res/rf)^3)


def redshift_space(pos, vel, z, cosmo, L, axis=2):
    """Displace along `axis` by v (1+z)/H(z): v in km/s (physical peculiar), H in km/s/(Mpc/h) -> Mpc/h; periodic."""
    Ez = float(cosmo.H(z)) / 100.0 if hasattr(cosmo, "H") else np.sqrt(cosmo.p.Omega_m * (1 + z) ** 3 + 1 - cosmo.p.Omega_m)
    out = np.array(pos, dtype=float)
    out[:, axis] = (out[:, axis] + vel[:, axis] * (1 + z) / (100.0 * Ez)) % L
    return out


# ---- power spectrum on a mesh ---------------------------------------------------------------
def cic_mesh(pos, L, res):
    """CIC density (counts) on a res^3 periodic mesh."""
    x = (np.asarray(pos) / L * res) % res
    i0 = np.floor(x).astype(np.int64)
    d = x - i0
    mesh = np.zeros((res,) * 3)
    for dx in (0, 1):
        wx = d[:, 0] if dx else 1 - d[:, 0]
        for dy in (0, 1):
            wy = d[:, 1] if dy else 1 - d[:, 1]
            for dz in (0, 1):
                wz = d[:, 2] if dz else 1 - d[:, 2]
                np.add.at(mesh, ((i0[:, 0] + dx) % res, (i0[:, 1] + dy) % res, (i0[:, 2] + dz) % res), wx * wy * wz)
    return mesh


def power_spectrum(pos, L, res=128, nbins=30, kmax=None):
    """Shell-averaged P(k) of the overdensity of points, CIC-deconvolved, shot noise NOT subtracted.
    Returns k, P, nmodes, shot = L^3 / N."""
    n = len(pos)
    mesh = cic_mesh(pos, L, res)
    delta = mesh / mesh.mean() - 1.0
    f = np.fft.rfftn(delta) / res ** 3
    kf = 2 * np.pi / L
    kx = np.fft.fftfreq(res, d=1.0 / res) * kf
    kz = np.fft.rfftfreq(res, d=1.0 / res) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    # CIC window deconvolution
    def sinc(q):
        return np.sinc(q / (2 * np.pi) * L / res)
    W = (sinc(KX) * sinc(KY) * sinc(KZ)) ** 2
    f = f / W
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    kmax = kmax or np.pi * res / L
    edges = np.linspace(kf, kmax, nbins + 1)
    P3 = np.abs(f) ** 2 * L ** 3
    # independent modes: rfft half-plane; weight kz=0 and kz=Nyquist planes once, others twice
    w = np.full(K.shape, 2.0); w[..., 0] = 1.0
    if res % 2 == 0:
        w[..., -1] = 1.0
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    Pw = np.bincount(idx[ok], weights=(P3 * w).ravel()[ok], minlength=nbins)
    kw = np.bincount(idx[ok], weights=(K * w).ravel()[ok], minlength=nbins)
    nm = np.bincount(idx[ok], weights=w.ravel()[ok], minlength=nbins)
    good = nm > 0
    return kw[good] / nm[good], Pw[good] / nm[good], nm[good], L ** 3 / n


def linear_bias(pos, cosmo, z, L=L_QUIJOTE, klo=0.02, khi=0.06, res=128):
    """b_1 from the halo auto-spectrum vs P_lin(k, z) over klo < k < khi (mode-count weighted)."""
    k, P, nm, shot = power_spectrum(pos, L, res=res)
    sel = (k > klo) & (k < khi)
    Plin = np.asarray(cosmo.P_lin(k[sel], z))
    b2 = (P[sel] - shot) / Plin
    b1sq = np.sum(nm[sel] * b2) / np.sum(nm[sel])
    # error from the scatter of the per-bin estimates
    err = np.sqrt(np.sum(nm[sel] * (b2 - b1sq) ** 2) / np.sum(nm[sel]) / max(sel.sum() - 1, 1))
    b1 = np.sqrt(max(b1sq, 0.0))
    return b1, err / (2 * b1) if b1 > 0 else np.inf, {"k": k, "P": P, "shot": shot, "nmodes": nm}


def mass_threshold_for_bias(cat, cosmo, z, b1_target, L=L_QUIJOTE, lo=None, hi=None, n_iter=14, **kw):
    """Bisection in log M_min so that the cut M > M_min has linear bias b1_target."""
    m = cat["mass"]
    lo = np.log10(lo or m.min()); hi = np.log10(hi or np.quantile(m, 0.999))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        sel = m > 10 ** mid
        b1, _, _ = linear_bias(cat["pos"][sel], cosmo, z, L, **kw)
        if b1 < b1_target:
            lo = mid
        else:
            hi = mid
    return 10 ** (0.5 * (lo + hi))


def cut_and_save(cat, mmin, out, meta=None):
    sel = cat["mass"] > mmin
    np.savez_compressed(out, pos=cat["pos"][sel].astype(np.float32), vel=cat["vel"][sel].astype(np.float32),
                        mass=cat["mass"][sel].astype(np.float32), mmin=mmin, L=L_QUIJOTE,
                        meta=json.dumps(meta or {}))
    return int(sel.sum())
