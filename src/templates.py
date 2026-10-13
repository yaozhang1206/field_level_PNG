"""Stage 4: DESI DR1 imaging-systematics templates and radial (n(z)) modes for the field-level model.

Imaging templates come from the DR1 LSS healpix maps (v1.5 hpmaps, nside 256 NESTED, separate N = BASS/MzLS and
S = DECaLS+DES files per tracer). Depths are converted to 5-sigma magnitudes, m = 22.5 - 2.5 log10(5 / sqrt(ivar)).
Each template is standardised (zero mean, unit variance) over the DR1 footprint separately in each photometric region,
weighted by the angular coverage.

Contamination model (multiplicative, per tracer t and region r):
    <N_t>(x) -> <N_t>(x) [1 + sum_k a_{t,r,k} tau_k(n^(x)) 1_r(n^(x))]
To first order a template adds the data-space vector  T_{t,r,k}(x) = <N_t>(x) tau_k(n^) 1_r  with amplitude a; marginalising
a with a flat prior is mode deprojection: C -> C + Lambda T T^T, Lambda -> infinity (Woodbury in FieldFisher).

Radial modes model an unknown n(z) shape per redshift shell s:  <N_s>(x) -> <N_s>(x) [1 + sum_l c_{s,l} P_l(u)],
u = 2 (chi - chi_lo)/(chi_hi - chi_lo) - 1, Legendre P_l, l = 0..L (l = 0 is the shell normalisation, as removed by the
DESI radial integral constraint).
"""
from __future__ import annotations

import os
import numpy as np

HPDIR = os.environ.get("DESI_HPMAPS_DIR", "/work/hdd/bdne/yzhang116/desi_dr1/hpmaps")
NSIDE = 256
DEPTHS = {"PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_Z", "PSFDEPTH_W1", "PSFDEPTH_W2", "GALDEPTH_G", "GALDEPTH_R", "GALDEPTH_Z"}
# template sets (DESI DR1 QSO/LRG regressions use subsets of these)
SETS = {
    "imaging3": ["EBV", "STARDENS", "PSFDEPTH_Z"],
    "imaging9": ["EBV", "STARDENS", "PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_Z", "PSFDEPTH_W1", "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z"],
}


def _read(tracer, region, names):
    import fitsio
    d = fitsio.read(os.path.join(HPDIR, f"{tracer}_mapprops_healpix_nested_nside256_{region}.fits"), columns=list(names))
    out = {}
    for n in names:
        x = np.asarray(d[n], dtype=float)
        if n in DEPTHS:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(x > 0, 22.5 - 2.5 * np.log10(5.0 / np.sqrt(x)), np.nan)
        x[~np.isfinite(x) | (np.asarray(d[n]) == 0)] = np.nan
        out[n] = x
    return out


def template_maps(tracer, names, coverage_nest=None, clip=5.0):
    """Standardised template maps per region: {region: (len(names), npix) array (NaN outside)}, and the region masks
    {region: bool npix}. coverage_nest: angular coverage (NESTED nside 256) used as weights and footprint."""
    import healpy as hp
    npix = hp.nside2npix(NSIDE)
    raw = {r: _read(tracer, r, names) for r in ("N", "S")}
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix), nest=True)
    dec = 90.0 - np.degrees(theta); ra = np.degrees(phi)
    ngc = (ra > 80) & (ra < 300)
    inN = np.isfinite(raw["N"][names[0]]) & (dec > 32.375) & ngc
    inS = np.isfinite(raw["S"][names[0]]) & ~inN
    cov = np.ones(npix) if coverage_nest is None else np.asarray(coverage_nest)
    regions, maps = {"N": inN & (cov > 0), "S": inS & (cov > 0)}, {}
    for r, m in regions.items():
        T = np.full((len(names), npix), np.nan)
        w = cov[m]
        for k, n in enumerate(names):
            x = raw[r][n][m]
            good = np.isfinite(x)
            mu = np.average(x[good], weights=w[good]); sd = np.sqrt(np.average((x[good] - mu) ** 2, weights=w[good]))
            y = np.clip((x - mu) / sd, -clip, clip)
            y[~good] = 0.0
            T[k, m] = y
        maps[r] = T
    return maps, regions


def coverage_nest256():
    """DR1 angular coverage (from geometry.dr1_angular_mask, nside 64 RING) evaluated on nside-256 NESTED pixels."""
    import healpy as hp
    from geometry import dr1_angular_mask
    cov64 = dr1_angular_mask(64)
    th, ph = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), nest=True)
    return cov64[hp.ang2pix(64, th, ph)]


def grid_directions(fm):
    """NESTED nside-256 pixel index of every voxel of the forward-model grid (direction from the observer)."""
    import healpy as hp
    x = (np.arange(fm.res) + 0.5) * fm.L / fm.res
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.stack([X, Y, Z], -1) - np.asarray(fm.observer)
    return hp.vec2pix(NSIDE, r[..., 0].ravel(), r[..., 1].ravel(), r[..., 2].ravel(), nest=True)


def data_space_templates(fm, tracers, idx, imaging=None, per_shell=False, radial_order=None):
    """Template vectors in the FieldFisher / FieldPosterior data ordering (tracers concatenated, footprint voxels).
    imaging: list of map names (or a SETS key) -> one vector per (tracer, region, template) [per shell if per_shell];
    radial_order: L -> Legendre l = 0..L per shell. Returns (T [k, n_data], labels)."""
    if isinstance(imaging, str):
        imaging = SETS[imaging]
    offs, n_data = {}, 0
    for t in tracers:
        offs[t] = n_data; n_data += len(idx[t])
    vecs, labels = [], []
    pix = grid_directions(fm) if imaging else None
    cov = coverage_nest256() if imaging else None
    for t in tracers:
        sh = [i for i, s in enumerate(fm.shells) if s.tracer == t]
        ids = np.asarray(idx[t])
        groups = [[i] for i in sh] if per_shell else [sh]
        if imaging:
            maps, regions = template_maps(t, imaging, cov)
            for g in groups:
                mean = sum(np.asarray(fm.nbar_cell[i]).ravel() for i in g)[ids]
                for r in ("N", "S"):
                    inr = regions[r][pix][ids]
                    for k, n in enumerate(imaging):
                        tau = np.nan_to_num(maps[r][k][pix][ids])
                        v = np.zeros(n_data); v[offs[t]:offs[t] + len(ids)] = mean * tau * inr
                        if np.any(v):
                            vecs.append(v); labels.append(f"{t}:{'s'.join(map(str, g)) if per_shell else 'all'}:{r}:{n}")
        if radial_order is not None:
            from numpy.polynomial.legendre import legval
            chi = fm.chi_grid.ravel()[ids]
            for i in sh:
                s = fm.shells[i]
                mean = np.asarray(fm.nbar_cell[i]).ravel()[ids]
                u = np.clip(2 * (chi - s.chi_lo) / (s.chi_hi - s.chi_lo) - 1, -1, 1)
                for l in range(radial_order + 1):
                    c = np.zeros(l + 1); c[l] = 1
                    v = np.zeros(n_data); v[offs[t]:offs[t] + len(ids)] = mean * legval(u, c)
                    vecs.append(v); labels.append(f"{t}:shell{i}:P{l}")
    return np.array(vecs), labels


def grid_template_maps(fm, tracers, imaging="imaging9"):
    """Template maps on the forward-model grid for ParamLayout(templates=...) / ForwardModel.set_templates:
    one map per (tracer, region, template), zero outside the region. Returns (labels [(tracer, label)], maps (K, res^3))."""
    if isinstance(imaging, str):
        imaging = SETS[imaging]
    pix = grid_directions(fm); cov = coverage_nest256()
    labels, maps = [], []
    for t in tracers:
        tm, regions = template_maps(t, imaging, cov)
        for r in ("N", "S"):
            inr = regions[r][pix]
            for k, n in enumerate(imaging):
                m = np.nan_to_num(tm[r][k][pix]) * inr
                if np.any(m):
                    labels.append((t, f"{r}:{n}")); maps.append(m)
    return labels, np.array(maps)


def wsys_contamination(fm, tracer):
    """Observed DR1 imaging modulation for a tracer on the grid: 1 / <WEIGHT_SYS> per nside-256 pixel from the DR1
    clustering catalogue (the DESI systematics weights undo it), normalised to unit mean over the footprint."""
    import fitsio, healpy as hp
    d = os.environ.get("DESI_DR1_DIR", "/work/hdd/bdne/yzhang116/desi_dr1")
    npix = hp.nside2npix(NSIDE); sw = np.zeros(npix); cnt = np.zeros(npix)
    for c in ("NGC", "SGC"):
        x = fitsio.read(os.path.join(d, f"{tracer}_{c}_clustering.dat.fits"), columns=["RA", "DEC", "WEIGHT_SYS"])
        p = hp.ang2pix(NSIDE, x["RA"], x["DEC"], lonlat=True, nest=True)
        sw += np.bincount(p, weights=x["WEIGHT_SYS"], minlength=npix); cnt += np.bincount(p, minlength=npix)
    # smooth over empty pixels: use nside-64 parents where a 256 pixel has no objects
    w = np.where(cnt > 0, sw / np.maximum(cnt, 1), np.nan)
    par = np.arange(npix) // 16
    sw64 = np.bincount(par, weights=np.nan_to_num(sw), minlength=npix // 16); c64 = np.bincount(par, weights=cnt, minlength=npix // 16)
    w = np.where(np.isfinite(w) & (cnt >= 3), w, np.where(c64[par] > 0, sw64[par] / np.maximum(c64[par], 1), 1.0))
    f = 1.0 / w
    pix = grid_directions(fm)
    g = f[pix].reshape((fm.res,) * 3)
    fp = fm.footprint[tracer]
    return g / g[fp].mean()
