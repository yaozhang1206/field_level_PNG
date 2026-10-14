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


# ================================================================================================================
# DESI DR1 PNG-analysis template scheme (Chaussidon et al. 2024, arXiv:2411.17623, Sec. 5.1.1; regressis conventions)
# ================================================================================================================
DUSTDIR = os.environ.get("DESI_DUST_DIR", "/work/hdd/bdne/yzhang116/desi_dr1/dust")
FOOTPRINT = os.environ.get("DESI_DR9_FOOTPRINT", "/work/hdd/bdne/yzhang116/desi_dr1/regressis_data/Legacy_Imaging_DR9_footprint_256.fits")
# features per tracer and redshift bin (bins = our shells); names as in the DR1 hpmaps, plus EBV_DIFF_GR / EBV_DIFF_RZ
DESI_FEATURES = {
    "LRG": {(0.4, 0.6): ["STARDENS", "HI", "PSFSIZE_R", "GALDEPTH_Z", "GALDEPTH_R", "PSFDEPTH_W1"],
            (0.6, 0.8): ["STARDENS", "HI", "PSFSIZE_R", "GALDEPTH_Z", "GALDEPTH_G", "PSFDEPTH_W1"],
            (0.8, 1.1): ["STARDENS", "HI", "PSFSIZE_R", "PSFSIZE_Z", "GALDEPTH_Z", "PSFDEPTH_W1"]},
    "QSO": {zb: ["STARDENS", "HI", "EBV_DIFF_GR", "EBV_DIFF_RZ", "PSFDEPTH_G", "PSFDEPTH_R", "PSFDEPTH_Z",
                 "PSFDEPTH_W1", "PSFDEPTH_W2", "PSFSIZE_G", "PSFSIZE_R", "PSFSIZE_Z"]
            for zb in [(0.8, 1.3), (1.3, 2.1), (2.1, 3.1)]},
}
# photometric regions in which the regressions are done independently
DESI_REGIONS = {"LRG": ["North", "South_NGC", "SGC"], "QSO": ["North", "South", "DES"]}


def desi_region_masks(tracer):
    """Boolean NESTED nside-256 masks of DESI's regression regions (regressis DR9 footprint flags; ISSOUTH contains DES)."""
    import fitsio
    f = fitsio.read(FOOTPRINT)
    N, S, D, ngc, sgc = (np.asarray(f[c], bool) for c in ("ISNORTH", "ISSOUTH", "ISDES", "ISNGC", "ISSGC"))
    if tracer == "LRG":
        return {"North": N, "South_NGC": S & ngc & ~D, "SGC": (S | D) & sgc}
    return {"North": N, "South": S & ~D, "DES": D}


def _ebv_diff():
    """E(B-V)_SFD - E(B-V)_DESI-stars from g-r and r-z (Zhou et al. DESI reddening maps), NESTED nside 256."""
    import fitsio, healpy as hp
    out = {}
    for band, col in (("gr", "EBV_GR"), ("rz", "EBV_RZ")):
        d = fitsio.read(os.path.join(DUSTDIR, f"desi_dust_{band}_256.fits"), columns=["HPXPIXEL", col, "EBV_SFD"])
        m = np.full(hp.nside2npix(NSIDE), np.nan)
        m[hp.ring2nest(NSIDE, d["HPXPIXEL"])] = d["EBV_SFD"] - d[col]
        out[f"EBV_DIFF_{band.upper()}"] = m
    return out


def desi_feature_maps(tracer, coverage_nest=None, clip=5.0):
    """Standardised feature maps for DESI's scheme: {(zbin, region): (names, (n_feat, npix) array, region mask)}.
    Raw feature values (regressis convention: no magnitude conversion), standardised per region (coverage-weighted),
    clipped at 5 sigma; North uses the N property maps, the other regions the S maps."""
    names_all = sorted({n for v in DESI_FEATURES[tracer].values() for n in v})
    hp_names = [n for n in names_all if not n.startswith("EBV_DIFF")]
    import fitsio
    raw = {r: fitsio.read(os.path.join(HPDIR, f"{tracer}_mapprops_healpix_nested_nside256_{r}.fits"), columns=hp_names) for r in ("N", "S")}
    ebvd = _ebv_diff() if any(n.startswith("EBV_DIFF") for n in names_all) else {}
    cov = np.ones(len(raw["N"])) if coverage_nest is None else np.asarray(coverage_nest)
    regions = desi_region_masks(tracer)
    out = {}
    for zb, names in DESI_FEATURES[tracer].items():
        for reg, m in regions.items():
            m = m & (cov > 0)
            src = raw["N"] if reg == "North" else raw["S"]
            T = np.zeros((len(names), len(cov)))
            w = cov[m]
            for k, n in enumerate(names):
                x = (ebvd[n] if n in ebvd else np.asarray(src[n], float))[m].copy()
                bad = ~np.isfinite(x) | ((x == 0) if n not in ebvd else False)
                if bad.all():
                    continue
                mu = np.average(x[~bad], weights=w[~bad]); sd = np.sqrt(np.average((x[~bad] - mu) ** 2, weights=w[~bad]))
                y = np.clip((x - mu) / sd, -clip, clip); y[bad] = 0.0
                T[k, m] = y
            out[(zb, reg)] = (names, T, m)
    return out


def grid_template_maps_desi(fm, tracers):
    """Per-shell, per-region, per-feature template maps on the grid (DESI scheme). The shells must coincide with the
    DESI redshift bins. Returns (labels [(tracer, 'z..:region:feature')], maps (K, res^3))."""
    pix = grid_directions(fm); cov = coverage_nest256()
    labels, maps = [], []
    for t in tracers:
        fmap = desi_feature_maps(t, cov)
        for i, s in enumerate(fm.shells):
            if s.tracer != t:
                continue
            zb = (s.zlo, s.zhi)
            if zb not in DESI_FEATURES[t]:
                raise ValueError(f"shell {zb} is not a DESI {t} redshift bin")
            inshell = np.asarray(fm.nbar_cell[i]).ravel() > 0
            for reg in DESI_REGIONS[t]:
                names, T, m = fmap[(zb, reg)]
                inreg = m[pix]
                for k, n in enumerate(names):
                    v = T[k][pix] * inreg * inshell
                    if np.any(v):
                        labels.append((t, f"z{s.zlo:g}-{s.zhi:g}:{reg}:{n}")); maps.append(v)
    return labels, np.array(maps)


def wsys_contamination_desi(fm, tracer, voxel_avg=True):
    """DR1 imaging modulation per DESI redshift bin: 1/<WEIGHT_SYS> per nside-256 pixel computed from the objects of
    each bin (WEIGHT_SYS is a per-pixel, per-bin regression output), applied to the voxels of the matching shell.
    Pixels without objects in a bin use the nside-64 parent average. Normalised to unit mean over each shell."""
    import fitsio, healpy as hp
    d = os.environ.get("DESI_DR1_DIR", "/work/hdd/bdne/yzhang116/desi_dr1")
    cat = [fitsio.read(os.path.join(d, f"{tracer}_{c}_clustering.dat.fits"), columns=["RA", "DEC", "Z", "WEIGHT_SYS"]) for c in ("NGC", "SGC")]
    cat = np.concatenate(cat)
    npix = hp.nside2npix(NSIDE); par = np.arange(npix) // 16
    pix_grid = grid_directions(fm)
    g = np.ones(fm.res ** 3)
    for i, s in enumerate(fm.shells):
        if s.tracer != tracer:
            continue
        sel = (cat["Z"] >= s.zlo) & (cat["Z"] < s.zhi)
        p = hp.ang2pix(NSIDE, cat["RA"][sel], cat["DEC"][sel], lonlat=True, nest=True)
        sw = np.bincount(p, weights=cat["WEIGHT_SYS"][sel], minlength=npix); cnt = np.bincount(p, minlength=npix)
        sw64 = np.bincount(par, weights=sw, minlength=npix // 16); c64 = np.bincount(par, weights=cnt, minlength=npix // 16)
        inshell = np.asarray(fm.nbar_cell[i]).ravel() > 0
        if voxel_avg:
            # WEIGHT_SYS is a per-pixel, per-bin regression output: average 1/w over the voxel's solid angle, using only
            # pixels that contain objects of the bin (others carry no information)
            inv = np.where(cnt > 0, cnt / np.where(sw > 0, sw, 1.0), np.nan)
            f = voxel_average(fm, inv)
            f = np.where(np.isfinite(f), f, 1.0)
        else:
            w = np.where(cnt > 0, sw / np.maximum(cnt, 1), np.where(c64[par] > 0, sw64[par] / np.maximum(c64[par], 1), 1.0))
            f = 1.0 / w[pix_grid]
        g[inshell] = f[inshell] / f[inshell].mean()
    return g.reshape((fm.res,) * 3)


def data_space_from_grid(fm, tracers, idx, maps, labels):
    """Data-space template vectors (FieldFisher ordering) from grid template maps: T_k(x) = <N_t>(x) tau_k(x)."""
    offs, n_data = {}, 0
    for t in tracers:
        offs[t] = n_data; n_data += len(idx[t])
    mean = {t: sum(np.asarray(fm.nbar_cell[i]).ravel() for i, s in enumerate(fm.shells) if s.tracer == t) for t in tracers}
    T = np.zeros((len(labels), n_data))
    for k, (t, _) in enumerate(labels):
        ids = np.asarray(idx[t]); T[k, offs[t]:offs[t] + len(ids)] = (mean[t] * np.asarray(maps[k]).ravel())[ids]
    return T


def grid_template_compact(fm, tracers, intercepts=False, voxel_avg=True):
    """Memory-light version of grid_template_maps_desi (each voxel belongs to at most one shell x region group):
    returns labels [(tracer, 'z..:region:feature')] and, per tracer, dict(F=(n_feat, res^3) float32 standardised
    feature values of the voxel's own region, G=(res^3,) int32 group id (0 = no template), g=(K_t,) group of each
    amplitude, f=(K_t,) feature column of each amplitude, k=(K_t,) amplitude indices into labels).
    intercepts: add a constant (INTERCEPT, value 1) per shell x region group -- the region renormalisation."""
    pix = grid_directions(fm); cov = coverage_nest256()
    labels, comp = [], {}
    for t in tracers:
        fmap = desi_feature_maps(t, cov)
        feats = sorted({n for v in DESI_FEATURES[t].values() for n in v}) + (["INTERCEPT"] if intercepts else [])
        F = np.zeros((len(feats), fm.res ** 3), np.float32); G = np.zeros(fm.res ** 3, np.int32)
        gl, fl, kl, ng = [], [], [], 0
        for i, s in enumerate(fm.shells):
            if s.tracer != t:
                continue
            zb = (s.zlo, s.zhi)
            inshell = np.asarray(fm.nbar_cell[i]).ravel() > 0
            for reg in DESI_REGIONS[t]:
                names, T, m = fmap[(zb, reg)]
                sel = inshell & m[pix]
                if not sel.any():
                    continue
                ng += 1; G[sel] = ng
                for k, n in enumerate(names + (["INTERCEPT"] if intercepts else [])):
                    fi = feats.index(n)
                    if n == "INTERCEPT":
                        F[fi, sel] = 1.0
                    else:
                        tk = np.where(m, T[k], np.nan)                    # region-standardised, NaN outside the region
                        F[fi, sel] = np.nan_to_num(voxel_average(fm, tk)[sel]) if voxel_avg else T[k][pix][sel]
                    labels.append((t, f"z{s.zlo:g}-{s.zhi:g}:{reg}:{n}")); gl.append(ng); fl.append(fi); kl.append(len(labels) - 1)
        comp[t] = dict(F=F, G=G, g=np.array(gl), f=np.array(fl), k=np.array(kl), n_groups=ng, feats=feats)
    return labels, comp


def dense_from_compact(fm, labels, comp):
    """Dense (K, res^3) maps from the compact form (for the Fisher data-space vectors)."""
    maps = np.zeros((len(labels), fm.res ** 3))
    for t, c in comp.items():
        for g, f, k in zip(c["g"], c["f"], c["k"]):
            sel = c["G"] == g
            maps[k, sel] = c["F"][f, sel]
    return maps


# ---- voxel-averaged angular maps ---------------------------------------------------------------------------------
_SUBPIX = {}


def voxel_subpixels(fm, nsub=None):
    """NESTED nside-256 pixel of nsub^3 sub-points per voxel, (res^3, nsub^3). A voxel's counts integrate the
    angular selection over its solid angle, so angular maps must be averaged over the voxel, not sampled at its
    centre (a 125 Mpc/h voxel spans ~2 deg at chi = 3500 Mpc/h, i.e. ~60 nside-256 pixels)."""
    import healpy as hp
    nsub = nsub or (4 if fm.res <= 96 else 3)
    key = (id(fm), nsub)
    if key in _SUBPIX:
        return _SUBPIX[key]
    h = fm.L / fm.res
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    x = (np.arange(fm.res) + 0.5) * h
    out = np.empty((fm.res ** 3, nsub ** 3), np.int64)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    c = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1) - np.asarray(fm.observer)
    j = 0
    for dx in off:
        for dy in off:
            for dz in off:
                r = c + h * np.array([dx, dy, dz])
                out[:, j] = hp.vec2pix(NSIDE, r[:, 0], r[:, 1], r[:, 2], nest=True); j += 1
    _SUBPIX[key] = out
    return out


def voxel_average(fm, pixmap, nsub=None):
    """Average of a NESTED nside-256 map over each voxel's sub-points, ignoring NaN; NaN where all are NaN."""
    sp = voxel_subpixels(fm, nsub)
    v = pixmap[sp]
    good = np.isfinite(v)
    n = good.sum(1)
    return np.where(n > 0, np.where(good, v, 0).sum(1) / np.maximum(n, 1), np.nan)
