"""Exact Gaussian-marginal Fisher of the linearised field-level model (Hutchinson), DR1 footprint.
Usage: python scripts/run_fisher_marginal.py --res 80 --nreal 2 --probes 8 [--linear] [--tracers QSO+LRG] [--p_qso 1.0]"""
import sys, os, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "test"))
import numpy as np, jax, jax.numpy as jnp
from stage1_config import build
from geometry import dr1_angular_mask
from forward_model import ForwardModel, ParamLayout
from fisher_field import FieldFisher, sigma_of

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=80); ap.add_argument("--nreal", type=int, default=2)
ap.add_argument("--probes", type=int, default=8); ap.add_argument("--linear", action="store_true")
ap.add_argument("--p_qso", type=float, default=None); ap.add_argument("--tracers", default="QSO+LRG")
ap.add_argument("--seed0", type=int, default=1000); ap.add_argument("--tag", default="")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build(p_qso=a.p_qso)
lay = ParamLayout(shells)
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
tag = a.tag or f"marginal_res{a.res}_p{a.p_qso or 1.6}{'_linear' if a.linear else ''}"
out = dict(args=vars(a), tag=tag, names=lay.names, realisations=[], mean={})
for trset in a.tracers.split(","):
    trs = [t for t in fm.tracers if t in trset]
    Fs, Fschur = [], []
    for r in range(a.nreal):
        eps = jax.random.normal(jax.random.PRNGKey(a.seed0 + r), (a.res,) * 3, dtype=jnp.float64)
        t0 = time.time()
        ff = FieldFisher(fm, lay, eps, lay.fiducial(), rsd=True, linear=a.linear, tracers=trs)
        Fs_ = ff.fisher(); Fschur.append(Fs_)
        Fm = ff.fisher_marginal(n_probes=a.probes, seed=r); Fs.append(Fm)
        rec = dict(tracers=trset, real=r, time=time.time() - t0, sigma_marginal=sigma_of(Fm), sigma_schur=sigma_of(Fs_),
                   probe_std=float(ff.F_marginal_probe_std), F_marginal=Fm.tolist(), F_schur=Fs_.tolist())
        out["realisations"].append(rec)
        print(f"[{tag}] {trset} real {r}: {rec['time']:.0f}s  sigma marginal {rec['sigma_marginal']:.2f} (probe std {rec['probe_std']:.2f})  schur {rec['sigma_schur']:.2f}", flush=True)
        json.dump(out, open(ROOT / "results" / f"fisher_field_{tag}.json", "w"), indent=1)
    out["mean"][trset] = dict(sigma_marginal_meanF=sigma_of(np.mean(Fs, 0)), sigma_schur_meanF=sigma_of(np.mean(Fschur, 0)),
                              sigma_marginal_list=[sigma_of(F) for F in Fs], sigma_schur_list=[sigma_of(F) for F in Fschur])
    json.dump(out, open(ROOT / "results" / f"fisher_field_{tag}.json", "w"), indent=1)
    print(f"[{tag}] {trset} mean: {out['mean'][trset]}", flush=True)
