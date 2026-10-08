"""Stage 1 Part B production: Schur-complement field-level Fisher with the DISCO-DJ 2LPT forward model.
Usage: python scripts/run_fisher_field.py --res 80 --nreal 5 [--p_qso 1.0] [--sample_bphi --fnl_fid 5] [--tag name]"""
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
ap.add_argument("--res", type=int, default=80); ap.add_argument("--nreal", type=int, default=3)
ap.add_argument("--p_qso", type=float, default=None); ap.add_argument("--sample_bphi", action="store_true")
ap.add_argument("--fnl_fid", type=float, default=0.0); ap.add_argument("--no_rsd", action="store_true")
ap.add_argument("--tag", default=""); ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--tracers", default="QSO+LRG", help="comma list of tracer sets, e.g. QSO,LRG,QSO+LRG")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build(p_qso=a.p_qso)
lay = ParamLayout(shells, sample_bphi=a.sample_bphi)
prior = lay.prior(0.4)
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
tag = a.tag or f"res{a.res}_p{a.p_qso or 1.6}_{'bphi_sampled' if a.sample_bphi else 'bphi_fixed'}_fnl{a.fnl_fid:g}{'_norsd' if a.no_rsd else ''}"
out = dict(args=vars(a), tag=tag, names=lay.names, footprint={t: int(fm.footprint[t].sum()) for t in fm.tracers},
           shells=[vars(s) for s in shells], realisations=[])
print(f"[{tag}] voxels: {out['footprint']}, n_theta={lay.n}", flush=True)
for trset in a.tracers.split(","):
    trs = [t for t in fm.tracers if t in trset]
    Fs = []
    for r in range(a.nreal):
        eps = jax.random.normal(jax.random.PRNGKey(a.seed0 + r), (a.res,) * 3, dtype=jnp.float64)
        t0 = time.time()
        ff = FieldFisher(fm, lay, eps, lay.fiducial(a.fnl_fid), prior=prior, rsd=not a.no_rsd, tracers=trs)
        F = ff.fisher()
        Fs.append(F)
        rec = dict(tracers=trset, real=r, time=time.time() - t0, cg_resid_max=float(max(ff.iters)), sigma=sigma_of(F),
                   sigma_unmarg_fnl=float(1 / np.sqrt(F[0, 0])), F=F.tolist(), F_tt=ff.F_tt.tolist())
        out["realisations"].append(rec)
        print(f"[{tag}] {trset} real {r}: {rec['time']:.0f}s, CG resid {rec['cg_resid_max']:.1e}, sigma {rec['sigma']:.2f}", flush=True)
        json.dump(out, open(ROOT / "results" / f"fisher_field_{tag}.json", "w"), indent=1)
    Fm = np.mean(Fs, axis=0)
    out.setdefault("mean", {})[trset] = dict(sigma_meanF=sigma_of(Fm), sigma_mean=float(np.mean([sigma_of(F) for F in Fs])),
                                            sigma_std=float(np.std([sigma_of(F) for F in Fs])))
    json.dump(out, open(ROOT / "results" / f"fisher_field_{tag}.json", "w"), indent=1)
    print(f"[{tag}] {trset} mean: {out['mean'][trset]}", flush=True)
