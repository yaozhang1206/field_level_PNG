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
ap.add_argument("--sample_bphi", action="store_true"); ap.add_argument("--fnl_fid", type=float, default=0.0)
ap.add_argument("--resume", action="store_true", help="continue from per-probe partial results in the output json")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build(p_qso=a.p_qso)
lay = ParamLayout(shells, sample_bphi=a.sample_bphi)
prior = lay.prior(0.4)
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
tag = a.tag or f"marginal_res{a.res}_p{a.p_qso or 1.6}{'_linear' if a.linear else ''}{'_bphi_sampled_fnl%g' % a.fnl_fid if a.sample_bphi else ''}"
outfile = ROOT / "results" / f"fisher_field_{tag}.json"
out = json.load(open(outfile)) if (a.resume and outfile.exists()) else dict(args=vars(a), tag=tag, names=lay.names, realisations=[], mean={}, partial={})
out.setdefault("partial", {})
for trset in a.tracers.split(","):
    trs = [t for t in fm.tracers if t in trset]
    Fs, Fschur = [], []
    for r in range(a.nreal):
        eps = jax.random.normal(jax.random.PRNGKey(a.seed0 + r), (a.res,) * 3, dtype=jnp.float64)
        t0 = time.time()
        key = f"{trset}:{r}"
        done = [rr for rr in out["realisations"] if rr["tracers"] == trset and rr["real"] == r]
        part = out["partial"].get(key, {})
        if done:
            prev = done[0].get("F_probes") or part.get("probes", [])
            if len(prev) >= a.probes:
                Fs.append(np.array(done[0]["F_marginal"])); Fschur.append(np.array(done[0]["F_schur"])); continue
            # extend an existing realisation to more probes
            part = dict(part, F_schur=done[0]["F_schur"], probes=prev); out["partial"][key] = part
            out["realisations"] = [rr for rr in out["realisations"] if not (rr["tracers"] == trset and rr["real"] == r)]
            print(f"[{tag}] {trset} real {r}: extending from {len(prev)} to {a.probes} probes", flush=True)
        ff = FieldFisher(fm, lay, eps, lay.fiducial(a.fnl_fid), prior=prior, rsd=True, linear=a.linear, tracers=trs)
        if "F_schur" in part:
            Fs_ = np.array(part["F_schur"]); ff.F_marg = Fs_; ff.iters = [0.0]
        else:
            Fs_ = ff.fisher(); part["F_schur"] = Fs_.tolist(); out["partial"][key] = part
            json.dump(out, open(outfile, "w"), indent=1)
        Fschur.append(Fs_)
        Fs_prev = [np.array(x) for x in part.get("probes", [])]
        def cb(Fs_list, key=key):
            out["partial"][key]["probes"] = [x.tolist() for x in Fs_list]
            json.dump(out, open(outfile, "w"), indent=1)
        Fm = ff.fisher_marginal(n_probes=a.probes, seed=r, probe_callback=cb, start_probe=len(Fs_prev), Fs_prev=Fs_prev); Fs.append(Fm)
        nprobe = len(ff.F_marginal_probes)
        rec = dict(tracers=trset, real=r, time=time.time() - t0, sigma_marginal=sigma_of(Fm), sigma_schur=sigma_of(Fs_),
                   probe_std=float(ff.F_marginal_probe_std), n_probes=nprobe, sigma_err=float(ff.F_marginal_probe_std / np.sqrt(nprobe)),
                   F_marginal=Fm.tolist(), F_schur=Fs_.tolist(), F_probes=[x.tolist() for x in ff.F_marginal_probes])
        out["realisations"].append(rec)
        print(f"[{tag}] {trset} real {r}: {rec['time']:.0f}s  sigma marginal {rec['sigma_marginal']:.2f} (probe std {rec['probe_std']:.2f})  schur {rec['sigma_schur']:.2f}", flush=True)
        json.dump(out, open(outfile, "w"), indent=1)
    out["mean"][trset] = dict(sigma_marginal_meanF=sigma_of(np.mean(Fs, 0)), sigma_schur_meanF=sigma_of(np.mean(Fschur, 0)),
                              sigma_marginal_list=[sigma_of(F) for F in Fs], sigma_schur_list=[sigma_of(F) for F in Fschur])
    json.dump(out, open(outfile, "w"), indent=1)
    print(f"[{tag}] {trset} mean: {out['mean'][trset]}", flush=True)
