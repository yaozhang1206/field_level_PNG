"""Stage 4 forecast: sigma(f_NL) of the DR1 field-level model with imaging templates and radial n(z) modes
marginalised (flat priors = deprojection), exact Gaussian-marginal Fisher (Hutchinson), several template sets at once.

    python scripts/run_fisher_syst.py --res 80 [--linear] [--probes 8] [--nreal 1] [--per_shell] [--radial 2]
        -> results/fisher_syst_res<res>[_linear].json
"""
import sys, os, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, jax, jax.numpy as jnp
from stage1_config import build
from geometry import dr1_angular_mask
from forward_model import ForwardModel, ParamLayout
from fisher_field import FieldFisher, sigma_of
from templates import data_space_templates, SETS

ap = argparse.ArgumentParser()
ap.add_argument("--res", type=int, default=80); ap.add_argument("--nreal", type=int, default=1)
ap.add_argument("--probes", type=int, default=8); ap.add_argument("--linear", action="store_true")
ap.add_argument("--radial", type=int, default=2, help="Legendre order of the per-shell radial modes")
ap.add_argument("--sample_bphi", action="store_true"); ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--tag", default="")
a = ap.parse_args()
L = 10000.0
cosmo, geoms, shells, nz = build()
lay = ParamLayout(shells, sample_bphi=a.sample_bphi)
prior = lay.prior(0.4)
fm = ForwardModel(cosmo, shells, L=L, res=a.res, nz_funcs=nz, ang_mask=dr1_angular_mask(64))
tag = a.tag or f"res{a.res}{'_linear' if a.linear else ''}{'_bphi' if a.sample_bphi else ''}"
outfile = ROOT / "results" / f"fisher_syst_{tag}.json"
out = json.load(open(outfile)) if outfile.exists() else dict(args=vars(a), names=lay.names, realisations={})

trs = fm.tracers
idx = {t: np.flatnonzero(fm.footprint[t].ravel()) for t in trs}
t0 = time.time()
from templates import grid_template_compact, dense_from_compact, data_space_from_grid
dlab, dcomp = grid_template_compact(fm, trs)
dmaps = dense_from_compact(fm, dlab, dcomp)
TD = data_space_from_grid(fm, trs, idx, dmaps, dlab)
TR, labR = data_space_templates(fm, trs, idx, radial_order=a.radial)
labels = [f"{t}:{l}" for t, l in dlab] + labR
T = np.concatenate([TD, TR], 0)
print(f"[syst] {len(T)} template vectors ({time.time()-t0:.0f}s): {len(dlab)} DESI imaging, {len(labR)} radial", flush=True)
nD = len(dlab)
rows = lambda pred: [i for i, l in enumerate(labels) if pred(i, l)]
configs = {
    "none": [],
    "desi": rows(lambda i, l: i < nD),
    "radial_P0": rows(lambda i, l: l.endswith(":P0")),
    f"radial_P{a.radial}": rows(lambda i, l: i >= nD),
    "desi+radial_P0": rows(lambda i, l: i < nD or l.endswith(":P0")),
    f"desi+radial_P{a.radial}": list(range(len(labels))),
}
for t in trs:
    configs[f"desi_{t}only"] = rows(lambda i, l, t=t: i < nD and l.startswith(t))
print("[syst] configs: " + ", ".join(f"{k} ({len(v)})" for k, v in configs.items()), flush=True)
out["configs"] = {k: len(v) for k, v in configs.items()}

for r in range(a.nreal):
    key = str(r)
    rec = out["realisations"].get(key, {})
    eps = jax.random.normal(jax.random.PRNGKey(a.seed0 + r), (a.res,) * 3, dtype=jnp.float64)
    ff = FieldFisher(fm, lay, eps, lay.fiducial(0.0), prior=prior, rsd=True, linear=a.linear, tracers=trs)
    if "dF" not in rec:
        dF = ff.deproj_loss(T, configs, checkpoint=str(outfile).replace(".json", f"_r{r}_ckpt.npz"))
        rec["dF"] = {c: v.tolist() for c, v in dF.items()}
        out["realisations"][key] = rec; json.dump(out, open(outfile, "w"))
    prev = [np.array(x) for x in rec.get("probes", [])]
    if len(prev) < a.probes:
        def cb(Fs, key=key):
            out["realisations"][key]["probes"] = [x.tolist() for x in Fs]
            json.dump(out, open(outfile, "w"))
        ff.fisher_marginal(n_probes=a.probes, seed=r, probe_callback=cb, start_probe=len(prev), Fs_prev=prev)
    rec = out["realisations"][key]
    F0 = np.mean(rec["probes"], 0) + prior
    rec["sigma"] = {c: sigma_of(F0 - np.array(rec["dF"][c])) for c in configs}
    json.dump(out, open(outfile, "w"))
    print(f"[syst] real {r}: " + "  ".join(f"{c} {v:.2f}" for c, v in rec["sigma"].items()) + f"  [{time.time()-t0:.0f}s]", flush=True)

summ = {}
F0 = np.mean([np.mean(v["probes"], 0) for v in out["realisations"].values()], 0) + prior
for c in configs:
    dF = np.mean([np.array(v["dF"][c]) for v in out["realisations"].values()], 0)
    summ[c] = dict(sigma=sigma_of(F0 - dF), dF_fnl_frac=float(dF[0, 0] / F0[0, 0]))
base = summ["none"]["sigma"]
for c in summ:
    summ[c]["ratio"] = summ[c]["sigma"] / base
out["summary"] = summ
json.dump(out, open(outfile, "w"), indent=1)
print("[syst] summary: " + "  ".join(f"{c} {v['sigma']:.2f} (x{v['ratio']:.3f})" for c, v in summ.items()), flush=True)
