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
ap.add_argument("--per_shell", action="store_true", help="also: imaging amplitudes free per redshift shell")
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
T9, lab9 = data_space_templates(fm, trs, idx, imaging="imaging9")
TR, labR = data_space_templates(fm, trs, idx, radial_order=a.radial)
blocks, labels = [T9, TR], lab9 + labR
if a.per_shell:
    TS, labS = data_space_templates(fm, trs, idx, imaging="imaging9", per_shell=True)
    blocks.append(TS); labels += labS
T = np.concatenate([b for b in blocks if len(b)], 0)
print(f"[syst] {len(T)} template vectors ({time.time()-t0:.0f}s): {len(lab9)} imaging, {len(labR)} radial"
      + (f", {len(labS)} per-shell imaging" if a.per_shell else ""), flush=True)
rows = lambda pred: [i for i, l in enumerate(labels) if pred(l)]
is_img = lambda l: l.split(":")[1] == "all"
img3 = set(SETS["imaging3"])
configs = {
    "none": [],
    "imaging3": rows(lambda l: is_img(l) and l.split(":")[-1] in img3),
    "imaging9": rows(is_img),
    "radial_P0": rows(lambda l: l.endswith(":P0")),
    f"radial_P{a.radial}": rows(lambda l: ":P" in l),
    f"imaging9+radial_P{a.radial}": rows(lambda l: is_img(l) or ":P" in l),
}
for t in trs:
    configs[f"imaging9_{t}only"] = rows(lambda l, t=t: is_img(l) and l.startswith(t))
if a.per_shell:
    configs["imaging9_per_shell"] = rows(lambda l: l.split(":")[1].isdigit())
    configs[f"imaging9_per_shell+radial_P{a.radial}"] = configs["imaging9_per_shell"] + rows(lambda l: ":P" in l)
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
