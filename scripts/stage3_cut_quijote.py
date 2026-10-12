"""Mass-cut a batch of Quijote halo catalogues to a DESI-like tracer and save compact .npz files.

    python scripts/stage3_cut_quijote.py --in_dir DIR --pattern 'halo_CV_{i}.dat' --ids 0-100 --z 0.5 \
        --b1 1.885 --tracer LRG --set CV --out_root /work/hdd/bdne/yzhang116/quijote_cuts [--mmin M] [--delete_raw]

The mass threshold is found once (bisection on the measured linear bias, averaged over the first --n_calib
catalogues) unless --mmin is given, then applied to every catalogue. A summary JSON (per-realisation b1, n, N)
is written next to the cuts. With --delete_raw the raw catalogue is removed after its cut is saved.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quijote import quijote_cosmology, read_catalogue, linear_bias, mass_threshold_for_bias, cut_and_save, L_QUIJOTE  # noqa


def parse_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True)
ap.add_argument("--pattern", default="halo_CV_{i}.dat")
ap.add_argument("--ids", default="0-9")
ap.add_argument("--z", type=float, required=True)
ap.add_argument("--b1", type=float, default=None, help="target linear bias (DESI b1(z) if omitted: --tracer)")
ap.add_argument("--tracer", default="LRG", choices=["LRG", "QSO"])
ap.add_argument("--set", default="CV")
ap.add_argument("--tag", default=None, help="output dir suffix (default: tracer)")
ap.add_argument("--out_root", default="/work/hdd/bdne/yzhang116/quijote_cuts")
ap.add_argument("--mmin", type=float, default=None)
ap.add_argument("--n_calib", type=int, default=3)
ap.add_argument("--delete_raw", action="store_true")
a = ap.parse_args()

B1 = {"LRG": (0.209, 1.415), "QSO": (0.237, 0.771)}
b1_target = a.b1 if a.b1 is not None else B1[a.tracer][0] * (1 + a.z) ** 2 + B1[a.tracer][1]
cosmo = quijote_cosmology()
ids = parse_ids(a.ids)
out_dir = os.path.join(a.out_root, f"{a.set}_z{a.z:g}_{a.tag or a.tracer}")
os.makedirs(out_dir, exist_ok=True)
summary_path = os.path.join(out_dir, "summary.json")
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"realisations": {}}

mmin = a.mmin or summary.get("mmin")
if mmin is None:
    ths = []
    for i in ids[: a.n_calib]:
        cat = read_catalogue(os.path.join(a.in_dir, a.pattern.format(i=i)))
        if len(cat["mass"]) == 0:
            continue
        ths.append(mass_threshold_for_bias(cat, cosmo, a.z, b1_target))
        print(f"calib {i}: M_min = {ths[-1]:.3e}", flush=True)
    mmin = float(np.exp(np.mean(np.log(ths))))
summary.update({"set": a.set, "z": a.z, "tracer": a.tracer, "b1_target": b1_target, "mmin": mmin, "L": L_QUIJOTE})
print(f"target b1 = {b1_target:.3f} at z = {a.z}: M_min = {mmin:.3e} Msun/h", flush=True)

for i in ids:
    out = os.path.join(out_dir, f"{i}.npz")
    if os.path.exists(out) and str(i) in summary["realisations"]:
        continue
    t = time.time()
    raw = os.path.join(a.in_dir, a.pattern.format(i=i))
    if not os.path.exists(raw) or os.path.getsize(raw) == 0:
        print(f"{i}: missing or empty, skipped", flush=True); continue
    cat = read_catalogue(raw)
    N = cut_and_save(cat, mmin, out, meta={"set": a.set, "z": a.z, "tracer": a.tracer, "realisation": i})
    sel = cat["mass"] > mmin
    b1, err, _ = linear_bias(cat["pos"][sel], cosmo, a.z)
    summary["realisations"][str(i)] = {"N": N, "n": N / L_QUIJOTE ** 3, "b1": float(b1), "b1_err": float(err)}
    json.dump(summary, open(summary_path, "w"), indent=1)
    if a.delete_raw:
        os.remove(raw)
    print(f"{i}: N = {N}, n = {N / L_QUIJOTE ** 3:.2e}, b1 = {b1:.3f} +- {err:.3f}  ({time.time() - t:.0f}s)", flush=True)

b = np.array([r["b1"] for r in summary["realisations"].values()])
print(f"done: {len(b)} realisations, b1 = {b.mean():.3f} +- {b.std():.3f}")
