"""Summarise Stage 3 halo-ensemble results (results/stage3/*.json) into test/notes/stage3_ensembles.md + figure."""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))
from _common import save_fig, write_note, plt
TRUTH = {"LC_p": 100.0, "LC_m": -100.0}
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "stage3", "*_p1m4d_*.json"))):
    R = json.load(open(f)); runs = R["runs"]
    if len(runs) < 2:
        continue
    name = os.path.basename(f)[:-5]
    truth = next((v for k, v in TRUTH.items() if name.startswith(k)), 0.0)
    m = np.array([r["summary"]["f_NL"]["mean"] for r in runs.values()]); s = np.array([r["summary"]["f_NL"]["std"] for r in runs.values()])
    b1 = np.mean([r["summary"][[k for k in r["summary"] if k.startswith("b1")][0]]["mean"] for r in runs.values()])
    chi = [r["chi2_per_dof"] for r in runs.values() if "chi2_per_dof" in r]
    rows.append(dict(name=name, truth=truth, n=len(m), mean=m.mean(), se=m.std(ddof=1) / np.sqrt(len(m)), scatter=m.std(ddof=1),
                     sig=s.mean(), b1=b1, pull=np.mean((m - truth) / s), pull_sd=np.std((m - truth) / s, ddof=1),
                     chi2=np.mean(chi) if chi else np.nan, m=m, s=s))
lines = ["Quijote halo ensembles, DR1 forward model (2LPT, Lagrangian bias + $b_\\nabla$, Kaiser RSD, $b_\\phi$ from UMF with the DR1 $p$),",
         "$16^3$ grid (62.5 Mpc/h), 4× CIC-deconvolved Fourier-truncated assignment for data and model, Poisson-level Gaussian noise, cold start.",
         "Pull = (posterior mean − truth)/posterior σ per box; a calibrated posterior has pull mean 0 and s.d. 1.", "",
         "| run | N | truth | $\\langle f_{\\rm NL}\\rangle$ | scatter | $\\langle\\sigma\\rangle$ | pull mean | pull s.d. | $\\langle b_1\\rangle$ | $\\chi^2$/dof |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['name']} | {r['n']} | {r['truth']:+.0f} | {r['mean']:+.1f} ± {r['se']:.1f} | {r['scatter']:.0f} | {r['sig']:.0f} | "
                 f"{r['pull']:+.2f} | {r['pull_sd']:.2f} | {r['b1']:.2f} | {r['chi2']:.3f} |")
z = [r for r in rows if r["truth"] == 0 and "mock" not in r["name"] and "bphi" not in r["name"] and "lpt1" not in r["name"] and "res16" in r["name"]]
if z:
    w = np.array([1 / r["se"] ** 2 for r in z]); mu = np.sum(w * np.array([r["mean"] for r in z])) / w.sum()
    lines += ["", f"Pooled $f_{{\\rm NL}} = 0$ halo samples (2LPT, $16^3$): $\\langle f_{{\\rm NL}}\\rangle = {mu:+.1f} \\pm {1/np.sqrt(w.sum()):.1f}$."]
fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
for j, r in enumerate(rows):
    ax.errorbar(r["mean"] - r["truth"], j, xerr=r["se"], fmt="o", color="C3" if "mock" in r["name"] else "C0")
ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"][:48] for r in rows], fontsize=7)
ax.set_xlabel("$\\langle f_{\\rm NL}\\rangle -$ truth (ensemble mean ± error)"); plt.tight_layout()
save_fig(fig, "stage3_ensembles")
write_note("stage3_ensembles", "Stage 3: f_NL recovery on Quijote halo ensembles (model-mismatch test)", lines)
print("\n".join(lines))
