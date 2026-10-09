"""Tier 2 (tree-level bispectrum) convergence with binning: linear Dk=0.002 vs logarithmic bins.
Reports Tier 1, Tier 1+2 and Tier-2-only sigma(f_NL), QSO+LRG, bphi fixed, 80^3 and 160^3 k_max."""
from _common import *
import numpy as np, time
from stage1_config import build
from forward_model import ParamLayout
from fisher_closed import ShellGeometry, Tier1, Tier2, sigma
cosmo, geoms, shells, _ = build(); sg = ShellGeometry(geoms); L = 1e4
lay = ParamLayout(shells)
lines, rows = [], {}
for res in (80, 160):
    kny = np.pi * res / L
    F1 = Tier1(cosmo, shells, lay, sg, L=L, res=res).fisher()
    for label, kw in [("linear Dk=0.002", dict(dk=0.002)), ("log 20 bins", dict(log_bins=20)), ("log 40 bins", dict(log_bins=40)), ("log 60 bins", dict(log_bins=60))]:
        t0 = time.time(); t2 = Tier2(cosmo, shells, lay, sg, kmax=kny, L=L, **kw); F2 = t2.fisher()
        s2 = sigma(F2) if F2[0, 0] > 0 else float("nan")
        rows[(res, label)] = (sigma(F1), sigma(F1 + F2), s2, len(t2.k1))
        lines.append(f"- ${res}^3$ ($k_{{\\max}}={kny:.3f}$), {label}: {len(t2.k1)} triangles; Tier 1 {sigma(F1):.2f}, Tier 1+2 {sigma(F1+F2):.2f}, Tier 2 only {s2:.1f}  [{time.time()-t0:.0f}s]")
        print(lines[-1], flush=True)
fig, ax = plt.subplots(figsize=(6, 3.4))
for res, m in ((80, "o"), (160, "s")):
    labs = [l for (r, l) in rows if r == res]; ax.plot(range(len(labs)), [rows[(res, l)][1] for l in labs], m + "-", label=f"{res}^3 Tier 1+2")
    ax.axhline(rows[(res, labs[0])][0], ls=":", color="k" if res == 80 else "gray", label=f"{res}^3 Tier 1")
ax.set_xticks(range(4)); ax.set_xticklabels(["lin 0.002", "log20", "log40", "log60"]); ax.set_ylabel(r"$\sigma(f_{NL})$"); ax.legend(fontsize=7)
ax.set_title("Tier 2 binning convergence, QSO+LRG")
save_fig(fig, "tier2_binning")
write_note("tier2_binning", "Tier 2 bispectrum Fisher: binning convergence", lines + ["", r"Compare with the linearised-2LPT exact-marginal field-level values 5.3 ($80^3$) and 4.0 ($160^3$)."])
