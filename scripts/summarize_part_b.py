"""Collect Part B results (results/fisher_field_*.json, results/fisher_closed.json, results/fisher_pk_full.json)
into test/notes/fisher_field.md, test/figures/fisher_field_summary.png and results/fisher_field.json."""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from _common import plt, save_fig, write_note

closed = json.load(open(ROOT / "results" / "fisher_closed.json"))["results"]
pk = json.load(open(ROOT / "results" / "fisher_pk_full.json"))["results"]
rows, summary = [], {}
hdr = ["| run | tracers | n_real | sigma(f_NL) Schur: mean +- scatter | sigma from mean F | unmarg. f_NL | closed Tier1 (k_f / 0.003) | Tier1+2 |", "|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(str(ROOT / "results" / "fisher_field_res*.json"))):
    d = json.load(open(f)); tag = d["tag"]; a = d["args"]
    for trset, m in d.get("mean", {}).items():
        recs = [r for r in d["realisations"] if r["tracers"] == trset]
        key = f"res{a['res']}_{trset}_{'bphi_sampled' if a['sample_bphi'] else 'bphi_fixed'}"
        c = closed.get(key, {})
        unm = np.mean([r["sigma_unmarg_fnl"] for r in recs])
        rows.append(f"| {tag} | {trset} | {len(recs)} | {m['sigma_mean']:.2f} +- {m['sigma_std']:.2f} | {m['sigma_meanF']:.2f} | {unm:.2f} | "
                    f"{c.get('sigma_tier1', float('nan')):.1f} / {c.get('sigma_tier1_kmin003', float('nan')):.1f} | {c.get('sigma_tier12', float('nan')):.1f} |")
        summary[f"{tag}:{trset}"] = dict(tag=tag, tracers=trset, res=a["res"], p_qso=a["p_qso"], sample_bphi=a["sample_bphi"],
                                         fnl_fid=a["fnl_fid"], n_real=len(recs), sigma_mean=m["sigma_mean"], sigma_std=m["sigma_std"],
                                         sigma_meanF=m["sigma_meanF"], sigma_unmarg=float(unm), closed=c)
# exact-marginal (Hutchinson) runs
rows += ["", "| exact-marginal run | tracers | n_real | sigma(f_NL) marginal (per realisation) | from mean F | Schur same runs | closed Tier1 exact (k_f / 0.003) |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(str(ROOT / "results" / "fisher_field_marginal_*.json"))):
    d = json.load(open(f)); tag = d["tag"]; a = d["args"]
    for trset, m in d.get("mean", {}).items():
        key = f"res{a['res']}_{trset}_bphi_fixed"; c = closed.get(key, {})
        rows.append(f"| {tag} | {trset} | {len(m['sigma_marginal_list'])} | {', '.join(f'{x:.2f}' for x in m['sigma_marginal_list'])} | {m['sigma_marginal_meanF']:.2f} | "
                    f"{', '.join(f'{x:.2f}' for x in m['sigma_schur_list'])} | {c.get('sigma_tier1', float('nan')):.1f} / {c.get('sigma_tier1_kmin003', float('nan')):.1f} |")
        summary[f"{tag}:{trset}"] = dict(tag=tag, tracers=trset, res=a["res"], p_qso=a["p_qso"], linear=a.get("linear", False),
                                         sigma_marginal=m["sigma_marginal_list"], sigma_marginal_meanF=m["sigma_marginal_meanF"],
                                         sigma_schur=m["sigma_schur_list"], closed=c)
# figure: sigma by configuration
fig, ax = plt.subplots(figsize=(8, 3.8))
labels, vals, errs, cl1, cl2 = [], [], [], [], []
marg_x, marg_y = [], []
for k, v in summary.items():
    if "sigma_marginal" in v:
        labels.append(f"{v['res']}^3 {v['tracers']}\np={v['p_qso'] or 1.6} exact-marg.")
        vals.append(np.nan); errs.append(0); marg_x.append(len(labels) - 1); marg_y.append(v["sigma_marginal_meanF"])
    else:
        labels.append(f"{v['res']}^3 {v['tracers']}\np={v['p_qso'] or 1.6} {'bphi~40%' if v['sample_bphi'] else 'bphi fixed'} Schur")
        vals.append(v["sigma_mean"]); errs.append(v["sigma_std"])
    cl1.append(v["closed"].get("sigma_tier1_kmin003", np.nan)); cl2.append(v["closed"].get("sigma_tier1", np.nan))
x = np.arange(len(labels))
ax.errorbar(x, vals, yerr=errs, fmt="o", label="Schur (profile), mean +- realisation scatter")
ax.plot(marg_x, marg_y, "D", color="C3", label="exact Gaussian-marginal (Hutchinson), linearised 2LPT model")
ax.plot(x, cl1, "s", mfc="none", label="closed Tier 1, k_min = 0.003")
ax.plot(x, cl2, "^", mfc="none", label="closed Tier 1, k_min = k_f")
ax.axhline(5, color="g", ls=":", label="go/no-go 5"); ax.axhline(pk["QSO+LRG_shells"]["sigma_fnl"], color="r", ls=":", label=f"Part A P(k) shells {pk['QSO+LRG_shells']['sigma_fnl']:.1f}")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylabel(r"$\sigma(f_{NL})$"); ax.legend(fontsize=7); ax.set_ylim(0, None)
ax.set_title("Stage 1 Part B: field-level Fisher forecasts, DESI DR1")
save_fig(fig, "fisher_field_summary")
json.dump(summary, open(ROOT / "results" / "fisher_field.json", "w"), indent=1)
note = ["Field-level Fisher forecasts (DISCO-DJ 2LPT + PNG + Lagrangian bias + RSD, DR1 footprint, Gaussian noise).",
        "Two estimators: Schur complement of the joint (eps, theta) Fisher (profile curvature; biased by (P+N)/(2P) per mode, see",
        "schur_dense16.md) and the exact Gaussian-marginal Fisher of the model linearised in eps (Hutchinson trace estimator).",
        "Marginalised over per-shell (b1, b2, bK) [+ bphi, bphidelta with 40% prior when sampled]; fiducial f_NL = 0 (5 when bphi sampled).",
        "Closed forms from fisher_closed.md (same shells/volumes). 'unmarg.' = 1/sqrt(F_ff) with all bias fixed.", ""] + hdr + rows + [
        "", "Realisation scatter reflects the few largest-scale modes that dominate the PNG signal; use the mean-F value.",
        "Figure: fisher_field_summary.png"]
write_note("fisher_field", "Part B: field-level Fisher results (Schur) vs closed forms", note)
print("\n".join(note))
