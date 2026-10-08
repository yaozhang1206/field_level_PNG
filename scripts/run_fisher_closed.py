"""Stage 1 Part B closed forms: Tier 1 (Gaussian mode sum) + Tier 2 (tree-level bispectrum)."""
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from _common import plt, save_fig, write_note
from stage1_config import build
from forward_model import ParamLayout
from fisher_closed import ShellGeometry, Tier1, Tier2, sigma

t0 = time.time()
cosmo, geoms, shells, _ = build()
sg = ShellGeometry(geoms)
L = 10000.0
res_list = [80, 160]
out, lines = {}, []
hdr = r"| grid ($k_{\rm Ny}$) | tracers | $b_\phi$ | Tier 1 ($P$ only) | Tier 1 + Tier 2 ($P+B$) | Tier 1, $k_{\min}=0.003$ | Tier 2 only |"
lines += [hdr, "|---|---|---|---|---|---|---|"]
def tracer_sets(tr):
    return [s for s in shells if s.tracer in tr]

for res in res_list:
    kny = np.pi * res / L
    for sample in (False, True):
        for tr in ("QSO", "LRG", "QSO+LRG"):
            sh = tracer_sets(tr)
            lay = ParamLayout(sh, sample_bphi=sample)
            prior = lay.prior(0.4)
            F1 = Tier1(cosmo, sh, lay, sg, L=L, res=res).fisher()
            F1b = Tier1(cosmo, sh, lay, sg, L=L, res=res, kmin=0.003).fisher()
            F2 = Tier2(cosmo, sh, lay, sg, kmax=kny, L=L).fisher()
            s1, s12, s1b = sigma(F1, 0, prior), sigma(F1 + F2, 0, prior), sigma(F1b, 0, prior)
            try:
                s2 = sigma(F2, 0, prior)
            except np.linalg.LinAlgError:
                s2 = float("nan")
            key = f"res{res}_{tr}_{'bphi_sampled' if sample else 'bphi_fixed'}"
            out[key] = dict(res=res, kny=kny, tracers=tr, bphi_sampled=sample, sigma_tier1=s1, sigma_tier12=s12,
                            sigma_tier1_kmin003=s1b, sigma_tier2=s2)
            lines.append(f"| ${res}^3$ ({kny:.3f}) | {tr} | {'40% prior' if sample else 'fixed'} | {s1:.1f} | {s12:.1f} | {s1b:.1f} | {s2:.1f} |")
        print(lines[-3:], f"{time.time()-t0:.0f}s", flush=True)

# information density figure: Tier 1 dF/dlnk and Tier 2 cumulative sigma vs kmax (QSO+LRG, bphi fixed, 160^3)
lay = ParamLayout(shells, sample_bphi=False)
kmaxs = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1])
s_t1, s_t12 = [], []
for km in kmaxs:
    F1 = Tier1(cosmo, shells, lay, sg, L=L, res=160, sphere_kmax=km).fisher()
    F2 = Tier2(cosmo, shells, lay, sg, kmax=km, L=L, dk=0.0025).fisher()
    s_t1.append(sigma(F1)); s_t12.append(sigma(F1 + F2))
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(kmaxs, s_t1, "o-", label="Tier 1: P(k,mu) Gaussian mode sum")
ax.plot(kmaxs, s_t12, "s-", label="Tier 1 + Tier 2 (tree-level B)")
ax.axhline(5, color="g", ls=":", label="go/no-go: 5"); ax.axhline(9.2, color="r", ls=":", label="DESI DR1 P(k) OQE: 9.2")
ax.set_xlabel(r"$k_{\max}$ [h/Mpc] (sphere)"); ax.set_ylabel(r"$\sigma(f_{NL})$, QSO+LRG, bias marginalised"); ax.set_ylim(0, 14); ax.legend(fontsize=7)
ax.set_title("Closed-form field-level forecast (bphi fixed, UMF)")
save_fig(fig, "fisher_closed_kmax")
json.dump(dict(results=out, kmax_curve=dict(kmax=kmaxs.tolist(), tier1=s_t1, tier12=s_t12),
               shells=[vars(s) for s in shells]), open(ROOT / "results" / "fisher_closed.json", "w"), indent=1)
head = [r"Shells (tracer, $z$ range: $b_1$, $b_2$, $b_K$, $b_\phi$, $b_{\phi\delta}$): " + "; ".join(f"{s.tracer} {s.zlo}–{s.zhi}: {s.b1:.2f}, {s.b2:.2f}, {s.bK:.2f}, {s.bphi:.2f}, {s.bphidelta:.2f}" for s in shells),
        r"Volumes: full DR1 footprint (9070 deg$^2$), shot noise from observed density $\bar c\,{\rm NX}$; $k_{\min} = 2\pi/L$ unless stated.",
        r"Tier 1 = Gaussian mode sum over the grid cube (Kaiser RSD; parameters $f_{\rm NL}$, $b_1$ per shell [+$b_\phi$]):",
        r"$$F_{ij} = \sum_{\rm shells}\int dV(z)\int\frac{d^3k}{(2\pi)^3}\,\frac{\partial_i P_g\,\partial_j P_g}{2\,[P_g(k,\mu) + 1/n_{\rm obs}(z)]^2},\qquad P_g = [b_1 + b_\phi f_{\rm NL}/T(k,z) + f\mu^2]^2 P_{\rm lin}(k,z).$$",
        r"Tier 2 = tree-level real-space bispectrum, $k < k_{\rm Ny}$ sphere, $\Delta k=0.002$, Gaussian covariance; adds $b_2$, $b_K$, $b_{\phi\delta}$:",
        r"$$B_g = c_1c_2c_3\,B_{\rm PNG} + \sum_{\rm cyc} c_1 c_2\left[2b_1F_2 + b_2 + 2b_K S_2 + b_{\phi\delta} f_{\rm NL}\left(\tfrac{1}{T_1}+\tfrac{1}{T_2}\right)\right]P_1P_2,\quad c_i = b_1 + \frac{b_\phi f_{\rm NL}}{T_i},\quad {\rm Var}[B] = \frac{s_B (2\pi)^6\,\tilde P_1\tilde P_2\tilde P_3}{8\pi^2 V k_1k_2k_3\Delta k^3}.$$",
        r"Marginalised over all bias parameters; '$b_\phi$ sampled' = 40% Gaussian prior on $b_\phi$ and $b_{\phi\delta}$ per shell.",
        r"NOTE: at fiducial $f_{\rm NL} = 0$ the Fisher is blind to the $b_\phi$ prior ($b_\phi$ enters only as $b_\phi f_{\rm NL}$), so 'sampled' = 'fixed' here;",
        r"propagating $f_{\rm NL} = A/b_\phi$ with a 40% prior on $b_\phi$ inflates $\sigma$ by $\approx\sqrt{1+3\cdot0.4^2} = 1.22$ (see memo).", ""]
tail = ["", r"Caveats: Tier 2 ignores RSD and the $P$–$B$ cross-covariance; Tier 1 with $k_{\min} = k_f$ counts modes larger than a shell.",
        f"Figure: fisher_closed_kmax.png ($\\sigma$ vs $k_{{\\max}}$, QSO+LRG, $160^3$). Runtime {time.time()-t0:.0f}s."]
write_note("fisher_closed", "Part B closed forms: Tier 1 ($P$) and Tier 1+2 ($P+B$) field-level Fisher", head + lines + tail)
print("\n".join(head + lines + tail))
