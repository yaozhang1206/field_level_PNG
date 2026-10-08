"""Stage 1 Part A: reproduce the DESI DR1 P(k) f_NL statistical errors with a Gaussian Fisher.

Outputs: results/fisher_pk.json, test/figures/fisher_pk_*.png, test/notes/fisher_pk.md
Run:  source env/setup_env.sh && python scripts/run_fisher_pk.py
"""
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "test"))
import numpy as np
from _common import plt, save_fig, write_note
from cosmology import Cosmology, bias_phi
from geometry import Geometry
from fisher_pk import FisherPk, FisherPkShells, sigma_from_fisher, add_prior

t0 = time.time()
cosmo = Cosmology(backend="class")
geom = {"LRG": Geometry(cosmo, "LRG", 0.4, 1.1), "QSO": Geometry(cosmo, "QSO", 0.8, 3.1)}
P0 = {"LRG": 5e4, "QSO": 3e4}
b1_of_z = {"LRG": lambda z: 0.209 * (1 + z) ** 2 + 1.415, "QSO": lambda z: 0.237 * (1 + z) ** 2 + 0.771}
DESI = {  # clean EZmock errors, Chaussidon+24 Tables 4-5: (lower, upper)
    "QSO_FKP": (13, 18), "QSO_OQE": (10, 13), "LRG_FKP": (11, 18), "QSO+LRG_OQE": (7.9, 10.5)}

NZ_KIND = os.environ.get("NZ_KIND", "full")   # 'full' (geometric footprint + completeness) or 'nx' (effective area)
configs = {
    "QSO_FKP": dict(tracer="QSO", z_eff=1.651, b1=2.44, p=1.6, kmin=0.003, kmax=0.08),
    "QSO_OQE_singlez": dict(tracer="QSO", z_eff=2.082, b1=3.08, p=1.6, kmin=0.003, kmax=0.08),
    "LRG_FKP": dict(tracer="LRG", z_eff=0.733, b1=2.06, p=1.0, kmin=0.006, kmax=0.08),
    "LRG_FKP_kmin003": dict(tracer="LRG", z_eff=0.733, b1=2.06, p=1.0, kmin=0.003, kmax=0.08),
}
res, F_all, lines = {}, {}, []


def report(name, F, Fbphi=None, extra=""):
    s = sigma_from_fisher(F, 0)
    s_b1fixed = sigma_from_fisher(F, 0, fixed=(1,))
    s_Sfixed = sigma_from_fisher(F, 0, fixed=(3,))
    s_Sprior = sigma_from_fisher(add_prior(F, 3, 2.0), 0)
    out = dict(sigma_fnl=s, sigma_fnl_b1_fixed=s_b1fixed, sigma_fnl_Sigmas_fixed=s_Sfixed,
               sigma_fnl_Sigmas_prior2=s_Sprior)
    if Fbphi is not None:
        out["sigma_bphi_fnl"] = sigma_from_fisher(Fbphi, 0)
    res[name] = out
    d = DESI.get(name.replace("_singlez", "").replace("_shells", "_OQE").replace("_kmin003", ""))
    tgt = f"DESI {d[0]}-{d[1]} (mean {np.mean(d):.1f})" if d else "-"
    lines.append(f"| {name} | {s:.1f} | {s_b1fixed:.1f} | {s_Sfixed:.1f} | {s_Sprior:.1f} | "
                 f"{out.get('sigma_bphi_fnl', float('nan')):.1f} | {tgt} | {extra} |")
    return out


# ---- single-z configurations ------------------------------------------------------------------
for name, c in configs.items():
    g = geom[c["tracer"]]
    fp = FisherPk(cosmo, g, c["z_eff"], c["b1"], c["p"], P0[c["tracer"]], c["kmin"], c["kmax"], nz_kind=NZ_KIND)
    F = fp.fisher(); Fb = fp.fisher(bphi_fnl=True)
    F_all[name] = (fp, F)
    report(name, F, Fb, extra=f"z_eff={c['z_eff']}, b1={c['b1']}, b_phi={bias_phi(c['b1'], c['p']):.2f}")
    print(name, res[name], f"{time.time()-t0:.0f}s", flush=True)

# ---- redshift-shell (optimal-weighting limit) ----------------------------------------------------
for tr, kmin in [("QSO", 0.003), ("LRG", 0.006)]:
    p = 1.6 if tr == "QSO" else 1.0
    fs = FisherPkShells(cosmo, geom[tr], b1_of_z[tr], p, P0[tr], kmin, 0.08, nshell=10, nz_kind=NZ_KIND)
    F = fs.fisher(); F_all[f"{tr}_shells"] = (fs, F)
    report(f"{tr}_shells", F, extra=f"10 shells, b1(z) template, p={p}")
    print(f"{tr}_shells", res[f"{tr}_shells"], f"{time.time()-t0:.0f}s", flush=True)


# ---- combinations (f_NL shared, nuisance parameters independent) -----------------------------------
def combine(Fs):
    n = 1 + 3 * len(Fs)
    F = np.zeros((n, n))
    for i, Fi in enumerate(Fs):
        idx = [0] + [1 + 3 * i + j for j in range(3)]
        F[np.ix_(idx, idx)] += Fi
    return F


for name, parts in [("QSO+LRG_FKP", ("QSO_FKP", "LRG_FKP")),
                    ("QSO+LRG_OQE_singlez", ("QSO_OQE_singlez", "LRG_FKP")),
                    ("QSO+LRG_shells", ("QSO_shells", "LRG_shells"))]:
    F = combine([F_all[p][1] for p in parts])
    s = sigma_from_fisher(F, 0)
    res[name] = dict(sigma_fnl=s)
    d = DESI["QSO+LRG_OQE"]
    lines.append(f"| {name} | {s:.1f} | - | - | - | - | DESI {d[0]}-{d[1]} (mean {np.mean(d):.1f}) | {'+'.join(parts)} |")
    print(name, s, flush=True)

# ---- sigma vs k_min and k_max (Fig. 8 analogue) ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
kcuts = {}
for name in ("QSO_FKP", "LRG_FKP_kmin003", "QSO_OQE_singlez"):
    fp, _ = F_all[name]
    Fk = fp.info_density()                                           # (nk, 4, 4)
    k = fp.k
    cum = np.cumsum(Fk, axis=0)                                      # F(k_max)
    rev = np.cumsum(Fk[::-1], axis=0)[::-1]                          # F(k_min)
    s_kmax = [sigma_from_fisher(cum[i], 0) if i > 4 else np.nan for i in range(len(k))]
    s_kmin = [sigma_from_fisher(rev[i], 0) if i < len(k) - 5 else np.nan for i in range(len(k))]
    axes[0].plot(k, s_kmax, label=name); axes[1].plot(k, s_kmin, label=name)
    kcuts[name] = dict(k=k.tolist(), sigma_kmax=np.array(s_kmax).tolist(), sigma_kmin=np.array(s_kmin).tolist())
axes[0].set_xlabel("k_max [h/Mpc]"); axes[0].set_ylabel(r"$\sigma(f_{NL})$"); axes[0].set_ylim(0, 60); axes[0].legend(fontsize=8)
axes[0].set_title("vs k_max (k_min = 0.003 / 0.006)")
axes[1].set_xlabel("k_min [h/Mpc]"); axes[1].set_ylim(0, 60); axes[1].set_xlim(0.003, 0.03); axes[1].set_title("vs k_min (k_max = 0.08)")
save_fig(fig, f"fisher_pk_kcuts_{NZ_KIND}")

# ---- information density -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for name in ("QSO_FKP", "LRG_FKP"):
    fp, _ = F_all[name]
    Fk = fp.info_density(); k = fp.k
    # marginalised information density: d(1/sigma^2)/dlnk using the full-F marginalisation weights
    Finv = np.linalg.inv(Fk.sum(0)); w = Finv[0] / Finv[0, 0]       # row vector: d theta_eff / d F
    dens = np.einsum("i,kij,j->k", w, Fk, w) * k / fp.dk            # per ln k
    ax.semilogx(k, dens / dens.max(), label=name)
ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel(r"$dF_{f_{NL}}^{\rm marg}/d\ln k$ (normalised)"); ax.legend(fontsize=8)
ax.set_title("Where the f_NL information comes from")
save_fig(fig, f"fisher_pk_info_density_{NZ_KIND}")

# ---- outputs --------------------------------------------------------------------------------------
json.dump(dict(nz_kind=NZ_KIND, results=res, kcuts=kcuts, geometry={t: g.summary(P0[t]) for t, g in geom.items()},
               configs=configs, desi_targets=DESI), open(ROOT / "results" / f"fisher_pk_{NZ_KIND}.json", "w"), indent=1)
hdr = [f"Gaussian Fisher, P0+P2, Dk=0.001, FKP-weighted covariance with the DR1 n(z) (NX column); nz_kind={NZ_KIND}",
       "('full': sample variance over the geometric DR1 footprint, shot noise from the observed density c_bar*NX;",
       " 'nx': everything over the completeness-weighted effective area).",
       "Parameters (f_NL, b1, s_n0, Sigma_s); fiducial f_NL=0, s_n0=0, Sigma_s=3 Mpc/h. Cosmology Planck 2018 via CLASS.",
       f"Geometry: " + "; ".join(f"{t}: V_eff={g.V_eff(P0[t])/1e9:.2f} (Gpc/h)^3, nbar_eff={g.nbar_eff(P0[t]):.2e}, area_eff={g.area_eff_deg2:.0f} deg^2, area_full={g.area_full_deg2:.0f} deg^2, c_bar={g.cbar:.2f}" for t, g in geom.items()),
       "", "| config | sigma(f_NL) marg. | b1 fixed | Sigma_s fixed | Sigma_s prior 2 | sigma(b_phi f_NL) | DESI clean-mock | notes |",
       "|---|---|---|---|---|---|---|---|"]
tail = ["", "DESI numbers are the asymmetric EZmock errors (Tables 4-5); the Fisher is symmetric, target is ~20% of the mean.",
        "'OQE_singlez' evaluates the model at the OQE z_eff/b1 but keeps the FKP n(z) weighting in the covariance;",
        "'shells' sums 10 redshift shells with the b1(z) template and is the optimal-weighting (OQE) limit.",
        f"Figures: fisher_pk_kcuts_{NZ_KIND}.png (sigma vs k_max / k_min), fisher_pk_info_density_{NZ_KIND}.png (where the signal is)."]
write_note(f"fisher_pk_{NZ_KIND}", "Part A: DESI DR1 P(k) Fisher vs Chaussidon et al. 2024", hdr + lines + tail)
print("\n".join(hdr + lines + tail))
print(f"total {time.time()-t0:.0f}s")
