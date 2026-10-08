"""n(z), footprint area and V_eff for DR1 LRG and QSO from the public clustering catalogues.
Checks against Chaussidon et al. 2024 Table 1 (LRG 2,130,621 / 530 deg^-2; QSO 1,189,129 / 185 deg^-2)."""
from _common import *
import numpy as np, json
from cosmology import Cosmology
from geometry import Geometry

cosmo = Cosmology(backend="class")
cfg = {"LRG": dict(zmin=0.4, zmax=1.1, P0=5e4, N_paper=2130621, dens_paper=530),
       "QSO": dict(zmin=0.8, zmax=3.1, P0=3e4, N_paper=1189129, dens_paper=185)}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
lines, out = [], {}
for ax, (t, c) in zip(axes, cfg.items()):
    g = Geometry(cosmo, t, c["zmin"], c["zmax"])
    s = g.summary(c["P0"]); s["N_paper"] = c["N_paper"]; s["dens_deg2"] = g.N_w / g.area_deg2
    s["V_eff_count"] = g.V_eff(c["P0"], "count"); s["nbar_eff_count"] = g.nbar_eff(c["P0"], "count")
    out[t] = s
    ax.plot(g.z, g.nz_nx * 1e4, label="NX column (DESI)")
    ax.plot(g.z, g.nz_count * 1e4, "--", label=f"N(z)/dV, pixel area={g.area_deg2:.0f} deg$^2$")
    ax.axhline(1e4 / c["P0"], color="k", lw=0.8, ls=":", label=f"n P0 = 1  (P0={c['P0']:.0e})")
    ax.set_xlabel("z"); ax.set_ylabel(r"$\bar n(z)$  [$10^{-4}\,(h/{\rm Mpc})^3$]"); ax.set_title(t); ax.legend(fontsize=7)
    lines += [f"## {t}  ({c['zmin']} < z < {c['zmax']})",
              f"- N (weighted) = {g.N_w:,.0f}  (paper: {c['N_paper']:,}); pixel area = {g.area_deg2:.0f} deg^2, "
              f"effective area (N/int NX dV) = {g.area_eff_deg2:.0f} deg^2 -> {g.N_w/g.area_eff_deg2:.0f} deg^-2 (paper {c['dens_paper']})",
              f"- V_total(eff area) = {g.V_total()/1e9:.2f} (Gpc/h)^3; V_eff(P0={c['P0']:.0e}) = {s['V_eff']/1e9:.2f} [NX, eff area] / {s['V_eff_count']/1e9:.2f} [count, pixel area] (Gpc/h)^3",
              f"- nbar_eff = {s['nbar_eff']:.2e} [NX] / {s['nbar_eff_count']:.2e} [count]; z_eff(FKP-weighted) = {s['z_eff']:.3f}", ""]
save_fig(fig, "geometry_nz")
json.dump(out, open(ROOT / "results" / "geometry.json", "w"), indent=1)
lines += ["Notes: NX is the catalogue density used for FKP weights (completeness-scaled). The 'count' estimate uses",
          "the HEALPix (nside=128) footprint of the galaxy catalogue itself; DR1 tiling is incomplete so this pixel area",
          "overestimates the effective area and the count-based n(z) is too low. The NX-based V_eff with the effective area",
          "is the one used in the Fisher. z_eff here is V_eff-weighted, not the pair-weighted z_eff of the paper."]
write_note("geometry_nz", "Geometry: n(z), area, V_eff for DR1 LRG/QSO", lines)
print("\n".join(lines))
