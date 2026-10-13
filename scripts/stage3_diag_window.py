"""Stage 3 diagnostic: grid-window consistency of data and model. Raw grid power (no deconvolution) of
NGP-gridded halos vs the forward model at the measured bias, for oversample = 1 (lattice CIC, Stage 1-2 model),
2 and 4 (block-averaged fine lattice). Real and redshift space (LOS = z).
    python scripts/stage3_diag_window.py --sample fiducial_HR_z0.5_LRG --ids 0-3 --res 16
"""
import argparse, os, sys, json
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import jax, jax.numpy as jnp
from quijote import quijote_cosmology, fourier_grid, ngp_mesh, cic_mesh, redshift_space, L_QUIJOTE
from forward_model import ForwardModel, ParamLayout, default_shell
sys.path.insert(0, os.path.join(ROOT, "test"))
from _common import save_fig, write_note, plt

ap = argparse.ArgumentParser(); ap.add_argument("--sample", required=True); ap.add_argument("--ids", default="0-3")
ap.add_argument("--res", type=int, default=16); ap.add_argument("--p", type=float, default=1.0); a = ap.parse_args()
L, res = L_QUIJOTE, a.res
lo, hi = map(int, a.ids.split("-")); ids = list(range(lo, hi + 1))
sdir = f"/work/hdd/bdne/yzhang116/quijote_cuts/{a.sample}"; summ = json.load(open(f"{sdir}/summary.json")); z = summ["z"]
b1 = float(np.mean([summ["realisations"][str(i)]["b1"] for i in ids]))
c = quijote_cosmology()
kf = 2 * np.pi / L; nb = 8
kx = np.fft.fftfreq(res, 1 / res) * kf; kz = np.fft.rfftfreq(res, 1 / res) * kf
KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij"); K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
MU2 = np.where(K > 0, KZ ** 2 / np.where(K > 0, K, 1) ** 2, 0)
edges = np.linspace(kf * 0.99, np.sqrt(3) * np.pi * res / L, nb + 1); idx = np.digitize(K.ravel(), edges) - 1
def pk(field, poles=False):
    f = np.fft.rfftn(field / field.mean() - 1) / res ** 3; P = (np.abs(f) ** 2 * L ** 3).ravel()
    w = np.full(K.shape, 2.0); w[..., 0] = 1; w[..., -1] = 1; w = w.ravel()
    out = []
    for j in range(nb):
        s = idx == j
        out.append(np.sum(P[s] * w[s]) / np.sum(w[s]) if s.any() else np.nan)
    return np.array(out)
kc = np.array([np.mean(K.ravel()[idx == j]) for j in range(nb)])
Plin = np.asarray(c.P_lin(kc, z))

rows = {}
for rsd in (False, True):
    Pd = []
    for i in ids:
        d = np.load(f"{sdir}/{i}.npz"); pos = d["pos"].astype(float)
        if rsd: pos = redshift_space(pos, d["vel"].astype(float), z, c, L)
        shot = L ** 3 / len(pos)
        Pd.append(pk(fourier_grid(pos, L, res, 4)) - shot)
    rows[("data", rsd)] = np.mean(Pd, 0) / Plin
    nbar = np.mean([summ["realisations"][str(i)]["n"] for i in ids])
    for os_, mos_ in ((1, 2), (1, 4), (2, 2), (2, 4), (4, 4)):
        sh = default_shell("LRG", z, z, b1, a.p)
        fm = ForwardModel(c, [sh], L=L, res=res, nz_funcs={"LRG": lambda zz: np.full_like(zz, nbar)}, observer=[L / 2, L / 2, -200 * L], oversample=os_, mesh_oversample=mos_)
        fm.nbar_cell[0] = jnp.full((res,) * 3, nbar * fm.V_cell); fm.footprint = {"LRG": np.ones((res,) * 3, bool)}
        lay = ParamLayout([sh]); th = lay.fiducial(0.0)
        f = jax.jit(lambda e: fm(e, th, lay, rsd)["LRG"])
        Pm = [pk(np.asarray(f(jax.random.normal(jax.random.PRNGKey(s), (res,) * 3, dtype=jnp.float64)))) for s in range(16)]
        rows[(f"model p{os_}m{mos_}", rsd)] = np.mean(Pm, 0) / Plin
    print(f"--- {'redshift' if rsd else 'real'} space: raw grid P / P_lin (b1^2 = {b1**2:.2f}); data NGP-gridded, shot-noise subtracted")
    print("k      " + "  ".join(f"{kk:.3f}" for kk in kc))
    for key, v in rows.items():
        if key[1] == rsd: print(f"{key[0]:12s} " + "  ".join(f"{x:5.2f}" for x in v))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, rsd in zip(axes, (False, True)):
    for key, v in rows.items():
        if key[1] == rsd: ax.plot(kc, v, "o-" if key[0] == "data" else ".--", label=key[0])
    ax.axhline(b1 ** 2, color="gray", lw=0.6); ax.set_xlabel("k [h/Mpc]"); ax.set_title(("redshift" if rsd else "real") + " space", fontsize=9)
axes[0].set_ylabel("raw grid $P(k)/P_{\\rm lin}$"); axes[0].legend(fontsize=8)
tag = f"stage3_window_{a.sample}_res{res}"
save_fig(fig, tag)
lines = [f"Sample {a.sample}, realisations {a.ids}, z = {z}, measured $b_1$ = {b1:.3f}; {res}$^3$ grid, cell {L/res:.1f} Mpc/h.",
         "Data: CIC on the $64^3$ mesh truncated to the $16^3$ grid's Fourier modes, shot noise $L^3/N$ subtracted. Model: same downsampling, forward model at the data's",
         "$b_1$ (Lazeyras $b_2$, $b_K = -2/7(b_1-1)$, $f_{\\rm NL} = 0$), 16 random fields. Raw grid power, no window deconvolution, so model and",
         "data should agree if the model's grid window matches the data's.", "", "| k | " + " | ".join(f"{kk:.3f}" for kk in kc) + " |", "|---|" + "---|" * nb]
for key, v in rows.items():
    lines.append(f"| {key[0]} ({'s' if key[1] else 'r'}) | " + " | ".join(f"{x:.2f}" for x in v) + " |")
write_note(tag, "Stage 3: grid-window consistency of NGP-gridded halos and the oversampled forward model", lines)
