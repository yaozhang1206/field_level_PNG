"""Forward-model conventions test (GPU): (1) 2LPT matter P(k) vs P_lin at the shell redshift,
(2) galaxy P(k) for f_NL = 0, +100, -100 vs the linear expectation [b1 + bphi f_NL/T(k)]^2 P_lin,
(3) RSD: P(k, mu) ratio vs Kaiser. Uses a periodic box with nbar uniform and no mask (pure box test)."""
from _common import *
import numpy as np, time, jax, jax.numpy as jnp, json
from cosmology import Cosmology, bias_phi
from forward_model import ForwardModel, ParamLayout, default_shell

res = int(os.environ.get("RES", 80)); L = 10000.0
cosmo = Cosmology(backend="class")
z = 1.651; b1 = 2.44; p = 1.6
# a single "shell" spanning the whole box so the field is periodic and uniform
sh = default_shell("QSO", z, z, b1, p)             # shell centre = z (selection is bypassed below)
nz = {"QSO": lambda zz: np.full_like(zz, 1.0)}
# observer far away along -z so the line of sight is the z axis (plane-parallel Kaiser check)
fm = ForwardModel(cosmo, [sh], L=L, res=res, nz_funcs=nz, ang_mask=None, observer=[L / 2, L / 2, -200 * L])
fm.nbar_cell[0] = jnp.ones((res,) * 3)              # pure 1 + delta_g
lay = ParamLayout([sh])
eps = jax.random.normal(jax.random.PRNGKey(1), (res,) * 3, dtype=jnp.float64)
lines = [f"res $={res}$, $L={L}$, cell $={L/res:.1f}$ Mpc/$h$, $z={z}$, $b_1={b1}$, $p={p}$, $b_\\phi={sh.bphi:.3f}$, $b_2={sh.b2:.3f}$, $b_K={sh.bK:.3f}$, $b_{{\\phi\\delta}}={sh.bphidelta:.3f}$",
         f"growth $D={fm.D[0]:.4f}$, $f={fm.f[0]:.4f}$; $k_{{\\rm Ny}}={np.pi*res/L:.4f}$"]

t0 = time.time()
dm = fm.matter_field(eps, 0.0, 0, rsd=False); dm.block_until_ready(); t_m = time.time() - t0
k, Pm = fm.power_spectrum(dm, nbins=18)
Plin = np.asarray(cosmo.P_lin(k, z))
lines.append(f"matter field (2LPT, no RSD): {t_m:.1f}s first call; $P_m/P_{{\\rm lin}}$ at $k<0.02$: {np.mean(Pm[k<0.02]/Plin[k<0.02]):.3f}")

Pg, Pg_lin = {}, {}
for fnl in (0.0, 100.0, -100.0):
    th = lay.fiducial(fnl)
    t0 = time.time(); g = fm(eps, th, lay, rsd=False)["QSO"]; g.block_until_ready(); dt = time.time() - t0
    k, Pg[fnl] = fm.power_spectrum(g, nbins=18)
    T = np.asarray(cosmo.T_Phi_delta(k, z))
    Pg_lin[fnl] = (b1 + sh.bphi * fnl / T) ** 2 * Plin
    r = Pg[fnl] / Pg_lin[fnl]
    lines.append(f"galaxy field $f_{{\\rm NL}}={fnl:+.0f}$: {dt:.1f}s; $P_g/[(b_1 + b_\\phi f_{{\\rm NL}}/T)^2 P_{{\\rm lin}}]$ at $k<0.01$: {np.mean(r[k<0.01]):.3f}, $0.01<k<0.02$: {np.mean(r[(k>0.01)&(k<0.02)]):.3f}")
# RSD
th = lay.fiducial(0.0)
gs = fm(eps, th, lay, rsd=True)["QSO"]
mu_edges = np.array([0, 0.33, 0.66, 1.0]); k2, Pmu = fm.power_spectrum(gs, nbins=18, mu_bins=mu_edges)
_, Pmu_real = fm.power_spectrum(fm(eps, th, lay, rsd=False)["QSO"], nbins=18, mu_bins=mu_edges)   # same modes, cancels cosmic variance
f = fm.f[0]
kaiser = [np.mean([(b1 + f * m ** 2) ** 2 for m in np.linspace(lo, hi, 20)]) / b1 ** 2 for lo, hi in zip(mu_edges[:-1], mu_edges[1:])]
for (lo, hi), P, Pr, kz in zip(zip(mu_edges[:-1], mu_edges[1:]), Pmu, Pmu_real, kaiser):
    sel = (k2 > 0.003) & (k2 < 0.015)
    lines.append(f"RSD $\\mu\\in[{lo:.2f},{hi:.2f})$: $P_s(k,\\mu)/P_{{\\rm real}}(k,\\mu)$ at $0.003<k<0.015$ = {np.mean(P[sel]/Pr[sel]):.3f}  (Kaiser $\\langle(1+\\beta\\mu^2)^2\\rangle$: {kz:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].loglog(k, Plin, "k-", label="P_lin(z)"); axes[0].loglog(k, Pm, "o", ms=4, label="2LPT matter")
for fnl, c in zip((0.0, 100.0, -100.0), ("C0", "C1", "C2")):
    axes[0].loglog(k, Pg[fnl], "s", ms=3, color=c, label=f"galaxy f_NL={fnl:+.0f}"); axes[0].loglog(k, Pg_lin[fnl], "--", color=c, lw=0.8)
axes[0].set_xlabel("k [h/Mpc]"); axes[0].set_ylabel("P(k)"); axes[0].legend(fontsize=7); axes[0].set_title("Box P(k) vs linear theory (dashed)")
for fnl, c in zip((0.0, 100.0, -100.0), ("C0", "C1", "C2")):
    axes[1].semilogx(k, Pg[fnl] / Pg_lin[fnl], "s-", ms=3, color=c, label=f"f_NL={fnl:+.0f}")
axes[1].semilogx(k, Pm / Plin, "ko-", ms=3, label="matter"); axes[1].axhline(1, color="k", lw=0.5); axes[1].set_ylim(0.6, 1.4)
axes[1].set_xlabel("k [h/Mpc]"); axes[1].set_ylabel("P_measured / P_linear-model"); axes[1].legend(fontsize=7); axes[1].set_title("Ratio to [b1 + bphi f_NL/T]^2 P_lin")
for (lo, hi), P, Pr, kz in zip(zip(mu_edges[:-1], mu_edges[1:]), Pmu, Pmu_real, kaiser):
    l, = axes[2].semilogx(k2, P / Pr, "o-", ms=3, label=f"mu in [{lo:.2f},{hi:.2f})"); axes[2].axhline(kz, color=l.get_color(), ls="--", lw=0.8)
axes[2].set_xlabel("k [h/Mpc]"); axes[2].set_ylabel("P_s(k,mu) / P_real(k,mu)"); axes[2].legend(fontsize=7); axes[2].set_title("RSD vs Kaiser (dashed), LOS = z")
save_fig(fig, f"forward_model_res{res}")
lines += ["", r"Expectations: matter ratio $\approx1$ at $k < k_{\rm Ny}/3$ (CIC + 2LPT), galaxy ratio $\approx1$ at low $k$ for all $f_{\rm NL}$ (tests the PNG",
          r"normalisation $T_{\Phi\to\delta}$ and $b_\phi$), RSD ratios match Kaiser at low $k$ (observer placed far along $-z$ so the LOS is the $z$ axis)."]
write_note(f"forward_model_res{res}", f"Forward model conventions (DISCO-DJ 2LPT + PNG + bias + RSD), ${res}^3$", lines)
print("\n".join(lines))
