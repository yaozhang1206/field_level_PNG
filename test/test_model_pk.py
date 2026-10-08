"""Sanity test of the Chaussidon et al. 2024 P(k) model: multipoles for f_NL = 0, +-50 and
the scale-dependent-bias term b_phi f_NL / T_Phi->delta(k)."""
from _common import *
import numpy as np, jax.numpy as jnp
from cosmology import Cosmology, bias_phi
from model_pk import PkModel

cosmo = Cosmology(backend="class")
k = np.logspace(np.log10(0.002), np.log10(0.3), 200)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
lines = []
for z, b1, p, lab in [(1.651, 2.44, 1.6, "QSO FKP"), (0.733, 2.06, 1.0, "LRG FKP")]:
    m = PkModel(cosmo, z, p)
    for fnl, ls in [(0, "-"), (50, "--"), (-50, ":")]:
        P = np.asarray(m.multipoles(k, (fnl, b1, 0.0, 4.0)))
        axes[0].loglog(k, k * P[0], ls, label=f"{lab} f_NL={fnl}" if fnl else lab)
        axes[1].semilogx(k, P[1] / P[0], ls)
    Tk = np.asarray(cosmo.T_Phi_delta(k, z))
    axes[2].loglog(k, bias_phi(b1, p) * 50 / Tk, label=f"{lab}: b_phi f_NL/T, f_NL=50 (b_phi={bias_phi(b1,p):.2f})")
    lines.append(f"- {lab}: z={z}, b1={b1}, p={p}: f={m.f:.3f}, b_phi={bias_phi(b1,p):.3f}, "
                 f"Delta b(k=0.005)={bias_phi(b1,p)*50/np.interp(0.005,k,Tk):.3f} for f_NL=50, "
                 f"P0(k=0.02)={np.interp(0.02,k,np.asarray(m.multipoles(k,(0,b1,0,4.0)))[0]):.0f}")
axes[0].set_xlabel("k [h/Mpc]"); axes[0].set_ylabel("k P_0(k)"); axes[0].legend(fontsize=7); axes[0].set_title("Monopole, f_NL=0 / +50 (dashed) / -50 (dotted)")
axes[1].set_xlabel("k [h/Mpc]"); axes[1].set_ylabel("P_2 / P_0"); axes[1].set_title("Quadrupole ratio")
axes[2].set_xlabel("k [h/Mpc]"); axes[2].set_ylabel(r"$\Delta b(k)$"); axes[2].legend(fontsize=7); axes[2].set_title(r"Scale-dependent bias $\propto k^{-2}$")
save_fig(fig, "model_pk")
lines = ["Model: P(k,mu) = [b1 + b_phi f_NL / T(k) + f mu^2]^2 / [1 + (k mu Sigma_s)^2/2]^2 P_lin + s_n0, Sigma_s=4 Mpc/h.",
         "Kaiser quadrupole ratio for f_NL=0 should be ~ (4/3 beta + 4/7 beta^2)/(1 + 2/3 beta + beta^2/5) at low k.", ""] + lines
for z, b1, lab in [(1.651, 2.44, "QSO"), (0.733, 2.06, "LRG")]:
    beta = float(cosmo.f(z)) / b1
    lines.append(f"- {lab}: expected low-k P2/P0 = {(4/3*beta+4/7*beta**2)/(1+2/3*beta+beta**2/5):.3f}")
write_note("model_pk", "P(k) model sanity check (Chaussidon+24 Eq. 2.4)", lines)
print("\n".join(lines))
