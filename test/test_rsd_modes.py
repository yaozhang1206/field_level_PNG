"""Mode-level check of displacement and RSD amplitudes (GPU, 80^3 box, observer far along -z)."""
from _common import *
import numpy as np, jax, jax.numpy as jnp
from cosmology import Cosmology
from forward_model import ForwardModel, ParamLayout, default_shell
res = 80; L = 10000.0; z = 1.651
cosmo = Cosmology(backend="class")
sh = default_shell("QSO", z, z, 2.44, 1.6)
fm = ForwardModel(cosmo, [sh], L=L, res=res, nz_funcs={"QSO": lambda zz: np.ones_like(zz)}, ang_mask=None, observer=[L/2, L/2, -200*L], n_order=1)
eps = jax.random.normal(jax.random.PRNGKey(3), (res,)*3, dtype=jnp.float64)
fd, dL, phi = fm.linear_fields(eps, 0.0)
psis = fm.lpt(fd)
D, f = fm.D[0], fm.f[0]
# 1. displacement rms vs expectation
psi1 = np.asarray(psis[0]); rms = np.sqrt(np.mean(psi1**2, axis=(0,1,2)))
k = fm.k_grid if hasattr(fm,'k_grid') else None
kk = np.asarray(fm.kmesh); Pk = np.asarray(fm.Pk_mesh)
w = np.ones_like(kk); w[...,0]=0.5; w[...,-1]=0.5
sig2 = np.sum(w*np.where(kk>0, Pk/np.where(kk>0,kk,1)**2, 0))*2/L**3/3   # per component, box modes only
print(f"psi_1 rms per component: {rms} ; expected sqrt(int P/k^2 d3k/(2pi)^3 /3) over box modes = {np.sqrt(sig2):.3f} Mpc/h (z=0 units)")
# 2. ZA density vs D*delta_L mode by mode
def xcorr(field, ref, kmax):
    a = np.asarray(jnp.fft.rfftn(field - jnp.mean(field))); b = np.asarray(jnp.fft.rfftn(ref))
    sel = (kk > 0) & (kk < kmax)
    return np.real(np.sum(a[sel]*np.conj(b[sel])))/np.sum(np.abs(b[sel])**2), np.sqrt(np.sum(np.abs(a[sel])**2)/np.sum(np.abs(b[sel])**2))
rho = fm.scatter(fm.positions(psis, 0, rsd=False), jnp.ones((res,)*3))
ref = D*dL
for kmax in (0.005, 0.01):
    print(f"ZA real space k<{kmax}: <d d_L*>/<|d_L|^2> = {xcorr(rho, ref, kmax)[0]:.3f}, amplitude ratio {xcorr(rho, ref, kmax)[1]:.3f}")
# 3. RSD: compare with (1 + f mu^2) D delta_L
k2 = np.where(kk>0, kk**2, 1); mu2 = np.asarray(fm.kvecs[2])**2/k2
ref_s = np.asarray(jnp.fft.irfftn(jnp.asarray((1 + f*mu2)) * jnp.fft.rfftn(ref), s=(res,)*3))
rho_s = fm.scatter(fm.positions(psis, 0, rsd=True), jnp.ones((res,)*3))
for kmax in (0.005, 0.01):
    print(f"ZA redshift space k<{kmax}: vs (1+f mu^2) D d_L: corr {xcorr(rho_s, ref_s, kmax)[0]:.3f}, amp {xcorr(rho_s, ref_s, kmax)[1]:.3f};  vs real-space ref: corr {xcorr(rho_s, ref, kmax)[0]:.3f}")
# 4. direct check of the RSD displacement: compare mu^2-weighted: project k_z modes only
sel = (kk>0)&(kk<0.008)&(mu2>0.8)
a = np.asarray(jnp.fft.rfftn(rho_s-jnp.mean(rho_s))); b = np.asarray(jnp.fft.rfftn(ref))
print(f"high-mu (mu^2>0.8), k<0.008: <d_s d_L*>/<|d_L|^2> = {np.real(np.sum(a[sel]*np.conj(b[sel])))/np.sum(np.abs(b[sel])**2):.3f}  (Kaiser expects 1 + f*<mu^2> = {1+f*np.mean(mu2[sel]):.3f})")
print(f"rsd displacement rms (f D psi_1 . zhat): {float(jnp.sqrt(jnp.mean((f*D*psis[0][...,2])**2))):.2f} Mpc/h; real displacement rms per comp {float(jnp.sqrt(jnp.mean((D*psis[0])**2))):.2f}")
