"""Stage 2 samplers on the flat state [eps, theta_w] (BlackJAX), plus chain diagnostics.

- nuts: window adaptation (step size + diagonal mass matrix) then NUTS.
- hmc: fixed number of leapfrog steps, step size adapted in the window.
- mclmc: microcanonical Langevin Monte Carlo (BlackJAX), tuned with its own adaptation.
State layout: the first res^3 entries are eps (unit prior), the rest are whitened theta.
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
import blackjax


def run_nuts(logdensity, state0, n_warmup=200, n_samples=1000, seed=0, target_accept=0.8,
             diagonal_mass=True, max_doublings=8, log=print, thin=1, unpack=None, record=None):
    """Window-adapted NUTS. `record(eps, theta) -> dict of scalars` is evaluated per kept sample."""
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=target_accept,
                                        is_mass_matrix_diagonal=diagonal_mass, progress_bar=False)
    (state, params), _ = warmup.run(kw, state0, num_steps=n_warmup)
    log(f"  warmup done: step size {float(params['step_size']):.3e}  {time.time()-t0:.0f}s")
    kernel = jax.jit(blackjax.nuts(logdensity, **params, max_num_doublings=max_doublings).step)
    return _sample(kernel, state, ks, n_samples, thin, unpack, record, log, t0)


def run_hmc(logdensity, state0, n_leapfrog=32, n_warmup=200, n_samples=1000, seed=0, target_accept=0.8,
            diagonal_mass=True, log=print, thin=1, unpack=None, record=None):
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    warmup = blackjax.window_adaptation(blackjax.hmc, logdensity, target_acceptance_rate=target_accept,
                                        is_mass_matrix_diagonal=diagonal_mass, progress_bar=False,
                                        num_integration_steps=n_leapfrog)
    (state, params), _ = warmup.run(kw, state0, num_steps=n_warmup)
    log(f"  warmup done: step size {float(params['step_size']):.3e}  {time.time()-t0:.0f}s")
    kernel = jax.jit(blackjax.hmc(logdensity, **params, num_integration_steps=n_leapfrog).step)
    return _sample(kernel, state, ks, n_samples, thin, unpack, record, log, t0)


def _sample(kernel, state, key, n_samples, thin, unpack, record, log, t0):
    thetas, recs, accept, nsteps = [], [], [], []
    for i in range(n_samples * thin):
        key, k = jax.random.split(key)
        state, info = kernel(k, state)
        if i % thin == 0:
            eps, th = unpack(state.position)
            thetas.append(np.asarray(th))
            if record is not None:
                recs.append(record(eps, th))
            accept.append(float(getattr(info, "acceptance_rate", np.nan)))
            nsteps.append(int(getattr(info, "num_integration_steps", 0)))
        if (i + 1) % max(1, (n_samples * thin) // 10) == 0:
            log(f"  sample {i+1}/{n_samples*thin}: f_NL = {float(th[0]):.2f}, accept {np.nanmean(accept[-50:]):.2f}, "
                f"mean leapfrog {np.mean(nsteps[-50:]):.0f}, {time.time()-t0:.0f}s")
    return dict(theta=np.array(thetas), records=recs, accept=np.array(accept), n_steps=np.array(nsteps),
                time=time.time() - t0, final_state=state)


def run_mclmc(logdensity, state0, n_warmup=1000, n_samples=5000, seed=0, log=print, thin=1, unpack=None, record=None):
    """MCLMC (Robnik et al.): adaptation of step size and momentum decoherence length L, then sampling."""
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    init = blackjax.mcmc.mclmc.init(position=state0, logdensity_fn=logdensity, rng_key=kw)
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity, integrator=blackjax.mcmc.integrators.isokinetic_mclachlan, inverse_mass_matrix=inverse_mass_matrix)
    state, params, _ = blackjax.mclmc_find_L_and_step_size(mclmc_kernel=kernel, num_steps=n_warmup, state=init, rng_key=kw,
                                                           diagonal_preconditioning=False)
    log(f"  mclmc tuning done: L={float(params.L):.2f} step={float(params.step_size):.3e}  {time.time()-t0:.0f}s")
    sampler = blackjax.mclmc(logdensity, L=params.L, step_size=params.step_size, inverse_mass_matrix=params.inverse_mass_matrix)
    step = jax.jit(sampler.step)
    thetas, recs = [], []
    for i in range(n_samples * thin):
        ks, k = jax.random.split(ks)
        state, info = step(k, state)
        if i % thin == 0:
            eps, th = unpack(state.position); thetas.append(np.asarray(th))
            if record is not None:
                recs.append(record(eps, th))
        if (i + 1) % max(1, (n_samples * thin) // 10) == 0:
            log(f"  sample {i+1}/{n_samples*thin}: f_NL = {float(th[0]):.2f}, {time.time()-t0:.0f}s")
    return dict(theta=np.array(thetas), records=recs, time=time.time() - t0, final_state=state, params=params)


# ---- diagnostics ---------------------------------------------------------------------------------
def autocorr(x):
    x = np.asarray(x, dtype=float); x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def tau_int(x, c=5.0):
    """Integrated autocorrelation time (Sokal windowing: smallest M with M >= c tau(M))."""
    rho = autocorr(x); n = len(rho)
    tau = 1.0 + 2.0 * np.cumsum(rho[1:])
    for M in range(1, n - 1):
        if M >= c * tau[M - 1]:
            return float(tau[M - 1])
    return float(tau[-1])


def summarize_chain(theta, names, burn=0):
    th = theta[burn:]
    out = {}
    for i, nm in enumerate(names):
        x = th[:, i]
        if np.std(x) == 0:
            continue
        t = tau_int(x)
        out[nm] = dict(mean=float(x.mean()), std=float(x.std()), tau=t, ess=float(len(x) / t))
    return out
