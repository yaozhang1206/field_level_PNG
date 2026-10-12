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
             diagonal_mass=True, max_doublings=8, log=print, thin=1, unpack=None, record=None, checkpoint=None):
    """Window-adapted NUTS. `record(eps, theta) -> dict of scalars` is evaluated per kept sample."""
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity, target_acceptance_rate=target_accept,
                                        is_mass_matrix_diagonal=diagonal_mass, progress_bar=False)
    (state, params), _ = warmup.run(kw, state0, num_steps=n_warmup)
    log(f"  warmup done: step size {float(params['step_size']):.3e}  {time.time()-t0:.0f}s")
    kernel = jax.jit(blackjax.nuts(logdensity, **params, max_num_doublings=max_doublings).step)
    return _sample(kernel, state, ks, n_samples, thin, unpack, record, log, t0, checkpoint)


def run_nuts_da(logdensity, state0, n_warmup=200, n_samples=1000, seed=0, target_accept=0.8, max_doublings=7,
                log=print, thin=1, unpack=None, record=None, checkpoint=None, step_size0=0.02):
    """NUTS with identity mass matrix and dual-averaging step-size adaptation in a Python loop: no storage of
    warmup positions (BlackJAX's window adaptation keeps all of them, 6.5 GB at 160^3)."""
    from blackjax.adaptation.step_size import dual_averaging_adaptation
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    n = state0.size
    imm = jnp.ones(n)
    kernel_fn = blackjax.nuts.build_kernel()
    step = jax.jit(lambda k, st, ss: kernel_fn(k, st, logdensity, ss, imm, max_num_doublings=max_doublings))
    state = blackjax.nuts.init(state0, logdensity)
    da_init, da_update, da_final = dual_averaging_adaptation(target=target_accept)
    da = da_init(step_size0); ss = step_size0
    for i in range(n_warmup):
        key, k = jax.random.split(key)
        state, info = step(k, state, ss)
        da = da_update(da, info.acceptance_rate)
        ss = float(jnp.exp(da.log_step_size))
        if (i + 1) % max(1, n_warmup // 5) == 0:
            log(f"  warmup {i+1}/{n_warmup}: step {ss:.3e}, accept {float(info.acceptance_rate):.2f}, leapfrog {int(info.num_integration_steps)}, {time.time()-t0:.0f}s")
    ss = float(da_final(da))
    log(f"  warmup done: step size {ss:.3e}  {time.time()-t0:.0f}s")
    kernel = jax.jit(lambda k, st: step(k, st, ss))
    return _sample(kernel, state, key, n_samples, thin, unpack, record, log, t0, checkpoint)


def run_nuts_window_loop(logdensity, state0, n_warmup=200, n_samples=1000, seed=0, target_accept=0.8, max_doublings=7,
                         log=print, thin=1, unpack=None, record=None, checkpoint=None, step_size0=0.02, resume=None):
    """resume: dict(position, step_size, inverse_mass_matrix) to continue a chain without warmup."""
    """Stan-style window adaptation (diagonal mass + dual-averaging step size) run as a Python loop so that no
    warmup positions are stored (BlackJAX's window_adaptation.run keeps all of them: 6.5 GB at 160^3)."""
    from blackjax.adaptation.window_adaptation import base, build_schedule
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    kernel_fn = blackjax.nuts.build_kernel()
    step = jax.jit(lambda k, st, ss, imm: kernel_fn(k, st, logdensity, ss, imm, max_num_doublings=max_doublings))
    a_init, a_update, a_final = base(is_mass_matrix_diagonal=True, target_acceptance_rate=target_accept)
    a_update = jax.jit(a_update)
    if resume is not None:
        state = blackjax.nuts.init(jnp.asarray(resume["position"]), logdensity)
        ss, imm = float(resume["step_size"]), jnp.asarray(resume["inverse_mass_matrix"])
        log(f"  resumed: step size {ss:.3e}, no warmup")
        kernel = jax.jit(lambda k, st: step(k, st, ss, imm))
        out = _sample(kernel, state, jax.random.PRNGKey(seed + 7919), n_samples, thin, unpack, record, log, t0, checkpoint)
        out["step_size"], out["inverse_mass_matrix"] = ss, np.asarray(imm)
        return out
    state = blackjax.nuts.init(state0, logdensity)
    astate = a_init(state.position, step_size0)
    schedule = build_schedule(n_warmup)
    for i, stage in enumerate(schedule):
        key, k = jax.random.split(key)
        state, info = step(k, state, astate.ss_state.step_size if hasattr(astate.ss_state, "step_size") else jnp.exp(astate.ss_state.log_step_size),
                           astate.imm_state.inverse_mass_matrix)
        astate = a_update(astate, stage, state.position, info.acceptance_rate)
        if (i + 1) % max(1, n_warmup // 5) == 0:
            log(f"  warmup {i+1}/{n_warmup}: step {float(jnp.exp(astate.ss_state.log_step_size)):.3e}, accept {float(info.acceptance_rate):.2f}, leapfrog {int(info.num_integration_steps)}, {time.time()-t0:.0f}s")
    ss, imm = a_final(astate)
    log(f"  warmup done: step size {float(ss):.3e}, inv. mass range [{float(imm.min()):.2e}, {float(imm.max()):.2e}]  {time.time()-t0:.0f}s")
    kernel = jax.jit(lambda k, st: step(k, st, ss, imm))
    out = _sample(kernel, state, key, n_samples, thin, unpack, record, log, t0, checkpoint)
    out["step_size"], out["inverse_mass_matrix"] = float(ss), np.asarray(imm)
    return out


def run_hmc(logdensity, state0, n_leapfrog=32, n_warmup=200, n_samples=1000, seed=0, target_accept=0.8,
            diagonal_mass=True, log=print, thin=1, unpack=None, record=None, checkpoint=None):
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    warmup = blackjax.window_adaptation(blackjax.hmc, logdensity, target_acceptance_rate=target_accept,
                                        is_mass_matrix_diagonal=diagonal_mass, progress_bar=False,
                                        num_integration_steps=n_leapfrog)
    (state, params), _ = warmup.run(kw, state0, num_steps=n_warmup)
    log(f"  warmup done: step size {float(params['step_size']):.3e}  {time.time()-t0:.0f}s")
    kernel = jax.jit(blackjax.hmc(logdensity, **params, num_integration_steps=n_leapfrog).step)
    return _sample(kernel, state, ks, n_samples, thin, unpack, record, log, t0, checkpoint)


def _sample(kernel, state, key, n_samples, thin, unpack, record, log, t0, checkpoint=None):
    """checkpoint(thetas, accept, nsteps) is called every 250 kept samples (partial results)."""
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
        if checkpoint is not None and len(thetas) % 250 == 0 and i % thin == 0:
            checkpoint(np.array(thetas), np.array(accept), np.array(nsteps), np.asarray(state.position))
    return dict(theta=np.array(thetas), records=recs, accept=np.array(accept), n_steps=np.array(nsteps),
                time=time.time() - t0, final_state=state)


def run_mclmc(logdensity, state0, n_warmup=1000, n_samples=5000, seed=0, log=print, thin=1, unpack=None, record=None,
              L_override=None, step_override=None, step_factor=1.0, checkpoint=None):
    """L_override / step_override replace the tuned values; step_factor rescales the tuned step size (bias checks)."""
    """MCLMC (Robnik et al.): adaptation of step size and momentum decoherence length L, then sampling."""
    key = jax.random.PRNGKey(seed)
    key, kw, ks = jax.random.split(key, 3)
    t0 = time.time()
    init = blackjax.mcmc.mclmc.init(position=state0, logdensity_fn=logdensity, rng_key=kw)
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity, integrator=blackjax.mcmc.integrators.isokinetic_mclachlan, inverse_mass_matrix=inverse_mass_matrix)
    state, params, _ = blackjax.mclmc_find_L_and_step_size(mclmc_kernel=kernel, num_steps=n_warmup, state=init, rng_key=kw,
                                                           diagonal_preconditioning=False)
    L = float(params.L) if L_override is None else float(L_override)
    ss = (float(params.step_size) if step_override is None else float(step_override)) * step_factor
    log(f"  mclmc tuning done: L={float(params.L):.2f} step={float(params.step_size):.3e}; using L={L:.2f} step={ss:.3e}  {time.time()-t0:.0f}s")
    sampler = blackjax.mclmc(logdensity, L=L, step_size=ss, inverse_mass_matrix=params.inverse_mass_matrix)
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
        if checkpoint is not None and len(thetas) % 5000 == 0 and i % thin == 0:
            checkpoint(np.array(thetas), np.zeros(len(thetas)), np.ones(len(thetas)), np.asarray(state.position))
    return dict(theta=np.array(thetas), records=recs, time=time.time() - t0, final_state=state, params=params,
                step_size=ss, inverse_mass_matrix=np.asarray(params.inverse_mass_matrix), L=L)


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
