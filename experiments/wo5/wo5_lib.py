"""Shared harness for the differentiable Hodgkin-Huxley experiments.

The equations below are transcribed from the rate functions and `HH.forward` of the
original model in mimic_cell_activity/HH.py, with two additions:

  1. `shift` -- a voltage offset applied to the sodium activation rate `alpha_m`, so
     that a variant can be expressed as a change in gating kinetics rather than in
     conductance (`shift_n` does the same for potassium activation).  HH.py has no
     such parameter.  With shift = 0 every function below is bit-identical to HH.py.
  2. a batched, torch.compile-fused integrator that keeps only running summary
     statistics instead of the whole trace, so that hundreds of millions of
     parameter sets can be swept without storing their trajectories.

Spike counting reproduces the spike detector of the original R859C search pipeline,
including the every-tenth-sample downsampling that the NEURON protocol
(Neuron_Test/R859C/neuron.hoc) applies before a trace is written.

The similarity scale is the one used by that pipeline:
        similarity % = (1 - candidate distance / untreated variant's distance) * 100
"""
import numpy as np
import torch

# ---- constants, as in HH.py (mV, mS/cm^2, uF/cm^2) ---------------------------
E_NA, E_K, E_L, CM = 50.0, -77.0, -54.0, 1.0
WT = (120.0, 36.0, 0.03)          # wild-type reference conductances (HH.py defaults)
VARIANT = (140.0, 20.0, 0.30)     # the variant point of the worked example
NAMES = ("Na", "K", "l")

# the extended protocol, shaped like the NEURON protocol in neuron.hoc: 35 current
# levels, 300 ms, 0.01 ms step, every 10th sample logged.
DT_EXT, STEPS_EXT, LOG_EVERY, N_CUR = 0.01, 30000, 10, 35
# injected-current levels for the extended protocol (I_LOW + k * I_STEP), chosen so
# that the wild type is silent at the low levels and fires 18-25 spikes at the high
# ones -- the same shape as the NEURON protocol, which is silent for its first 13 of
# 35 levels.
I_LOW, I_STEP = 0.0, 0.5
# the short protocol used for the gradient fits: 1500 steps over 10 ms at I = 8
DT_R2, STEPS_R2, I_R2 = 10.0 / 1500.0, 1500, 8.0

SPIKE_THRESHOLD = -10.0           # mV, the pipeline's spike threshold


# ---- rate functions, as in HH.py ---------------------------------------------
def alphaN(v, shift_n=0.0):
    """HH.alphaN.  `shift_n` displaces potassium activation along the voltage axis.
    It is the analogue of the Kv7.2 D212G mutation of ModelDB 118986, whose activation
    half-point moves from -32.4 mV to -27.7 mV (kmtwt.mod against kmquad.mod), a
    depolarizing shift of +4.7 mV."""
    u = v - shift_n
    return 0.01 * (u + 50) / (1 - torch.exp(-(u + 50) / 10))


def betaN(v, shift_n=0.0):
    return 0.125 * torch.exp(-((v - shift_n) + 60) / 80)


def alphaM(v, shift=0.0):
    """HH.alphaM.  `shift` displaces the sodium activation rate along the
    voltage axis; a positive shift means more depolarization is needed to
    activate, which is the direction of the R859C change."""
    u = v - shift
    return 0.1 * (u + 35) / (1 - torch.exp(-(u + 35) / 10))


def betaM(v, shift=0.0, shift_beta=False):
    u = v - shift if shift_beta else v
    return 4.0 * torch.exp(-0.0556 * (u + 60))


def alphaH(v):
    return 0.07 * torch.exp(-0.05 * (v + 60))


def betaH(v):
    return 1 / (1 + torch.exp(-0.1 * (v + 30)))


def steady_state(v, shift=0.0, shift_beta=False, shift_n=0.0):
    """Gating variables at steady state, as in HH.__init__."""
    m = alphaM(v, shift) / (alphaM(v, shift) + betaM(v, shift, shift_beta))
    h = alphaH(v) / (alphaH(v) + betaH(v))
    n = alphaN(v, shift_n) / (alphaN(v, shift_n) + betaN(v, shift_n))
    return m, h, n


def step(v, m, h, n, gNa_p, gK_p, gl_p, dt, I, shift=0.0, shift_beta=False,
         shift_n=0.0):
    """One forward-Euler step, same order of operations as HH.forward."""
    gNa = gNa_p * h * m ** 3
    gK = gK_p * n ** 4
    INa = gNa * (v - E_NA)
    IK = gK * (v - E_K)
    Il = gl_p * (v - E_L)
    am, bm = alphaM(v, shift), betaM(v, shift, shift_beta)
    m2 = m + dt * (am * (1 - m) - bm * m)
    n2 = n + dt * (alphaN(v, shift_n) * (1 - n) - betaN(v, shift_n) * n)
    h2 = h + dt * (alphaH(v) * (1 - h) - betaH(v) * h)
    v2 = v + dt * ((1 / CM) * (I - (INa + IK + Il)))
    return v2, m2, h2, n2


# ---- spike counting, the original pipeline's rule ---------------------------
def spike_count_np(v_logged):
    """`v_logged` is (..., n_samples) already downsampled every LOG_EVERY steps.
    Counts complete above-threshold excursions -- the pipeline's own rule."""
    above = v_logged > SPIKE_THRESHOLD
    transitions = (above[..., 1:] != above[..., :-1]).sum(axis=-1)
    return transitions // 2


def spike_count_torch(above_prev, above_now, acc):
    """Incremental form of the same rule, for the streaming integrator."""
    return acc + (above_now != above_prev).to(acc.dtype)


# ---- trace-producing integrator (small runs; keeps the whole trace) ----------
def simulate_trace(gNa, gK, gl, I, steps, dt, shift=0.0, shift_beta=False,
                   v0=-60.0, device="cuda", log_every=1, dtype=torch.float32):
    """Returns (n_logged, B) voltages. Broadcasts gNa/gK/gl/I to a common batch."""
    gNa, gK, gl, I = [torch.as_tensor(x, dtype=dtype, device=device)
                      for x in (gNa, gK, gl, I)]
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape, I.shape)
    gNa, gK, gl, I = [x.expand(B).contiguous() for x in (gNa, gK, gl, I)]
    v = torch.full(B, float(v0), dtype=dtype, device=device)
    m, h, n = steady_state(v, shift, shift_beta)
    out = torch.empty((steps // log_every, *B), dtype=dtype, device=device)
    with torch.no_grad():
        for i in range(steps):
            v, m, h, n = step(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta)
            if (i + 1) % log_every == 0:
                out[i // log_every] = v
    return out


# ---- streaming integrator (large sweeps; keeps only summary statistics) ------
def _block(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta, k):
    for _ in range(k):
        v, m, h, n = step(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta)
    return v, m, h, n


_compiled = {}


def get_block(k, fused=True):
    if not fused:
        return _block
    if k not in _compiled:
        _compiled[k] = torch.compile(_block, dynamic=False)
    return _compiled[k]


def simulate_stats(gNa, gK, gl, I, steps=STEPS_EXT, dt=DT_EXT, shift=0.0,
                   shift_beta=False, v0=-60.0, device="cuda", log_every=LOG_EVERY,
                   fused=True, dtype=torch.float32):
    """Integrate without storing the trace. Returns a dict of summary statistics
    per batch element: spike count (the pipeline's rule, on the downsampled
    signal), mean voltage, and a finiteness flag.

    One fused block == `log_every` integration steps == one logged sample, so the
    sampling grid is exactly the one neuron.hoc:40-52 writes.
    """
    gNa, gK, gl, I = [torch.as_tensor(x, dtype=dtype, device=device)
                      for x in (gNa, gK, gl, I)]
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape, I.shape)
    gNa, gK, gl, I = [x.expand(B).contiguous() for x in (gNa, gK, gl, I)]
    v = torch.full(B, float(v0), dtype=dtype, device=device)
    m, h, n = steady_state(v, shift, shift_beta)

    fn = get_block(log_every, fused)          # one call == one logged sample
    trans = torch.zeros(B, dtype=torch.int32, device=device)
    vsum = torch.zeros(B, dtype=torch.float32, device=device)
    above = v > SPIKE_THRESHOLD
    nsamp = steps // log_every
    with torch.no_grad():
        for _ in range(nsamp):
            v, m, h, n = fn(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta, log_every)
            a = v > SPIKE_THRESHOLD
            trans += (a != above).to(torch.int32)
            above = a
            vsum += torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    finite = torch.isfinite(v)
    return {"spikes": (trans // 2).to(torch.int32),
            "v_mean": vsum / nsamp,
            "finite": finite}


# ---- the pipeline's similarity scale -----------------------------------------
def similarity(distance, untreated_distance):
    """The pipeline's similarity percentage. 100% = matches the reference
    exactly; 0% = exactly as far off as the untreated variant; negative = worse."""
    return (1.0 - np.asarray(distance, dtype=float)
            / np.asarray(untreated_distance, dtype=float)) * 100.0


def gap_closed(loss, loss_untreated):
    """Percentage of the untreated variant's loss removed by a fit."""
    return (1.0 - np.asarray(loss, dtype=float)
            / np.asarray(loss_untreated, dtype=float)) * 100.0


# ---- fast streaming statistics for the large topology sweeps -----------------
# simulate_stats above makes one fused call per logged sample (every 10 steps), so a
# 300 ms trajectory costs 3,000 sequential calls and at small batches the launch
# overhead dominates.  The version below does `steps_per_call` integration steps per
# fused call and accumulates the threshold crossings INSIDE the call, every
# `log_every` steps, so the sampling grid is unchanged but there are far fewer calls.

_fast = {}


def _stats_block(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta,
                 above, trans, nsteps, log_every, shift_n=0.0):
    for j in range(nsteps):
        v, m, h, n = step(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta, shift_n)
        if (j + 1) % log_every == 0:
            a = v > SPIKE_THRESHOLD
            trans = trans + (a != above).to(trans.dtype)
            above = a
    return v, m, h, n, above, trans


def get_stats_block(nsteps, log_every, fused=True):
    key = (nsteps, log_every)
    if not fused:
        return _stats_block
    if key not in _fast:
        _fast[key] = torch.compile(_stats_block, dynamic=False)
    return _fast[key]


def simulate_stats_fast(gNa, gK, gl, I, steps=STEPS_EXT, dt=DT_EXT, shift=0.0,
                        shift_beta=False, v0=-60.0, device="cuda",
                        log_every=LOG_EVERY, steps_per_call=100, fused=True,
                        dtype=torch.float32, shift_n=0.0):
    """Same result as simulate_stats, far fewer kernel launches."""
    assert steps % steps_per_call == 0 and steps_per_call % log_every == 0
    gNa, gK, gl, I = [torch.as_tensor(x, dtype=dtype, device=device)
                      for x in (gNa, gK, gl, I)]
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape, I.shape)
    gNa, gK, gl, I = [x.expand(B).contiguous() for x in (gNa, gK, gl, I)]
    v = torch.full(B, float(v0), dtype=dtype, device=device)
    m, h, n = steady_state(v, shift, shift_beta, shift_n)
    above = v > SPIKE_THRESHOLD
    trans = torch.zeros(B, dtype=torch.int32, device=device)
    fn = get_stats_block(steps_per_call, log_every, fused)
    with torch.no_grad():
        for _ in range(steps // steps_per_call):
            v, m, h, n, above, trans = fn(v, m, h, n, gNa, gK, gl, dt, I, shift,
                                          shift_beta, above, trans,
                                          steps_per_call, log_every, shift_n)
    return {"spikes": (trans // 2).to(torch.int32), "finite": torch.isfinite(v)}
