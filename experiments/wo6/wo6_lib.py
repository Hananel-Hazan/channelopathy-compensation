"""Differentiable objectives for the compensating-parameter search.

Gradient descent on the voltage mean squared error does not reliably locate the
compensating configuration.  This module supplies alternative, still differentiable,
objectives to compare against it.

The hypothesis being tested: voltage mean squared error is dominated by action-potential
ALIGNMENT.  A spike displaced by a millisecond is penalised about as heavily as a spike
that never happened, so the loss surface is rugged in a way that reflects timing rather
than firing behaviour.  Every objective below is an attempt to remove that cliff while
staying differentiable.

The integrator is `wo5_lib.step`, which is bit-identical to the forward-Euler update
of `HH.forward` in mimic_cell_activity/HH.py when both kinetic shifts are zero.

Shapes.  A batch is P problems x C injected-current levels, flattened to P*C.  Every
objective takes a trajectory of shape (T, P, C) and a wild-type target of shape
(T, 1, C), and returns one loss per problem, shape (P,).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W                                          # noqa: E402

torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.accumulated_recompile_limit = 2048

THR = W.SPIKE_THRESHOLD          # -10 mV, the spike-detection threshold of the NEURON pipeline
LOG_EVERY = W.LOG_EVERY          # 10 -- neuron.hoc logs every 10th integration step


# --------------------------------------------------------------- integrator
_blocks = {}


def _traj_block(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_n, k):
    """k integration steps, returning only the final voltage -- one logged sample."""
    for _ in range(k):
        v, m, h, n = W.step(v, m, h, n, gNa, gK, gl, dt, I, shift, False, shift_n)
    return v, m, h, n


def _get_block(fused=True):
    if not fused:
        return _traj_block
    if "f" not in _blocks:
        _blocks["f"] = torch.compile(_traj_block, dynamic=False)
    return _blocks["f"]


def traj(gNa, gK, gl, I, steps=W.STEPS_EXT, dt=W.DT_EXT, shift=0.0, shift_n=0.0,
         v0=-60.0, log_every=LOG_EVERY, grad=True, fused=True):
    """Downsampled voltage trajectory, autograd live. Returns (steps//log_every, B).

    Only every `log_every`-th sample is kept, which is both what the NEURON pipeline
    sees and what keeps the stored activation graph small enough to differentiate.
    """
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape)
    v = torch.full(B, float(v0), dtype=torch.float32, device=gNa.device)
    m, h, n = W.steady_state(v, shift, False, shift_n)
    fn = _get_block(fused)
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(steps // log_every):
            v, m, h, n = fn(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_n, log_every)
            out.append(v)
    return torch.stack(out)


# --------------------------------------------------------------- helpers
def _smooth(x, sigma):
    """Gaussian smoothing along time. x is (T, P, C)."""
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    T, P, C = x.shape
    y = x.permute(1, 2, 0).reshape(P * C, 1, T)
    y = F.conv1d(F.pad(y, (r, r), mode="replicate"), k)
    return y.reshape(P, C, T).permute(2, 0, 1)


def soft_spikes(v, k=0.5, thr=THR):
    """Smooth surrogate for the pipeline's spike count.

    The NEURON pipeline counts complete above-threshold excursions.  Here the hard indicator is replaced by
    sigmoid(k * (v - threshold)); the sum of that signal's POSITIVE increments equals
    the number of times it rises from 0 to 1, which is the number of spikes, and it is
    differentiable everywhere.  Returns (T-1, P, C) rising-edge density and (P, C) count.
    """
    s = torch.sigmoid(k * (v - thr))
    rise = torch.relu(s[1:] - s[:-1])
    return rise, rise.sum(dim=0)


def _exp_filter(x, tau):
    """Causal exponential filter along time, for the van Rossum distance."""
    L = max(3, int(6 * tau))
    t = torch.arange(L, dtype=x.dtype, device=x.device)
    k = torch.exp(-t / tau)
    k = (k / k.sum()).flip(0).view(1, 1, -1)
    T, P, C = x.shape
    y = x.permute(1, 2, 0).reshape(P * C, 1, T)
    y = F.conv1d(F.pad(y, (L - 1, 0)), k)
    return y.reshape(P, C, T).permute(2, 0, 1)


def soft_dtw(a, b, gamma=1.0):
    """Soft dynamic time warping (Cuturi and Blondel 2017), vectorised over the batch.

    Dynamic time warping itself is not differentiable, but this relaxation of it is: the
    minimum in the recursion is replaced by a soft minimum with temperature gamma, which
    has a gradient everywhere and converges to true dynamic time warping as gamma -> 0.

    a is (Ta, N), b is (Tb, N); returns (N,).  The recursion is evaluated one
    anti-diagonal at a time, on which all cells are independent, so the only Python-level
    loop is over the Ta+Tb-1 diagonals rather than over the Ta*Tb cells.
    """
    Ta, N = a.shape
    Tb = b.shape[0]
    dev, dt = a.device, a.dtype
    D = (a.unsqueeze(1) - b.unsqueeze(0)) ** 2                  # (Ta, Tb, N)
    BIG = torch.tensor(1e10, device=dev, dtype=dt)

    def lo(d):
        return max(0, d - Tb + 1)

    prev2 = torch.empty(0, N, device=dev, dtype=dt)             # diagonal d-2
    prev1 = torch.empty(0, N, device=dev, dtype=dt)             # diagonal d-1

    for d in range(Ta + Tb - 1):
        i0, i1 = lo(d), min(d, Ta - 1)
        idx = torch.arange(i0, i1 + 1, device=dev)
        jdx = d - idx
        cost = D[idx, jdx]                                      # (L, N)

        def take(src, want_i, dprev):
            """Value of cell (want_i, dprev-want_i) from the stored diagonal `src`."""
            if src.numel() == 0:
                return BIG.expand(want_i.shape[0], N)
            off = want_i - lo(dprev)
            ok = (want_i >= 0) & (dprev - want_i >= 0) & (off >= 0) & (off < src.shape[0])
            g = src[off.clamp(0, max(src.shape[0] - 1, 0))]
            return torch.where(ok.unsqueeze(1), g, BIG)

        if d == 0:
            out = cost
        else:
            up   = take(prev1, idx - 1, d - 1)                  # (i-1, j)
            left = take(prev1, idx,     d - 1)                  # (i, j-1)
            diag = take(prev2, idx - 1, d - 2)                  # (i-1, j-1)
            stack = torch.stack([up, left, diag])
            out = cost - gamma * torch.logsumexp(-stack / gamma, dim=0)
        prev2, prev1 = prev1, out
    return prev1[0]


# --------------------------------------------------------------- objectives
# Each returns a per-problem loss, shape (P,).

def obj_voltage_mse(v, tgt, **kw):
    """Voltage mean squared error, the baseline objective."""
    return ((v - tgt) ** 2).mean(dim=(0, 2))


def obj_lowpass_mse(v, tgt, sigma=20.0, **kw):
    """Voltage mean squared error after smoothing both traces, which should blunt the
    alignment cliff directly. `sigma` is in logged samples (one sample = 0.1 ms)."""
    return ((_smooth(v, sigma) - _smooth(tgt, sigma)) ** 2).mean(dim=(0, 2))


def obj_rate_curve(v, tgt, k=0.5, **kw):
    """Mean squared error between the soft spike-count-versus-current curves. This is
    the differentiable analogue of the spike-count criterion the NEURON pipeline
    selects on."""
    _, c = soft_spikes(v, k)
    _, ct = soft_spikes(tgt, k)
    return ((c - ct) ** 2).mean(dim=1)


def obj_van_rossum(v, tgt, tau=20.0, k=0.5, **kw):
    """Van Rossum-type spike-train distance: convolve each soft spike train with an
    exponential kernel and take the squared difference, so that a displaced spike costs
    a little rather than as much as a missing spike."""
    r, _ = soft_spikes(v, k)
    rt, _ = soft_spikes(tgt, k)
    return ((_exp_filter(r, tau) - _exp_filter(rt, tau)) ** 2).mean(dim=(0, 2))


def obj_softdtw(v, tgt, gamma=1.0, stride=20, **kw):
    """Soft dynamic time warping on the coarsened traces (the recursion is quadratic in
    time, so the traces are strided down first)."""
    T, P, C = v.shape
    a = v[::stride].permute(1, 2, 0).reshape(P * C, -1).T
    b = tgt[::stride].expand(-1, P, -1).permute(1, 2, 0).reshape(P * C, -1).T
    return soft_dtw(a, b, gamma).reshape(P, C).mean(dim=1)


def obj_rate_plus_voltage(v, tgt, w=1.0, k=0.5, **kw):
    """Weighted combination of the firing-rate-curve and voltage objectives."""
    return obj_rate_curve(v, tgt, k=k) + w * obj_voltage_mse(v, tgt)


OBJECTIVES = {
    "voltage MSE":       (obj_voltage_mse, {}),
    "low-pass voltage MSE, sigma 2 ms":     (obj_lowpass_mse, {"sigma": 20.0}),
    "low-pass voltage MSE, sigma 10 ms":    (obj_lowpass_mse, {"sigma": 100.0}),
    "firing-rate curve MSE (soft count)":   (obj_rate_curve, {"k": 0.5}),
    "van Rossum distance, tau 2 ms":        (obj_van_rossum, {"tau": 20.0}),
    "rate curve + voltage MSE":             (obj_rate_plus_voltage, {"w": 1.0}),
    "soft dynamic time warping, gamma 1":   (obj_softdtw, {"gamma": 1.0, "stride": 30}),
    "soft dynamic time warping, gamma 10":  (obj_softdtw, {"gamma": 10.0, "stride": 30}),
}
