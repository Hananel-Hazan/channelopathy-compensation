"""Gradient fitting of conductances on the differentiable Hodgkin-Huxley model.

The fit can (a) hold an arbitrary SUBSET of the three conductances fixed, and (b)
carry a fixed kinetic shift of the sodium activation.  The forward pass is
`wo5_lib.step`, which is bit-identical to `HH.forward` in the original model
(mimic_cell_activity/HH.py) when the shift is zero.

Settings: Adam, non-negativity clamp at 1e-6, integration step 10/1500 ms, 1500 steps,
injected current 8, per-conductance learning rates scaled to each conductance's own
magnitude -- sodium 0.5, potassium 0.15, leak 0.001.

Success measure:
    gap closed = (1 - best loss reached / loss of the untreated variant) * 100
using the BEST loss, not the final one, because the explicit integrator can diverge
after converging.
"""
import numpy as np
import torch

import wo5_lib as W

LRS = {"Na": 0.5, "K": 0.15, "l": 0.001}
CLAMP_MIN = 1e-6
ITERS = 400


BLOCK = 10          # integration steps per fused call
_blocks = {}

torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.accumulated_recompile_limit = 2048


def _loss_block(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta, tgt, sse, k):
    """`k` integration steps, accumulating the squared error against `tgt` (k, B)
    as it goes. The trace is never materialised, which keeps the compiled graph
    small -- returning the stacked trace instead makes the Inductor compiler run out
    of host memory."""
    for j in range(k):
        v, m, h, n = W.step(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta)
        d = v - tgt[j]
        sse = sse + d * d
    return v, m, h, n, sse


def _get_block(fused=True):
    if not fused:
        return _loss_block
    if "f" not in _blocks:
        _blocks["f"] = torch.compile(_loss_block, dynamic=False)
    return _blocks["f"]


def mse_roll(gNa, gK, gl, I, steps, dt, shift, target, shift_beta=False, v0=-60.0,
             grad=True, fused=True, block=BLOCK):
    """Mean squared error of each batch element's trajectory against `target`.

    `target` is (steps, B) or (steps, 1). The step loop is handed to torch.compile
    in blocks of `block` steps; at these batch sizes the eager loop is bound by
    kernel-launch overhead, which makes an unfused fit unusably slow. `fused=False`
    runs the same computation eagerly.
    """
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape)
    v = torch.full(B, float(v0), dtype=torch.float32, device=gNa.device)
    m, h, n = W.steady_state(v, shift, shift_beta)
    sse = torch.zeros(B, dtype=torch.float32, device=gNa.device)
    fn = _get_block(fused)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        done = 0
        while done < steps:
            k = min(block, steps - done)
            v, m, h, n, sse = fn(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta,
                                 target[done:done + k], sse, k)
            done += k
    return sse / steps


def roll(gNa, gK, gl, I, steps, dt, shift, shift_beta=False, v0=-60.0, grad=True):
    """Integrate keeping the whole trace (eager; used only for building targets)."""
    B = torch.broadcast_shapes(gNa.shape, gK.shape, gl.shape)
    v = torch.full(B, float(v0), dtype=torch.float32, device=gNa.device)
    m, h, n = W.steady_state(v, shift, shift_beta)
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(steps):
            v, m, h, n = W.step(v, m, h, n, gNa, gK, gl, dt, I, shift, shift_beta)
            out.append(v)
    return torch.stack(out)


def target_trace(params=W.WT, shift=0.0, I=W.I_R2, steps=W.STEPS_R2, dt=W.DT_R2,
                 device="cuda", B=1):
    g = [torch.full((B,), float(x), dtype=torch.float32, device=device) for x in params]
    return roll(g[0], g[1], g[2], I, steps, dt, shift, grad=False).detach()


def fit_batch(starts, free_mask, target, shift=0.0, I=W.I_R2, steps=W.STEPS_R2,
              dt=W.DT_R2, iters=ITERS, device="cuda", lrs=None,
              clamp_min=CLAMP_MIN, fused=True):
    """Optimize B independent problems at once.

    starts     : (B, 3) array of (Na, K, l) starting conductances
    free_mask  : (B, 3) boolean -- True where the parameter is optimized. Entries
                 that are False are restored to their starting value after every
                 update, which is what "held fixed" means here.
    target     : (steps, B) or (steps, 1) reference trace
    shift      : scalar, or (B,) array -- the kinetic variant, held fixed throughout

    Batch elements never interact in the forward pass, so summing the per-element
    losses and calling backward once gives every element its own gradient. This
    matters for speed: at these batch sizes the integration is launch-bound, so B
    problems cost almost exactly what 1 problem costs.

    Returns the best-loss iterate per element and the full trajectories.
    """
    lrs = lrs or LRS
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    free_mask = np.asarray(free_mask, dtype=bool).reshape(-1, 3)
    B = starts.shape[0]
    if np.isscalar(shift):
        sh = float(shift)
    else:
        sh = torch.tensor(np.asarray(shift, dtype=np.float64), dtype=torch.float32,
                          device=device)

    p = [torch.tensor(starts[:, j], dtype=torch.float32, device=device,
                      requires_grad=True) for j in range(3)]
    fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=device)
             for j in range(3)]
    keep = [torch.tensor(~free_mask[:, j], device=device) for j in range(3)]
    opt = torch.optim.Adam([{"params": [p[j]], "lr": lrs[W.NAMES[j]]}
                            for j in range(3)])

    hist_loss = np.empty((iters, B)); hist_loss[:] = np.nan
    hist_p = np.empty((iters, B, 3)); hist_p[:] = np.nan
    best_loss = np.full(B, np.inf)
    best_p = starts.copy()
    best_it = np.full(B, -1, dtype=int)
    would_go_negative = np.zeros(B, dtype=bool)

    for it in range(iters):
        per = mse_roll(p[0], p[1], p[2], I, steps, dt, sh, target, fused=fused)
        opt.zero_grad(set_to_none=True)
        torch.nan_to_num(per, nan=0.0, posinf=0.0, neginf=0.0).sum().backward()
        opt.step()
        with torch.no_grad():
            for j in range(3):
                would_go_negative |= (p[j] < 0).cpu().numpy()
                p[j].clamp_(min=clamp_min)
                p[j].copy_(torch.where(keep[j], fixed[j], p[j]))

        lv = per.detach().double().cpu().numpy()
        pv = np.stack([q.detach().double().cpu().numpy() for q in p], axis=1)
        hist_loss[it] = lv
        hist_p[it] = pv
        imp = np.isfinite(lv) & (lv < best_loss)
        best_loss[imp] = lv[imp]
        best_p[imp] = pv[imp]
        best_it[imp] = it

    return {"best_loss": best_loss, "best_params": best_p, "best_iter": best_it,
            "hist_loss": hist_loss, "hist_params": hist_p,
            "would_go_negative": would_go_negative}


def fit(start, free, target, **kw):
    """Single-problem convenience wrapper around fit_batch."""
    mask = np.array([[n in free for n in W.NAMES]])
    r = fit_batch(np.array([start]), mask, target, **kw)
    return {"best": {"loss": float(r["best_loss"][0]),
                     "params": tuple(r["best_params"][0]),
                     "iter": int(r["best_iter"][0])},
            "would_go_negative": bool(r["would_go_negative"][0]),
            "free": tuple(free), "start": tuple(start)}


def loss_of(params, target, shift=0.0, I=W.I_R2, steps=W.STEPS_R2, dt=W.DT_R2,
            device="cuda"):
    """Mean squared error of one parameter triple against a target trace."""
    g = [torch.full((1,), float(x), dtype=torch.float32, device=device) for x in params]
    with torch.no_grad():
        return float(mse_roll(g[0], g[1], g[2], I, steps, dt, shift,
                              target.reshape(steps, -1), grad=False)[0])


def losses_batched(P, target, shift=0.0, I=W.I_R2, steps=W.STEPS_R2, dt=W.DT_R2,
                   device="cuda", chunk=20000):
    """Mean squared error for many parameter triples. `P` is (N, 3).

    The target is (steps, 1) and broadcasts over the batch. Chunked so the stored
    trace never exceeds a few hundred megabytes.
    """
    P = np.asarray(P, dtype=np.float64)
    N = P.shape[0]
    out = np.empty(N, dtype=np.float64)
    tgt = target.reshape(steps, 1)
    # every chunk is padded to exactly `chunk` rows so that only ONE batch shape is
    # ever compiled; recompiling per shape exhausts the Inductor compiler's host
    # memory.
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        blk = np.empty((chunk, 3)); blk[:] = P[s0]
        blk[: e - s0] = P[s0:e]
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=device) for j in range(3)]
        with torch.no_grad():
            r = mse_roll(g[0], g[1], g[2], I, steps, dt, shift, tgt, grad=False)
        out[s0:e] = r[: e - s0].double().cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Multi-current fitting.
#
# The single-current fit above matches the wild-type voltage trace at one injected
# current.  A parameter set that matches the wild type at one current can be far off
# across the rest of the current-response curve.  The NEURON pipeline evaluates across
# all 35 injected-current levels (neuron.hoc), so the functions below fit and score
# across several levels.
#
# Fitting uses a SUBSET of the levels and evaluation uses all 35, which also makes the
# evaluation a genuine held-out test rather than a restatement of the objective.
# ---------------------------------------------------------------------------

FIT_CURRENTS = np.array([0.0, 3.0, 6.0, 7.0, 8.5, 11.0, 14.0, 17.0])


def target_traces_multi(currents, params=W.WT, shift=0.0, steps=W.STEPS_EXT,
                        dt=W.DT_EXT, device="cuda"):
    """(steps, C) wild-type traces, one column per injected-current level."""
    C = len(currents)
    g = [torch.full((C,), float(x), dtype=torch.float32, device=device) for x in params]
    I = torch.tensor(np.asarray(currents), dtype=torch.float32, device=device)
    return roll(g[0], g[1], g[2], I, steps, dt, shift, grad=False).detach()


def fit_batch_multi(starts, free_mask, target_C, currents, shift=0.0,
                    steps=W.STEPS_EXT, dt=W.DT_EXT, iters=200, device="cuda",
                    lrs=None, clamp_min=CLAMP_MIN, block=25):
    """As fit_batch, but each problem is scored against the wild type at every level
    in `currents` and the per-problem loss is the mean over levels.

    starts (P, 3); free_mask (P, 3); target_C (steps, C); currents (C,).
    shift is a scalar or a (P,) array.
    """
    lrs = lrs or LRS
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    free_mask = np.asarray(free_mask, dtype=bool).reshape(-1, 3)
    P = starts.shape[0]
    C = len(currents)
    I = torch.tensor(np.tile(np.asarray(currents), P), dtype=torch.float32, device=device)
    tgt = target_C.repeat(1, P)                                   # (steps, P*C)
    if np.isscalar(shift):
        sh = float(shift)
    else:
        sh = torch.tensor(np.repeat(np.asarray(shift, dtype=np.float64), C),
                          dtype=torch.float32, device=device)

    p = [torch.tensor(starts[:, j], dtype=torch.float32, device=device,
                      requires_grad=True) for j in range(3)]
    fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=device)
             for j in range(3)]
    keep = [torch.tensor(~free_mask[:, j], device=device) for j in range(3)]
    opt = torch.optim.Adam([{"params": [p[j]], "lr": lrs[W.NAMES[j]]} for j in range(3)])

    best_loss = np.full(P, np.inf)
    best_p = starts.copy()
    best_it = np.full(P, -1, dtype=int)
    would_go_negative = np.zeros(P, dtype=bool)

    for it in range(iters):
        g = [q.unsqueeze(1).expand(P, C).reshape(-1) for q in p]
        per = mse_roll(g[0], g[1], g[2], I, steps, dt, sh, tgt, block=block)
        per_prob = torch.nan_to_num(per, nan=1e12, posinf=1e12,
                                    neginf=1e12).view(P, C).mean(dim=1)
        opt.zero_grad(set_to_none=True)
        per_prob.sum().backward()
        opt.step()
        with torch.no_grad():
            for j in range(3):
                would_go_negative |= (p[j] < 0).cpu().numpy()
                p[j].clamp_(min=clamp_min)
                p[j].copy_(torch.where(keep[j], fixed[j], p[j]))

        lv = per_prob.detach().double().cpu().numpy()
        pv = np.stack([q.detach().double().cpu().numpy() for q in p], axis=1)
        imp = np.isfinite(lv) & (lv < best_loss)
        best_loss[imp] = lv[imp]
        best_p[imp] = pv[imp]
        best_it[imp] = it

    return {"best_loss": best_loss, "best_params": best_p, "best_iter": best_it,
            "would_go_negative": would_go_negative}


def losses_multi(P3, target_C, currents, shift=0.0, steps=W.STEPS_EXT, dt=W.DT_EXT,
                 device="cuda", chunk_sets=2048, block=25):
    """Multi-current mean squared error for many parameter triples, no gradients."""
    P3 = np.asarray(P3, dtype=np.float64).reshape(-1, 3)
    N, C = P3.shape[0], len(currents)
    out = np.empty(N)
    Ifull = torch.tensor(np.tile(np.asarray(currents), chunk_sets),
                         dtype=torch.float32, device=device)
    tgt = target_C.repeat(1, chunk_sets)
    for s0 in range(0, N, chunk_sets):
        e = min(s0 + chunk_sets, N)
        blk = np.empty((chunk_sets, 3)); blk[:] = P3[s0]
        blk[: e - s0] = P3[s0:e]
        g = [torch.tensor(np.repeat(blk[:, j], C), dtype=torch.float32, device=device)
             for j in range(3)]
        with torch.no_grad():
            r = mse_roll(g[0], g[1], g[2], Ifull, steps, dt, shift, tgt,
                         grad=False, block=block)
        out[s0:e] = torch.nan_to_num(r, nan=1e12, posinf=1e12).view(
            chunk_sets, C).mean(dim=1)[: e - s0].double().cpu().numpy()
    return out
