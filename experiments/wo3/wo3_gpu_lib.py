"""Shared harness for the GPU overhead measurement.

The baseline is the HH class of mimic_cell_activity/HH.py, loaded unchanged by
`wo2_lib` (which reads that file and executes only the part above its driver
code).

`step()` below is a functional rewrite of `HH.forward` with the same operations
in the same order. It exists because a CUDA graph and torch.compile both need a
pure function of tensors; `HH.forward` mutates attributes in place. Equivalence
between the two is measured, not assumed -- see wo3_taskC2.py.

The benchmark unit is the same as in wo2_taskD_benchmark.py:
    one candidate parameter set = one conductance triple simulated across all
    35 injected-current levels for 300 ms at an integration step of 0.01 ms,
    i.e. 35 batch elements x 30,000 integration steps.
"""
import os
import sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import wo2_lib as L                                    # noqa: E402  (loads HH.py unchanged)

CUR_LEVELS = 35          # injected-current levels per candidate parameter set
STEPS_FULL = 30000       # 300 ms at 0.01 ms
DT = 0.01
I_CONST = 8.0

# reversal potentials (mV) and membrane capacitance, copied from HH.__init__
E_NA, E_K, E_L, CM = 50.0, -77.0, -54.0, 1.0


# ---------------------------------------------------------------- rate functions
# copied from the HH class's rate methods. torch.as_tensor on an existing
# float32 CUDA tensor is an identity, so it is dropped here; wo3_taskC2.py checks
# that this makes no numerical difference.

def _alphaN(v):
    return 0.01 * (v + 50) / (1 - torch.exp(-(v + 50) / 10))


def _betaN(v):
    return 0.125 * torch.exp(-(v + 60) / 80)


def _alphaM(v):
    return 0.1 * (v + 35) / (1 - torch.exp(-(v + 35) / 10))


def _betaM(v):
    return 4.0 * torch.exp(-0.0556 * (v + 60))


def _alphaH(v):
    return 0.07 * torch.exp(-0.05 * (v + 60))


def _betaH(v):
    return 1 / (1 + torch.exp(-0.1 * (v + 30)))


def step(v, m, h, n, gNa_p, gK_p, gl_p, dt, I):
    """One forward-Euler integration step, in the same order as HH.forward."""
    gNa = gNa_p * h * m ** 3
    gK = gK_p * n ** 4
    INa = gNa * (v - E_NA)
    IK = gK * (v - E_K)
    Il = gl_p * (v - E_L)

    m2 = m + dt * ((_alphaM(v) * (1 - m)) - _betaM(v) * m)
    n2 = n + dt * ((_alphaN(v) * (1 - n)) - _betaN(v) * n)
    h2 = h + dt * ((_alphaH(v) * (1 - h)) - _betaH(v) * h)

    v2 = v + dt * ((1 / CM) * (I - (INa + IK + Il)))
    return v2, m2, h2, n2


def chunk(v, m, h, n, gNa_p, gK_p, gl_p, dt, I, k):
    """k integration steps, unrolled. This is the body that gets fused."""
    for _ in range(k):
        v, m, h, n = step(v, m, h, n, gNa_p, gK_p, gl_p, dt, I)
    return v, m, h, n


# ---------------------------------------------------------------- initial state

def init_state(B, device, v0=-60.0, seed=0):
    """Initial (v, m, h, n) exactly as HH.__init__ builds them (gates at their
    steady state for v0), and a random batch of B conductance triples over the
    same ranges as wo2_taskD_benchmark.py."""
    rng = np.random.default_rng(seed)
    gNa = torch.tensor(rng.uniform(60, 260, B), dtype=torch.float, device=device)
    gK = torch.tensor(rng.uniform(26, 49, B), dtype=torch.float, device=device)
    gl = torch.tensor(rng.uniform(0.1, 0.5, B), dtype=torch.float, device=device)
    v = torch.full((B,), float(v0), dtype=torch.float, device=device)
    m = _alphaM(v) / (_alphaM(v) + _betaM(v))
    h = _alphaH(v) / (_alphaH(v) + _betaH(v))
    n = _alphaN(v) / (_alphaN(v) + _betaN(v))
    return v, m, h, n, gNa, gK, gl


def baseline_hh(B, device, seed=0, dt=DT, I=I_CONST, v0=-60.0):
    """The original HH object from HH.py, seeded identically to init_state."""
    rng = np.random.default_rng(seed)
    gNa = rng.uniform(60, 260, B)
    gK = rng.uniform(26, 49, B)
    gl = rng.uniform(0.1, 0.5, B)
    hh = L.build(gNa, gK, gl, dt, I, B=B, device=device, v0=v0)
    return hh
