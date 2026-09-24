"""Load the `HH` class from mimic_cell_activity/HH.py without executing that module.

The module cannot simply be imported because it runs a 5,000-epoch fit at module
level.  Instead the class body (from `class HH()` up to `def plot(`) is read as text
and executed in a private namespace.

The class is used unchanged: the rate equations, the forward Euler step, the
constants and the steady-state gating initialisation all come from HH.py.  Only
`dt` and the injected current are set by the caller.
"""
import os
import re

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HH_PY = os.path.join(REPO, "mimic_cell_activity", "HH.py")


def load_hh_class(path=HH_PY):
    """Return (HH class, 1-based source line range of the class in `path`)."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("class HH()"))
    end = next(i for i, l in enumerate(lines) if l.startswith("def plot("))
    body = "".join(lines[start:end])
    ns = {"torch": torch}
    exec(compile(body, path, "exec"), ns)
    return ns["HH"], (start + 1, end)


def simulate(g_Na, g_K, g_l, dt, n_steps, current, v0=-60.0):
    """Run the HH model forward for `n_steps` Euler steps of `dt` ms with a constant
    injected current and return (t, v, m, h, n) as numpy arrays."""
    HH, _ = load_hh_class()
    with torch.no_grad():
        hh = HH(v=v0, g_Na=g_Na, g_K=g_K, g_l=g_l, dt=dt)
        # the injected current is a class constant (const_I = 7 in HH.py);
        # it is overridden here so the value used is explicit
        hh.const_I = torch.tensor(float(current), dtype=torch.float)
        v = torch.zeros(n_steps)
        m = torch.zeros(n_steps)
        h = torch.zeros(n_steps)
        n = torch.zeros(n_steps)
        for i in range(n_steps):
            hh.forward()
            v[i], m[i], h[i], n[i] = hh.v, hh.m, hh.h, hh.n
    import numpy as np
    t = np.arange(n_steps) * dt        # time in ms, spaced by dt (the plot helper in HH.py uses a fixed 0-10 linspace instead)
    return t, v.numpy(), m.numpy(), h.numpy(), n.numpy()
