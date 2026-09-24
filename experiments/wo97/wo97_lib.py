"""Evaluate a conductance triple on the NEURON R859C model (ModelDB 87585).

Two protocols are provided:

  "exploration"  the protocol of the original R859C search.  Loads
                 Neuron_Test/R859C/neuron.hoc, which builds 35 cells (one per
                 injected current level) and advances them together.  That file does
                 NOT set `secondorder`, so NEURON's default fully implicit solver is
                 used, and the wild-type reference ladder is SIMULATED in the same
                 process rather than read from the archived text file.

  "harness"      a Python-built section per level, h.secondorder = 2,
                 h.continuerun(300).

The spike detector is transcribed from the original R859C search pipeline.

Requires the compiled working directory produced by build_work.py.
"""
import os

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__)) + "/work"
CURRENTS = list(range(20, 361, 10))          # 35 levels, 20-360 pA in 10 pA steps


def detector(V):
    """The pipeline's spike detector: clip to +/-10 mV, count transitions in
    both directions, take half, round down.  A spike still in progress at the end
    of the sweep therefore rounds down."""
    v = np.atleast_2d(V).astype(float).copy()
    v[v < -10] = -10
    v[v > -10] = 10
    return (np.diff(v, axis=1) != 0).sum(axis=1) // 2


def start():
    """Load NEURON with the R859C cell file from the working directory.  Returns the
    h object."""
    os.chdir(WORK)
    from neuron import h
    h.load_file("stdrun.hoc")
    h.load_file("neuron.hoc")
    return h


# ---------------------------------------------------------------- exploration --
def exploration_wt(h):
    """The wild-type reference ladder, simulated exactly as the exploration did.

    `secondorder` is set explicitly to 0, the value neuron.hoc leaves in
    place, so that this never depends on what ran before it in the same process.
    It is a NEURON global: a harness_ladder() call earlier in the process would
    otherwise leave it at 2 and silently change the solver used here."""
    h.secondorder = 0
    h.initWT()
    h.runWTsim()
    V = np.array([h.data_vecs_WT[i].as_numpy() for i in range(len(h.data_vecs_WT))])
    return detector(V), V


def exploration_mt(h, triple=None):
    """The mutant ladder.  `triple` is (g_leak, g_K, g_Na) applied to every level,
    exactly as the original search does; None leaves the file's own
    baseline (0.0005, 0.06, 0.2) in place.  `secondorder` is set explicitly to 0
    for the reason given in exploration_wt()."""
    h.secondorder = 0
    h.initMT()
    if triple is not None:
        x, y, z = triple
        for c in range(len(h.data_vecs_Mut)):
            h.MuTcell[c].isoma.gl_ichanR859C1 = x
            h.MuTcell[c].isoma.gkfbar_ichanR859C1 = y
            h.MuTcell[c].isoma.gnatbar_ichanR859C1 = z
    h.runMuTsim()
    V = np.array([h.data_vecs_Mut[i].as_numpy() for i in range(len(h.data_vecs_Mut))])
    return detector(V), V


# ------------------------------------------------------------------- harness --
def harness_ladder(h, mech, triple=None, secondorder=2):
    """One Python-built section per level, h.secondorder = 2, h.continuerun(300),
    sampled every 10th step."""
    g_leak, g_K, g_Na = (0.0005, 0.06, 0.2) if triple is None else triple
    out = []
    for I in CURRENTS:
        s = h.Section()
        s.nseg, s.L, s.diam, s.Ra, s.cm = 1, 25, 25, 210, 1
        s.insert(mech)
        for seg in s:
            setattr(seg, "gnatbar_" + mech, g_Na)
            setattr(seg, "gkfbar_" + mech, g_K)
            setattr(seg, "gl_" + mech, g_leak)
            setattr(seg, "el_" + mech, -60)
        s.enat, s.ekf = 50, -80
        ic = h.IClamp(s(0.5)); ic.delay, ic.dur, ic.amp = 50, 200, I / 1000.0
        vec = h.Vector().record(s(0.5)._ref_v)
        h.secondorder = secondorder; h.dt = 0.01
        h.finitialize(-60); h.continuerun(300)
        out.append(np.array(vec)[::10])
    h.secondorder = 0                    # restore: it is a global, see exploration_wt()
    V = np.array(out)
    return detector(V), V


# -------------------------------------------------------------------- archive --
def archived(fn):
    """The stored ladder text files in Neuron_Test/R859C/, counted with the same
    detector."""
    REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
    with open(os.path.join(ARCH, fn)) as f:
        f.readline(); f.readline()
        rows = [[float(p) for p in l.rstrip("\n").split("\t") if p != ""]
                for l in f if l.strip()]
    return detector(np.array(rows)[:, 1:].T)
