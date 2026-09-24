"""NEURON Kv7.2 case study: one CA1 pyramidal cell, driven and scored.

The cell, its geometry, its mechanisms and its stimulus all come from ModelDB entry
118986 (`mutant/fig6a.hoc`), which reproduces figure 6a of Miceli et al. (2009).  The
mutation is the one that file implements: the whole M-current is carried by `kmtwt` in
the wild type and by `kmtquad` in the mutant, with the total conductance unchanged, so
the variant is a gating change at identical maximal conductance.

What this module adds is the ability to set the three pharmacologically accessible
conductances to arbitrary values and to score the cell the way the R859C pipeline scores
it: action-potential count at each of several injected-current levels, summed absolute
difference from the wild type, expressed on the pipeline's percentage scale.

**The conductance assignment mirrors `fig6a.hoc` line for line**, including the axonal
multiplier and the distance-dependent A-type gradient in the apical tree, and
`validate()` checks that at the file's own default values it reproduces the file's own
result exactly.
"""
import os, time
import numpy as np
from neuron import h

MUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutant")

# fig6a.hoc's own values, lines 11-26
DEF = dict(gna=0.045, gkdr=0.02, ka=0.03, ghd=0.00001, gkm=0.006,
           gcat=0.0001, gahp=0.00001)
CM, RM_NUM, RA_ALL, RA_AX, AXONM, VREST = 0.75, 28000.0, 500.0, 50.0, 3.0, -65.0
G_PAS_DEF = 1.0 / (RM_NUM / CM)
SPIKE_THRESHOLD = 0.0        # mV, upward crossing; the trace overshoots to about +39 mV

_loaded = False
_secs = {}


def load():
    """Load the published model once per process."""
    global _loaded
    if _loaded:
        return
    # `fig6a.hoc` loads its geometry and helper files by relative path, so the working
    # directory has to be the model's while it loads -- but `h.chdir` moves the whole
    # process, and every result file written afterwards would land inside the downloaded
    # model package.  Restore it once the model is in memory; the mechanisms do not need
    # it after that.
    cwd = os.getcwd()
    h.nrn_load_dll(os.path.join(MUT, "x86_64", "libnrnmech.so"))
    h.chdir(MUT)
    h.load_file("fig6a.hoc")
    h.chdir(cwd)
    os.chdir(cwd)
    for group in ("axon", "soma", "dendrite", "apical_dendrite", "user5"):
        _secs[group] = [s for s in h.allsec() if s.name().split("[")[0] == group]
    _secs["soma0"] = [s for s in h.allsec() if s.name() == "soma[0]"][0]
    _loaded = True


def set_conductances(gna=DEF["gna"], gkdr=DEF["gkdr"], g_pas=G_PAS_DEF, ka=DEF["ka"]):
    """Re-apply fig6a.hoc's insert block with these values.

    Sodium and delayed-rectifier potassium are the two gated conductances the pipeline
    optimizes; the leak is the third.  The A-type conductance is exposed because the
    published file makes it distance-dependent and a search over it would otherwise be
    silently disabled, but it is held at its default unless asked for.
    """
    for s in _secs["axon"]:
        s.gbar_nax = gna * AXONM
        s.gkdrbar_kdr = gkdr * AXONM
        s.gkabar_kap = ka
        s.g_pas = g_pas
    for s in _secs["soma"]:
        s.gbar_na3 = gna * AXONM
        s.gkdrbar_kdr = gkdr * AXONM
        s.gkabar_kap = ka
        s.g_pas = g_pas
    for s in _secs["dendrite"]:
        s.gbar_na3 = 0.0                       # fig6a.hoc:  basal dendrites carry none
        s.g_pas = g_pas
    for group in ("apical_dendrite", "user5"):
        for s in _secs[group]:
            s.gbar_na3 = gna
            s.gkdrbar_kdr = gkdr
            s.g_pas = g_pas
            for seg in s:
                xdist = h.distance(seg.x, sec=s)
                if xdist > 100:
                    seg.kad.gkabar = ka * (1 + xdist / 100)
                    seg.kap.gkabar = 0.0
                else:
                    seg.kap.gkabar = ka * (1 + xdist / 100)
                    seg.kad.gkabar = 0.0


def set_mutant(is_mutant):
    """fig6a.hoc's own switch: `flag` selects which mechanism carries the M-current.
    The hoc `init()` reads it, and `gkm`, on every run."""
    h.flag = 1 if is_mutant else 0


def spike_count(amp_nA, tstop=None):
    """Action potentials at the soma during one current step, counted by upward
    threshold crossing -- the pipeline's own criterion."""
    h.stim.amp = float(amp_nA)
    if tstop is not None:
        h.tstop = float(tstop)
    tv, vv = h.Vector(), h.Vector()
    tv.record(h._ref_t)
    vv.record(_secs["soma0"](0.5)._ref_v)
    h.run()
    t, v = np.asarray(tv), np.asarray(vv)
    if not np.all(np.isfinite(v)):
        return -1                              # diverged; caller rejects the candidate
    a = v > SPIKE_THRESHOLD
    idx = np.where((~a[:-1]) & a[1:])[0]
    st = t[idx + 1]
    on = (st >= h.stim.delay) & (st <= h.stim.delay + h.stim.dur)
    return int(on.sum())


def curve(currents, is_mutant, **cond):
    """Spike count at every injected-current level, for one configuration."""
    set_mutant(is_mutant)
    set_conductances(**cond)
    return np.array([spike_count(a) for a in currents], dtype=int)


def spike_count_guarded(amp_nA, budget):
    """`spike_count` with a wall-clock limit.

    A far-corner conductance set can make the variable-step integrator stiff, and one
    such draw would otherwise stall a whole shard.  The run is advanced in 50 ms slices
    of model time and abandoned if it has not finished within `budget` seconds, which is
    reported as -1 so the caller rejects the candidate.  `h.run()` is init() followed by
    continuerun(tstop), so this is the same run, only interruptible.
    """
    h.stim.amp = float(amp_nA)
    tv, vv = h.Vector(), h.Vector()
    tv.record(h._ref_t)
    vv.record(_secs["soma0"](0.5)._ref_v)
    t0 = time.time()
    h.init()
    while h.t < h.tstop:
        h.continuerun(min(h.t + 50.0, float(h.tstop)))
        if time.time() - t0 > budget:
            return -1
    t, v = np.asarray(tv), np.asarray(vv)
    if not np.all(np.isfinite(v)):
        return -1
    a = v > SPIKE_THRESHOLD
    idx = np.where((~a[:-1]) & a[1:])[0]
    st = t[idx + 1]
    on = (st >= h.stim.delay) & (st <= h.stim.delay + h.stim.dur)
    return int(on.sum())


def curve_guarded(currents, is_mutant, budget=90.0, **cond):
    """`curve` with the wall-clock guard; stops at the first level that overruns."""
    set_mutant(is_mutant)
    set_conductances(**cond)
    out = []
    for a in currents:
        c = spike_count_guarded(a, budget)
        out.append(c)
        if c < 0:
            out.extend([-1] * (len(currents) - len(out)))
            break
    return np.array(out, dtype=int)


def similarity(distance, untreated_distance):
    """Similarity percentage of the search pipeline, as in `wo5_lib.similarity`.
    100 % = matches the wild type exactly; 0 % = as far off as the untreated mutant."""
    return (1.0 - np.asarray(distance, float) / float(untreated_distance)) * 100.0


def validate(verbose=True):
    """The published file's own result, through this module's parameter path.

    fig6a.hoc drives the cell at 0.47 nA for 400 ms and gives 1 action potential for the
    wild type and 8 for the mutant (`wo6b_taskD1_fig6a.json`, and the screenshot shipped
    with the model shows 8 mutant spikes).  If setting the conductances by hand does not
    give those two numbers, the parameter path is wrong and nothing below it is usable.
    """
    load()
    wt = curve([0.47], False)[0]
    mt = curve([0.47], True)[0]
    ok = (wt == 1) and (mt == 8)
    if verbose:
        print(f"validate: wild type {wt} (expect 1), mutant {mt} (expect 8) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, wt, mt
