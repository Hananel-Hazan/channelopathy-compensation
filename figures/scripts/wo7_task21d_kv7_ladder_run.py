"""Record the Kv7.2 D212G somatic voltage traces across the stimulus ladder.

Runs the ModelDB 118986 CA1 pyramidal-cell model (through
experiments/wo6/phase2/wo6b_lib.py) at every injected-current level of the
conductance search, for three arms: the reference cell, the untreated variant,
and the variant compensated with the best prescription from
experiments/wo6/phase2/wo6b_taskD2_search.json.  The spike counts recorded here
are compared with those stored by the search.

Writes figures/scripts/wo7_kv7_ladder_traces.npz (keys t_<arm>_<i>, v_<arm>_<i>,
float32, ms and mV) and wo7_kv7_ladder_traces.json (counts and protocol).  The
figure is drawn by wo7_task21d_kv7_ladder_plot.py.  Requires NEURON; runs on the
CPU.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P2 = os.path.join(REPO, "experiments", "wo6", "phase2")
sys.path.insert(0, P2)
import wo6b_lib as L                                    # noqa: E402
from neuron import h                                    # noqa: E402

OUT = os.path.join(REPO, "figures", "scripts", "wo7_kv7_ladder_traces.npz")
META = os.path.join(REPO, "figures", "scripts", "wo7_kv7_ladder_traces.json")


def trace(amp_nA):
    """One run at `amp_nA`; returns somatic (t, v) and the number of upward
    crossings of L.SPIKE_THRESHOLD during the current step."""
    h.stim.amp = float(amp_nA)
    tv, vv = h.Vector(), h.Vector()
    tv.record(h._ref_t)
    vv.record(L._secs["soma0"](0.5)._ref_v)
    h.run()
    t, v = np.asarray(tv), np.asarray(vv)
    a = v > L.SPIKE_THRESHOLD
    idx = np.where((~a[:-1]) & a[1:])[0]
    st = t[idx + 1]
    on = (st >= h.stim.delay) & (st <= h.stim.delay + h.stim.dur)
    return t, v, int(on.sum())


def main():
    s = json.load(open(os.path.join(P2, "wo6b_taskD2_search.json")))
    currents = list(s["currents_nA"])
    ok = [c for c in s["candidates"] if not c.get("rejected")
          and c.get("similarity") is not None]
    best = max(ok, key=lambda c: c["similarity"])

    L.load()
    print(f"NEURON {h.nrnversion(0)} | ladder {currents} nA")
    print(f"stimulus {h.stim.delay}-{h.stim.delay + h.stim.dur} ms, "
          f"tstop {h.tstop} ms, {h.celsius} C")
    print(f"compensated prescription: {best['cond']}\n")

    arms = [
        ("reference",   False, {}),
        ("variant",     True,  {}),
        ("compensated", True,  best["cond"]),
    ]

    store, counts, t0 = {}, {}, time.time()
    for name, is_mut, cond in arms:
        L.set_mutant(is_mut)
        L.set_conductances(**cond)
        cs = []
        for i, a in enumerate(currents):
            t, v, n = trace(a)
            store[f"t_{name}_{i}"] = t.astype(np.float32)
            store[f"v_{name}_{i}"] = v.astype(np.float32)
            cs.append(n)
            print(f"   {name:<12} {a:>5} nA -> {n:>3} action potentials "
                  f"({len(t):,} samples)", flush=True)
        counts[name] = cs
        print(f"   {name:<12} total {sum(cs)}\n", flush=True)

    np.savez_compressed(OUT, **store)
    with open(META, "w") as f:
        json.dump({"currents_nA": currents, "counts": counts,
                   "stim_delay_ms": float(h.stim.delay),
                   "stim_dur_ms": float(h.stim.dur),
                   "tstop_ms": float(h.tstop), "celsius": float(h.celsius),
                   "compensated_cond": best["cond"],
                   "compensated_similarity_pct": best["similarity"],
                   "d_untreated": s["d_untreated"],
                   "seconds": round(time.time() - t0, 1)}, f,
                  indent=2)
    print(f"wrote {OUT}\nwrote {META}   ({time.time() - t0:.0f} s)")

    # the counts recorded here should match those stored by the search
    ref_ok = counts["reference"] == list(s["wild_type_counts"])
    var_ok = counts["variant"] == list(s["mutant_counts"])
    cmp_ok = counts["compensated"] == list(best["counts"])
    print(f"\ncross-check against the stored search: reference {ref_ok}, "
          f"variant {var_ok}, compensated {cmp_ok}")
    if not (ref_ok and var_ok and cmp_ok):
        print("   stored reference   :", s["wild_type_counts"])
        print("   recorded reference :", counts["reference"])
        print("   stored variant     :", s["mutant_counts"])
        print("   recorded variant   :", counts["variant"])
        print("   stored compensated :", best["counts"])
        print("   recorded compensated:", counts["compensated"])


if __name__ == "__main__":
    main()
