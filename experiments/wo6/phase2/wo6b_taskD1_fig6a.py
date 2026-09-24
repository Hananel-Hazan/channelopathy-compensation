"""Reproduce figure 6a of Miceli et al. (2009) with the published NEURON model.

The published figure is reproduced before anything is built on top of the model.
ModelDB entry 118986 ships `fig6a.hoc`, which the readme says
"reproduces the traces shown in Fig.6a of the paper": a CA1 pyramidal cell driven by a
0.47 nA somatic step for 400 ms, run twice -- once with the whole M-current carried by the
wild-type mechanism (`kmtwt`) and once by the mutant one (`kmtquad`, the Kv7.2 D212G
subunit).  The paper's claim is that the mutation raises firing frequency.

`fig6a.hoc` is loaded unmodified.  It builds a graph window, which does nothing without a
display; the model, the stimulus and the two-state `init()` are all taken from that file
rather than re-implemented, so this is the published protocol and not a paraphrase of it.
The only additions are a voltage recording and a spike count.

Writes wo6b_taskD1_fig6a.json and wo6b_taskD1_fig6a_traces.npz next to this script.
"""
import json, os, platform, sys
import numpy as np
from neuron import h

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MUT = os.path.join(REPO, "experiments", "wo6", "phase2", "mutant")
sys.path.insert(0, MUT)
h.nrn_load_dll(f"{MUT}/x86_64/libnrnmech.so")
h.chdir(MUT)
h.load_file("fig6a.hoc")

THRESH = 0.0          # mV; the trace overshoots well past this on every spike


def spike_times(t, v, thresh=THRESH):
    """Upward threshold crossings, which is how the pipeline counts action potentials."""
    a = np.asarray(v) > thresh
    idx = np.where((~a[:-1]) & a[1:])[0]
    return np.asarray(t)[idx + 1]


def soma_seg():
    """`geo9068802.hoc:1` declares `soma[2]`, and the hoc plot records `soma.v(rel)`,
    which in an arrayed declaration means `soma[0]`.  Fetch that section by name."""
    for sec in h.allsec():
        if sec.name() == "soma[0]":
            return sec(float(h.rel))
    raise RuntimeError("soma[0] not found")


def run_one(flag):
    h.flag = flag
    tv, vv = h.Vector(), h.Vector()
    tv.record(h._ref_t)
    vv.record(soma_seg()._ref_v)
    h.run()
    t, v = np.array(tv), np.array(vv)
    st = spike_times(t, v)
    # the current step is on from stim.del to stim.del + stim.dur
    on = (st >= h.stim.delay) & (st <= h.stim.delay + h.stim.dur)
    return t, v, st[on]


HOST = platform.node()
print(f"machine {HOST} | NEURON {h.nrnversion(5) if hasattr(h, 'nrnversion') else ''}")
print(f"stimulus {h.stim.amp} nA from {h.stim.delay} to {h.stim.delay + h.stim.dur} ms, "
      f"tstop {h.tstop} ms, celsius {h.celsius}, total M-current conductance {h.gkm}\n")

OUT = {"host": HOST, "stim_amp_nA": float(h.stim.amp), "stim_dur_ms": float(h.stim.dur),
       "tstop_ms": float(h.tstop), "gkm": float(h.gkm), "arms": {}}
traces = {}
for name, flag in (("wild type", 0), ("mutant (Kv7.2 D212G)", 1)):
    t, v, st = run_one(flag)
    dur_s = float(h.stim.dur) / 1000.0
    rate = len(st) / dur_s
    isi = np.diff(st)
    OUT["arms"][name] = {
        "spikes_during_step": int(len(st)),
        "firing_rate_Hz": float(rate),
        "first_spike_latency_ms": float(st[0] - h.stim.delay) if len(st) else None,
        "mean_isi_ms": float(isi.mean()) if isi.size else None,
        "last_isi_ms": float(isi[-1]) if isi.size else None,
        "resting_v_mV": float(v[0]),
        "peak_v_mV": float(v.max()),
        "spike_times_ms": [float(x) for x in st]}
    traces[name] = (t, v)
    print(f"{name:<24} {len(st):>3} action potentials during the 400 ms step "
          f"= {rate:6.2f} Hz   first at {st[0]-h.stim.delay:6.2f} ms after step onset")

wt = OUT["arms"]["wild type"]["spikes_during_step"]
mt = OUT["arms"]["mutant (Kv7.2 D212G)"]["spikes_during_step"]
OUT["mutant_minus_wildtype_spikes"] = mt - wt
OUT["mutant_over_wildtype_rate"] = (mt / wt) if wt else None
print(f"\nmutant fires {mt - wt:+d} action potentials relative to wild type "
      f"({mt/wt:.2f}x the rate)" if wt else "")
print("The paper's claim is that the mutation INCREASES firing frequency.")
print(f"-> reproduced: {mt > wt}")

np.savez_compressed(f"{MUT}/../wo6b_taskD1_fig6a_traces.npz",
                    t_wt=traces["wild type"][0], v_wt=traces["wild type"][1],
                    t_mt=traces["mutant (Kv7.2 D212G)"][0],
                    v_mt=traces["mutant (Kv7.2 D212G)"][1])
with open(f"{MUT}/../wo6b_taskD1_fig6a.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\n-> wo6b_taskD1_fig6a.json, wo6b_taskD1_fig6a_traces.npz")
