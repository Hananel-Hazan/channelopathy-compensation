"""Kv7.2 D212G reproduction panel: the published protocol, reproduced.

Panel a plots the somatic voltage of the reference (wild-type) and variant
(Kv7.2 D212G) cells under the protocol of ModelDB 118986's `fig6a.hoc` (0.47 nA
somatic step from 5 ms to 405 ms, tstop 500 ms, 35 degrees C), with each cell's
spike times marked above the traces.  Panel b shows the screenshot distributed
with the model package, for comparison.

Inputs (produced by experiments/wo6/phase2/wo6b_taskD1_fig6a.py, which loads the
model's `fig6a.hoc` unmodified and adds only a voltage recording and a spike
count):
    experiments/wo6/phase2/wo6b_taskD1_fig6a_traces.npz
    experiments/wo6/phase2/wo6b_taskD1_fig6a.json
    experiments/wo6/phase2/mutant/screenshot.jpg   (from scripts/fetch_modeldb.sh)

Writes figures/kv7_2_reproduction_published_protocol.{pdf,png,csv}.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P2 = os.path.join(REPO, "experiments", "wo6", "phase2")
OUT = os.path.join(REPO, "figures")

STIM_ON, STIM_OFF, STIM_AMP = 5.0, 405.0, 0.47     # ms, ms, nA


def main():
    z = np.load(os.path.join(P2, "wo6b_taskD1_fig6a_traces.npz"))
    d = json.load(open(os.path.join(P2, "wo6b_taskD1_fig6a.json")))
    wt = d["arms"]["wild type"]
    mt = d["arms"]["mutant (Kv7.2 D212G)"]

    print(f"protocol {STIM_AMP} nA, {STIM_ON}-{STIM_OFF} ms, "
          f"tstop 500 ms, 35 C  (fig6a.hoc:22,:43-45,:47)")
    print(f"reference: {wt['spikes_during_step']} action potential "
          f"({wt['firing_rate_Hz']} Hz), peak {wt['peak_v_mV']:.2f} mV")
    print(f"variant:   {mt['spikes_during_step']} action potentials "
          f"({mt['firing_rate_Hz']} Hz), peak {mt['peak_v_mV']:.2f} mV")
    print(f"ratio {d['mutant_over_wildtype_rate']:g}x\n")
    print("variant spike times (ms): " +
          ", ".join(f"{t:.1f}" for t in mt["spike_times_ms"]))
    print("reference spike times (ms): " +
          ", ".join(f"{t:.1f}" for t in wt["spike_times_ms"]))

    apply_style()
    fig = plt.figure(figsize=(7.6, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.30,
                      left=0.11, right=0.985, top=0.90, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1])

    # ---- panel a: the reproduction ------------------------------------------
    ax.axvspan(STIM_ON, STIM_OFF, color=C.MUTED, alpha=0.06, linewidth=0, zorder=0)
    ax.plot(z["t_wt"], z["v_wt"], "-", color=C.WT, linewidth=1.1,
            label=f"reference (wild type) — {wt['spikes_during_step']} action potential")
    ax.plot(z["t_mt"], z["v_mt"], "-", color=C.VARIANT, linewidth=1.1, alpha=0.9,
            label=f"variant (Kv7.2 D212G) — {mt['spikes_during_step']} action potentials")

    # individual spike times, marked as ticks above the traces
    for t in mt["spike_times_ms"]:
        ax.plot([t, t], [46, 52], "-", color=C.VARIANT, linewidth=1.1,
                solid_capstyle="butt", zorder=5)
    for t in wt["spike_times_ms"]:
        ax.plot([t, t], [54, 60], "-", color=C.WT, linewidth=1.1,
                solid_capstyle="butt", zorder=5)

    ax.set_xlim(0, 500)
    ax.set_ylim(-75, 64)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("somatic membrane potential (mV)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              fontsize=7.5, borderaxespad=0.0)
    ax.set_title("a   reproduced from the model's own fig6a.hoc, unmodified",
                 loc="left", color=C.INK, pad=22)
    ax.text(500, 57, "spike times  ", ha="right", va="center", fontsize=7,
            color=C.MUTED)

    # ---- panel b: the screenshot shipped with the model ---------------------
    img = mpimg.imread(os.path.join(P2, "mutant", "screenshot.jpg"))
    axs.imshow(img)
    axs.set_axis_off()
    axs.set_title("b   the screenshot distributed inside ModelDB 118986, "
                  "for comparison", loc="left", color=C.INK)

    fig.suptitle("Kv7.2 D212G: the published protocol reproduced on the model",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.075,
             f"Protocol as shipped: {STIM_AMP} nA somatic current step from "
             f"{STIM_ON:g} ms to {STIM_OFF:g} ms, 500 ms total, 35 °C "
             f"(fig6a.hoc:22, :43–45, :47). A single flag selects which mechanism "
             f"carries the M-current.\n"
             f"Variant spike times (ms): "
             + ", ".join(f"{t:.1f}" for t in mt["spike_times_ms"])
             + f".  Reference fires once, at {wt['spike_times_ms'][0]:.1f} ms.  "
               f"The variant fires {d['mutant_over_wildtype_rate']:g}× the "
               f"reference rate.\n"
             f"Panel b is the screenshot distributed with ModelDB 118986 "
             f"(downloaded by scripts/fetch_modeldb.sh), shown for comparison.",
             ha="left", va="top", fontsize=7, color=C.MUTED)

    paths = save(fig, os.path.join(OUT, "kv7_2_reproduction_published_protocol"),
                 allow=("b   the screenshot",))
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "kv7_2_reproduction_published_protocol.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "spikes_during_step", "firing_rate_Hz",
                    "first_spike_latency_after_onset_ms", "peak_mV",
                    "resting_mV", "spike_times_ms"])
        for name, a in (("reference (wild type)", wt),
                        ("variant (Kv7.2 D212G)", mt)):
            w.writerow([name, a["spikes_during_step"], a["firing_rate_Hz"],
                        round(a["first_spike_latency_ms"], 3),
                        round(a["peak_v_mV"], 3), a["resting_v_mV"],
                        "; ".join(f"{t:.3f}" for t in a["spike_times_ms"])])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
