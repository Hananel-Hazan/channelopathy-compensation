"""Kv7.2 D212G stimulus-ladder figure: somatic voltage at every injected-current level.

One panel per current level (3 x 4 grid), with the reference, untreated variant and
compensated variant traces overlaid; each panel title gives the current in nA and
the spike count of each arm.  The time axis is in ms and the voltage axis in mV.

Reads wo7_kv7_ladder_traces.{npz,json} from this directory (recorded by
wo7_task21d_kv7_ladder_run.py) and writes
figures/kv7_2_stimulus_ladder_traces.{pdf,png,csv}.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")

ARMS = [("reference", C.WT, "-", "reference (wild type)"),
        ("variant", C.VARIANT, "-", "variant (Kv7.2 D212G), untreated"),
        ("compensated", C.COMPENSATED, "--", "variant, compensated")]


def main():
    z = np.load(os.path.join(HERE, "wo7_kv7_ladder_traces.npz"))
    m = json.load(open(os.path.join(HERE, "wo7_kv7_ladder_traces.json")))
    cur = m["currents_nA"]
    counts = m["counts"]
    on, off = m["stim_delay_ms"], m["stim_delay_ms"] + m["stim_dur_ms"]

    print(f"{len(cur)} levels x {len(ARMS)} arms")
    for name, *_ in ARMS:
        print(f"   {name:<12} {counts[name]}  total {sum(counts[name])}")

    apply_style()
    nrow, ncol = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.4, 7.1),
                             sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        ax.axvspan(on, off, color=C.MUTED, alpha=0.06, linewidth=0, zorder=0)
        for name, col, ls, _lab in ARMS:
            ax.plot(z[f"t_{name}_{k}"], z[f"v_{name}_{k}"], ls, color=col,
                    linewidth=0.75, zorder=3)
        ax.set_title(f"{cur[k]:g} nA\nreference {counts['reference'][k]} · "
                     f"variant {counts['variant'][k]} · "
                     f"compensated {counts['compensated'][k]} spikes",
                     loc="left", fontsize=7.0, color=C.INK, pad=3)
        ax.set_xlim(0, m["tstop_ms"])
        ax.set_ylim(-80, 55)
        ax.grid(True, linewidth=0.4)
        ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("somatic membrane\npotential (mV)")

    for name, col, ls, lab in ARMS:
        axes[0, 0].plot([], [], ls, color=col, label=lab, linewidth=1.4)
    axes[0, 0].legend(loc="upper right", fontsize=6.5, handlelength=1.4,
                      borderpad=0.2)

    fig.suptitle("Kv7.2 D212G stimulus ladder — somatic voltage at every "
                 "injected-current level", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.004,
             f"Multi-compartment CA1 pyramidal model, ModelDB 118986. "
             f"{m['stim_dur_ms']:g} ms current step beginning at "
             f"{m['stim_delay_ms']:g} ms (shaded), {m['tstop_ms']:g} ms total, "
             f"{m['celsius']:g} °C. Spikes counted by upward crossing of "
             f"0 mV at the soma (wo6b_lib.py:31).\n"
             f"Compensated arm: sodium ×0.79, delayed-rectifier potassium ×1.94, "
             f"leak ×0.97, with the mutated M-current untouched — restoring "
             f"{m['compensated_similarity_pct']:.1f}% of the reference firing "
             f"pattern.\n"
             f"Totals over the ladder: reference {sum(counts['reference'])}, "
             f"untreated variant {sum(counts['variant'])}, compensated "
             f"{sum(counts['compensated'])} action potentials.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    paths = save(fig, os.path.join(OUT, "kv7_2_stimulus_ladder_traces"))
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "kv7_2_stimulus_ladder_traces.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["injected_current_nA", "spikes_reference", "spikes_variant",
                    "spikes_compensated"])
        for i, a in enumerate(cur):
            w.writerow([a, counts["reference"][i], counts["variant"][i],
                        counts["compensated"][i]])
        w.writerow(["TOTAL", sum(counts["reference"]), sum(counts["variant"]),
                    sum(counts["compensated"])])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
