"""NEURON voltage-trace ladder for the R859C variant.

Plots the archived somatic voltage traces of the wild-type and R859C cells at all
35 injected-current levels (20-360 pA) in a 7 x 5 grid, with each panel titled by
the spike counts of the pipeline detector (see wo7_task13_excitability.py).

The traces were integrated at dt = 0.01 ms and logged every tenth step, so one
sample is 0.1 ms and a 300 ms sweep has 3,001 samples.  The time axis is taken from
the time column of the trace files, not from the sample index (which would make
the sweep appear ten times too long).

Writes figures/r859c_stimulus_ladder_traces.{pdf,png} and the plotted data as
figures/r859c_stimulus_ladder_traces.csv.
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save
from wo7_task13_excitability import load, spike_count_pipeline, FILES

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")

STIM_DELAY, STIM_DUR = 50.0, 200.0      # ms; stim.del and stim.dur in Neuron_Test/R859C/neuron.hoc


def main():
    t, cur, v_wt = load(FILES["WT"])
    _, _, v_mt = load(FILES["MT"])
    sc_wt = spike_count_pipeline(v_wt)
    sc_mt = spike_count_pipeline(v_mt)

    print(f"traces {v_wt.shape[0]} levels x {v_wt.shape[1]} "
          f"samples, t = {t[0]:.1f}..{t[-1]:.1f} ms")
    print(f"sample interval = {t[1] - t[0]:.4f} ms "
          f"(dt = 0.01 ms, logged every 10th step: neuron.hoc:253, :261)")
    print(f"a sample-index axis would run 0..{v_wt.shape[1] - 1} "
          f"-- a factor of {(v_wt.shape[1] - 1) / t[-1]:.1f} larger than time in ms\n")

    apply_style()
    fig, axes = plt.subplots(7, 5, figsize=(13.5, 11.5), sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        ax.axvspan(STIM_DELAY, STIM_DELAY + STIM_DUR, color=C.MUTED,
                   alpha=0.06, linewidth=0, zorder=0)
        ax.plot(t, v_wt[k], "-", color=C.WT, linewidth=0.7, zorder=3)
        ax.plot(t, v_mt[k], "-", color=C.VARIANT, linewidth=0.7,
                alpha=0.85, zorder=4)
        ax.set_title(f"{cur[k]} pA   —   wild type {sc_wt[k]}, variant {sc_mt[k]}",
                     loc="left", fontsize=7.5, color=C.INK, pad=2.5)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-85, 55)
        ax.grid(True, linewidth=0.4)
        ax.tick_params(labelsize=6.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (ms)", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("mV", fontsize=7.5)

    # one legend for the whole grid, so identity is never colour-alone
    axes[0, 0].plot([], [], "-", color=C.WT, label="wild type")
    axes[0, 0].plot([], [], "-", color=C.VARIANT, label="R859C variant")
    axes[0, 0].legend(loc="upper right", fontsize=6.5, handlelength=1.2,
                      borderpad=0.2)

    fig.suptitle("R859C stimulus ladder — somatic voltage at all 35 injected-current "
                 "levels", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.004,
             "Archived NEURON traces, 0.01 ms integration step logged every tenth "
             "step (0.1 ms samples, 3,001 points over 300 ms). Shaded band is the "
             "200 ms current step beginning at 50 ms.\n"
             "Time axis derived as sample index × 0.1 ms. "
             "Spike counts in each panel title use the R859C search pipeline's "
             "spike detector.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    paths = save(fig, os.path.join(OUT, "r859c_stimulus_ladder_traces"))
    print("wrote " + ", ".join(paths))

    # ---- the trace data behind it -------------------------------------------
    p = os.path.join(OUT, "r859c_stimulus_ladder_traces.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_ms"] + [f"wild_type_{c}pA_mV" for c in cur]
                   + [f"R859C_{c}pA_mV" for c in cur])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}"]
                       + [f"{v_wt[k, i]:.4f}" for k in range(len(cur))]
                       + [f"{v_mt[k, i]:.4f}" for k in range(len(cur))])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
