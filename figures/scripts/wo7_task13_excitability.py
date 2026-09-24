"""R859C three-region excitability figure and the spike-count similarity denominator.

Reads the archived NEURON current-clamp traces of the wild-type and R859C cells
(35 injected-current levels, 20-360 pA in 10 pA steps), counts action potentials at
each level, and writes:

    figures/r859c_excitability_ladder.csv       per-level spike counts and region
    figures/similarity_denominator.json         summed |WT - variant| spike count
                                                (the search's own value, 37, is read
                                                from similarity_denominator_d37.json,
                                                written by similarity_denominator_d37.py)
    figures/r859c_excitability_ladder.{pdf,png} the figure

Spike detection is transcribed from the original R859C search pipeline's detector
(`find_spikes` plus the Spike Count line of `calculate_spike_properties`).  It is
not an upward-threshold-crossing count: it clips the trace to +/-10 mV, marks every
transition in both directions, and takes half the transition count, so a spike
still in progress at the end of the sweep rounds down.  The spike counts stored in
the candidate database were produced by this rule, so it defines the denominator.
A plain upward-crossing count is computed as well and reported alongside.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
FILES = {"WT": os.path.join(ARCH, "APthreshold WT-2005 - log every 10th data point.txt"),
         "MT": os.path.join(ARCH, "APthreshold R859C - log every 10th data point.txt")}
OUT = os.path.join(REPO, "figures")


def load(path):
    """Read a tab-separated trace file: returns t (ms), currents (pA), and
    voltages (mV) with shape (n_currents, n_time)."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        f.readline()                      # the "3000" sample-count line
        rows = [[float(p) for p in line.rstrip("\n").split("\t") if p != ""]
                for line in f if line.strip()]
    data = np.array(rows)
    currents = np.array([int(h.split()[0]) for h in header[1:] if h.strip()])
    return data[:, 0], currents, data[:, 1:].T          # t, I, v[(n_currents, n_time)]


# ---- spike detector of the original R859C search pipeline, transcribed -------
def find_spikes(x):
    v = x.copy()
    threshold = -10
    new_v = np.zeros(v.shape)
    v[np.where(v < threshold)] = -10
    v[np.where(v > threshold)] = 10
    new_v[np.where(np.diff(v) < 0)] = -1
    new_v[np.where(np.diff(v) > 0)] = 1
    return ((new_v != 0).sum(axis=1),)


# ---- pipeline Spike Count: half the number of threshold transitions ----------
def spike_count_pipeline(x):
    return np.array(find_spikes(x)[0] / 2, dtype=int)


# ---- alternative rule: number of upward crossings of -10 mV -------------------
def spike_count_upward(x):
    above = x > -10
    return (above[:, 1:] & ~above[:, :-1]).sum(axis=1).astype(int)


def contiguous_runs(mask, currents):
    """Yield (lo_pA, hi_pA, n_levels) for each maximal run of True in mask."""
    runs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((int(currents[start]), int(currents[i - 1]), i - start))
            start = None
    return runs


def main():
    t, cur, v_wt = load(FILES["WT"])
    _, cur2, v_mt = load(FILES["MT"])
    assert (cur == cur2).all()

    sc_wt = spike_count_pipeline(v_wt)
    sc_mt = spike_count_pipeline(v_mt)
    up_wt = spike_count_upward(v_wt)
    up_mt = spike_count_upward(v_mt)

    print(f"{v_wt.shape[0]} current levels x {v_wt.shape[1]} "
          f"samples, t = {t[0]:.1f} .. {t[-1]:.1f} ms")
    print(f"currents {cur[0]}..{cur[-1]} pA in steps of {cur[1]-cur[0]}\n")

    d_pipeline = int(np.abs(sc_wt - sc_mt).sum())
    d_upward = int(np.abs(up_wt - up_mt).sum())

    print(f"{'I (pA)':>7} {'WT':>4} {'R859C':>6} {'diff':>5}   "
          f"{'WT_up':>6} {'MT_up':>6}")
    for c, a, b, ua, ub in zip(cur, sc_wt, sc_mt, up_wt, up_mt):
        print(f"{c:>7} {a:>4} {b:>6} {int(b)-int(a):>+5}   {ua:>6} {ub:>6}")

    less = sc_mt < sc_wt
    same = sc_mt == sc_wt
    more = sc_mt > sc_wt
    firing = (sc_wt > 0) | (sc_mt > 0)

    print(f"\ntotals: WT {int(sc_wt.sum())}, R859C {int(sc_mt.sum())}  "
          f"(upward-crossing rule: WT {int(up_wt.sum())}, R859C {int(up_mt.sum())})")
    print(f"summed |WT - variant| over 35 levels, pipeline detector = {d_pipeline}")
    print(f"summed |WT - variant| over 35 levels, upward-crossing    = {d_upward}")
    i150 = int(np.where(cur == 150)[0][0])
    print(f"at 150 pA: WT {sc_wt[i150]}, R859C {sc_mt[i150]}")
    print(f"\nregions (levels where at least one cell fires):")
    for name, mask in (("less excitable", less & firing),
                       ("indistinguishable", same & firing),
                       ("more excitable", more & firing)):
        print(f"  {name:<20} {contiguous_runs(mask & firing, cur)}")
    print(f"  silent in both       {contiguous_runs(~firing, cur)}")

    os.makedirs(OUT, exist_ok=True)

    # ---- CSV beside the figure ----------------------------------------------
    csv_path = os.path.join(OUT, "r859c_excitability_ladder.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["injected_current_pA", "spikes_wild_type", "spikes_R859C",
                    "difference_variant_minus_wt", "abs_difference", "region",
                    "spikes_wild_type_upward_rule", "spikes_R859C_upward_rule"])
        for i, c in enumerate(cur):
            if not firing[i]:
                region = "silent in both"
            elif less[i]:
                region = "less excitable"
            elif same[i]:
                region = "indistinguishable"
            else:
                region = "more excitable"
            w.writerow([int(c), int(sc_wt[i]), int(sc_mt[i]),
                        int(sc_mt[i]) - int(sc_wt[i]),
                        abs(int(sc_mt[i]) - int(sc_wt[i])), region,
                        int(up_wt[i]), int(up_mt[i])])
    print(f"\nwrote {csv_path}")

    # ---- the similarity denominator, stored as JSON ---------------------------
    # the search's own value comes from its in-process ladders (NEURON), computed
    # by similarity_denominator_d37.py; this script only reads it
    with open(os.path.join(REPO, "figures", "similarity_denominator_d37.json")) as f:
        d_search = int(json.load(f)["denominator"])
    denom = {
        "measure": "Spike Count",
        "quantity": "observation_summary_statistics_MT['Spike Count WT delta'].sum()",
        "definition": "summed |wild-type - untreated variant| action-potential count "
                      "over all 35 injected-current levels",
        "value_pipeline_detector": d_pipeline,
        "value_upward_crossing_rule": d_upward,
        "computed_from_search_ladders": d_search,
        "wild_type_total_spikes": int(sc_wt.sum()),
        "variant_total_spikes": int(sc_mt.sum()),
        "detector_source": "R859C search pipeline spike detector (find_spikes above)",
        "traces": {k: os.path.relpath(v, REPO) for k, v in FILES.items()},
    }
    dpath = os.path.join(REPO, "figures", "similarity_denominator.json")
    with open(dpath, "w") as f:
        json.dump(denom, f, indent=2)
    print(f"wrote {dpath}")

    # ---- figure --------------------------------------------------------------
    apply_style()
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(7.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})

    # region shading, drawn first so it sits under the data
    regions = [("less excitable\n140-210 pA", less & firing, C.BAND_LESS),
               ("indistinguishable\n220-250 pA", same & firing, C.BAND_SAME),
               ("more excitable\n260-360 pA", more & firing, C.BAND_MORE)]
    step = cur[1] - cur[0]
    for _, mask, col in regions:
        for lo, hi, _n in contiguous_runs(mask, cur):
            for a in (ax, axd):
                a.axvspan(lo - step / 2, hi + step / 2, color=col,
                          alpha=0.09, linewidth=0, zorder=0)

    # white ring on the marks so the two series stay separable where they coincide
    ax.plot(cur, sc_wt, "-o", color=C.WT, label="wild type", zorder=3,
            markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(cur, sc_mt, "-s", color=C.VARIANT, label="R859C variant", zorder=4,
            markersize=3.6, markeredgecolor="white", markeredgewidth=0.7)
    ax.set_ylabel("action potentials in 300 ms (count)")
    ax.set_xlim(cur[0] - step, cur[-1] + step)
    ax.set_ylim(-2.5, 36)

    # series legend top-left; region legend in the empty lower-right of the panel
    leg1 = ax.legend(loc="upper left", ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=[Patch(facecolor=c, alpha=0.28, label=n.replace("\n", " "))
                       for n, _m, c in regions],
              loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1, fontsize=7.5)

    # the headline point, direct-labelled in clear space below the curves
    ax.annotate(f"150 pA:  wild type {sc_wt[i150]},  variant {sc_mt[i150]}",
                xy=(150, 4.0), xytext=(178, 8.0),
                color=C.MUTED, fontsize=8, va="center",
                arrowprops=dict(arrowstyle="-", color=C.MUTED, linewidth=0.7,
                                shrinkA=2, shrinkB=2))

    diff = sc_mt - sc_wt
    axd.axhline(0, color=C.MUTED, linewidth=0.8, zorder=2)
    axd.bar(cur, diff, width=step * 0.62, zorder=3,
            color=[C.VARIANT if d < 0 else (C.BAND_SAME if d == 0 else C.WT)
                   for d in diff])
    axd.set_ylabel("variant − wild type\n(count)")
    axd.set_xlabel("injected current (pA)")
    axd.set_xticks(cur[::2])
    axd.set_ylim(diff.min() - 1.2, diff.max() + 1.2)

    fig.suptitle("R859C excitability across the full stimulus ladder",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.005,
             "Somatic current clamp: 50 ms delay, 200 ms step, 300 ms total;\n"
             "35 levels, 20–360 pA in 10 pA increments. Spikes counted with the\n"
             "R859C search pipeline's spike detector.\n"
             f"Totals over the ladder: wild type {int(sc_wt.sum())}, "
             f"variant {int(sc_mt.sum())}.",
             ha="left", va="top", fontsize=7.5, color=C.MUTED)

    paths = save(fig, os.path.join(OUT, "r859c_excitability_ladder"))
    print("wrote " + ", ".join(paths))


if __name__ == "__main__":
    main()
