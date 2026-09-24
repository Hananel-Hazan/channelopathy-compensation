"""Plot solution topology against the selection threshold.

Reads wo7_topology_sweep.json (written by wo7_task23_topology_sweep.py), prints
the full sweep table for every variant, and writes
figures/solution_topology_vs_threshold.{pdf,png} (via wo7_style.save) and
figures/solution_topology_vs_threshold.csv.

The decomposition is shown in both spaces, with all three principal directions
at every selection threshold, plus a check of whether the linear-space result
merely follows absolute magnitude.

Why both spaces: the search box is defined multiplicatively (a factor either side
of a baseline) and the lattice is geometric, so the region and the sampling both
live in log space.  A decomposition in linear coordinates is strongly affected by
how far apart the three baselines sit in absolute units (sodium 120, potassium 36,
leak 0.03 mS/cm2, four orders apart) rather than only by the shape of the
solution set.

Why all three directions: one number cannot separate a band from the
alternatives.  A band has pc1 near 100; a plane has pc1 and pc2 comparable with
pc3 near zero; a filled blob has all three comparable.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "figures")

CASE = {
    "sodium activation +6.1 mV (R859C, ModelDB 87585)": ("R859C (sodium +6.1 mV)", "-"),
    "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)":
        ("Kv7.2 D212G (potassium +4.7 mV)", "--"),
}
PCCOL = {"pc1": C.WT, "pc2": C.VARIANT, "pc3": C.COMPENSATED}
AXCOL = {"sodium": C.VARIANT, "potassium": C.WT, "leak": C.N}
BASE = (120.0, 36.0, 0.03)


def series(v, key, space):
    suffix = "_log" if space == "log" else ""
    return ([s["frac_of_spiking_pct"] for s in v["sweep"]],
            [s[key + suffix] for s in v["sweep"]])


def main():
    R = json.load(open(os.path.join(HERE, "wo7_topology_sweep.json")))
    V = R["variants"]
    print(f"machine {R['host']} | {R['gpu']} | lattice {R['grid_n']}^3 = "
          f"{R['grid_n']**3:,} points per variant\n")

    for vname, v in V.items():
        print(f"=== {vname}")
        print(f"    untreated distance {v['d_untreated']}, best {v['best_distance']} "
              f"({v['best_similarity_pct']:.2f} %), {v['n_tied_at_min']} tied")
        print(f"    {'kept':>9} {'share%':>8} | "
              f"{'pc1':>6} {'pc2':>6} {'pc3':>6} {'plane':>6} {'PC1 on':>10} | "
              f"{'pc1':>6} {'pc2':>6} {'pc3':>6} {'plane':>6} {'PC1 on':>10}")
        print(f"    {'':>9} {'':>8} | {'--------- LINEAR ---------':^40} | "
              f"{'----------- LOG -----------':^40}")
        for s in v["sweep"]:
            print(f"    {s['k']:>9,} {s['frac_of_spiking_pct']:>8.3f} | "
                  f"{s['pc1']:>6.2f} {s['pc2']:>6.2f} {s['pc3']:>6.2f} "
                  f"{s['plane_residual_pct']:>6.2f} {s['pc1_axis']:>10} | "
                  f"{s['pc1_log']:>6.2f} {s['pc2_log']:>6.2f} {s['pc3_log']:>6.2f} "
                  f"{s['plane_residual_pct_log']:>6.2f} {s['pc1_axis_log']:>10}")
        print(f"    PC1 carried by (linear): "
              f"{sorted({s['pc1_axis'] for s in v['sweep']})}   "
              f"(log): {sorted({s['pc1_axis_log'] for s in v['sweep']})}\n")

    print("=" * 78)
    print("TEST: is the linear-space decomposition tracking absolute magnitude?")
    print("  prediction -- if it is, PC1 sits on SODIUM (baseline 120, the largest")
    print("  of 120 / 36 / 0.03) at every threshold and stays there.\n")
    for vname, v in V.items():
        n = len(v["sweep"])
        on_na = sum(s["pc1_axis"] == "sodium" for s in v["sweep"])
        on_na_log = sum(s["pc1_axis_log"] == "sodium" for s in v["sweep"])
        mean_load = float(np.mean([s["pc1_loading"][0] for s in v["sweep"]]))
        print(f"  {vname[:60]}")
        print(f"      linear: PC1 on sodium at {on_na:>3}/{n} thresholds, "
              f"mean |loading on sodium| {mean_load:.3f}")
        print(f"      log   : PC1 on sodium at {on_na_log:>3}/{n} thresholds")
    print("=" * 78 + "\n")

    apply_style()
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.2, 4.5))

    for ax, space, title in (
            (axa, "log", "a   log space — where the box and the sampling live"),
            (axb, "linear", "b   linear space")):
        for vname, (lbl, ls) in CASE.items():
            if vname not in V:
                continue
            for key in ("pc1", "pc2", "pc3"):
                x, y = series(V[vname], key, space)
                ax.plot(x, y, ls, color=PCCOL[key], linewidth=1.5, alpha=0.95)
        ax.set_xscale("log")
        ax.set_ylim(-3, 103)
        ax.set_xlabel("selection threshold — candidates kept, as a\n"
                      "share of those that fire (%)")
        ax.set_title(title, loc="left", color=C.INK, fontsize=9)
    axa.set_ylabel("share of the variation (%)")
    axb.set_ylabel("share of the variation (%)")

    axa.legend(handles=[Line2D([], [], color=PCCOL[k], lw=1.8,
                               label=f"{k} (dimensionless share)")
                        for k in ("pc1", "pc2", "pc3")]
               + [Line2D([], [], color=C.MUTED, lw=1.5, ls=ls, label=lbl)
                  for lbl, ls in CASE.values()],
               loc="center left", fontsize=6.5)

    vname = "sodium activation +6.1 mV (R859C, ModelDB 87585)"
    if vname in V:
        v = V[vname]
        x = [s["frac_of_spiking_pct"] for s in v["sweep"]]
        for j, axis_name in enumerate(("sodium", "potassium", "leak")):
            axc.plot(x, [s["pc1_loading"][j] for s in v["sweep"]], "-",
                     color=AXCOL[axis_name], linewidth=1.6,
                     label=f"{axis_name} (baseline {BASE[j]:g} mS/cm²)")
        axc.set_xscale("log")
        axc.set_ylim(-0.03, 1.03)
        axc.set_xlabel("selection threshold — candidates kept, as a\n"
                       "share of those that fire (%)")
        axc.set_ylabel("|loading| of the first direction\n(dimensionless, 0–1)")
        axc.legend(loc="center left", fontsize=6.5)
        axc.set_title("c   linear space: which axis carries it",
                      loc="left", color=C.INK, fontsize=9)

    fig.suptitle("Solution topology: all three principal directions, in both spaces",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.048,
             f"Differentiable model, log-spaced lattice of {R['grid_n']}³ = "
             f"{R['grid_n']**3:,} conductance triples per variant over the "
             f"pipeline's own box, each evaluated at all 35 injected-current levels. "
             f"Selection is every candidate within a given\n"
             f"spike-count distance of the wild type, not the best k — the score is "
             f"an integer, so candidates arrive in large tie blocks and a top-k cut "
             f"slices arbitrarily through one. Each axis is normalised by its own "
             f"range before the decomposition.\n"
             f"Panel c: which axis the linear first direction sits on. It is NOT "
             f"pinned to the largest-baseline axis — it moves, and at the loosest "
             f"selections it goes to the leak axis (loading 1.00), which has the "
             f"narrowest box in decades\n(÷50 to ×50 against ÷5,000 to ×5,000 for "
             f"the gated conductances). That is where the linear second direction "
             f"collapses; in log space it does not collapse.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    # the section label in panel b's title is not a quantity; suppress the style
    # check's report for it
    paths = save(fig, os.path.join(OUT, "solution_topology_vs_threshold"),
                 allow=())
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "solution_topology_vs_threshold.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "candidates_kept", "share_of_spiking_pct",
                    "spike_count_distance_cut", "similarity_cut_pct",
                    "linear_pc1_pct", "linear_pc2_pct", "linear_pc3_pct",
                    "linear_plane_residual_pct", "linear_pc1_axis",
                    "linear_pc1_loading_sodium", "linear_pc1_loading_potassium",
                    "linear_pc1_loading_leak",
                    "log_pc1_pct", "log_pc2_pct", "log_pc3_pct",
                    "log_plane_residual_pct", "log_pc1_axis",
                    "log_pc1_loading_sodium", "log_pc1_loading_potassium",
                    "log_pc1_loading_leak"])
        for vname, v in V.items():
            for s in v["sweep"]:
                w.writerow([vname, s["k"], round(s["frac_of_spiking_pct"], 6),
                            s["distance_cut"], round(s["similarity_cut_pct"], 4),
                            round(s["pc1"], 4), round(s["pc2"], 4),
                            round(s["pc3"], 4),
                            round(s["plane_residual_pct"], 4), s["pc1_axis"],
                            *[round(x, 4) for x in s["pc1_loading"]],
                            round(s["pc1_log"], 4), round(s["pc2_log"], 4),
                            round(s["pc3_log"], 4),
                            round(s["plane_residual_pct_log"], 4), s["pc1_axis_log"],
                            *[round(x, 4) for x in s["pc1_loading_log"]]])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
