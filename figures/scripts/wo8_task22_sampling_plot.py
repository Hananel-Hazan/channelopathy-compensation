"""Plot the comparison of uniform and log-uniform sampling.

Reads wo7_sampling.json (written by wo7_task22_sampling_run.py) and writes
figures/sampling_scheme_comparison.{pdf,png} and
figures/sampling_scheme_comparison.csv (one row per variant, scheme and seed).

This is a controlled comparison of the two sampling schemes on the
differentiable model, not a replay of either original search, so the legend
names the scheme.

Panel a: candidates drawn before the first success, under both schemes, for
every variant.  Bars are the median across independent repeats (each with its
own seed); the horizontal line is the full spread.

Panel b: the fraction of draws producing a cell that fires at all (candidates
that never spike are discarded by the search).  This explains panel a: a box
spanning several decades is mostly very large conductances by volume, and
drawing uniformly across the interval puts almost all of the draws there.

Where a scheme did not reach the target within the budget in any repeat, the
bar is drawn at the budget, hatched and labelled "not reached"; such bars are
lower bounds on the true cost.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "figures")

SHORT = {
    "sodium activation +6.1 mV (R859C, ModelDB 87585)": "sodium +6.1 mV\n(R859C)",
    "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)":
        "potassium +4.7 mV\n(Kv7.2 D212G)",
    "potassium activation +9.4 mV (twice the measured shift)":
        "potassium +9.4 mV",
    "combined: sodium +6.1 mV and potassium +4.7 mV": "both shifts\ntogether",
}
# the sampling schemes are methods, not cell types, so they use colours distinct
# from those wo7_style reserves for wild type, variant and compensated cells
UNI, LOG = "#7B3294", "#E69F00"


def main():
    R = json.load(open(os.path.join(HERE, "wo7_sampling.json")))
    V = R["variants"]
    budget = R["budget_per_repeat"]
    names = list(V)
    n = len(names)

    print(f"machine {R['host']} | {R['gpu']}")
    print(f"{R['repeats']} independent repeats per (variant, scheme), "
          f"{budget:,} draws each, seeds {R['seed0']}..{R['seed0']+R['repeats']-1}")
    print(f"success = within {R['margin_pct']:g} percentage points of the known "
          f"best for that variant\n")

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True,
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    y = np.arange(n)[::-1]
    h = 0.34

    ratios = []
    for i, vn in enumerate(names):
        for scheme, col, off in (("uniform", UNI, +h / 2), ("loguniform", LOG, -h / 2)):
            s = V[vn]["schemes"][scheme]
            reached = s["n_reached"] > 0
            val = s["first_hit_median"] if reached else budget
            lo = s["first_hit_min"] if reached else budget
            hi = s["first_hit_max"] if reached else budget
            axa.barh(y[i] + off, val, height=h, color=col, linewidth=0,
                     zorder=3, hatch=None if reached else "///",
                     edgecolor="white" if reached else col,
                     alpha=1.0 if reached else 0.45)
            if reached and hi > lo:
                axa.plot([lo, hi], [y[i] + off] * 2, "-", color=C.INK,
                         linewidth=1.0, zorder=5)
            lbl = (f"{val:,.0f}" if reached
                   else f"not reached in {budget:,}"
                        f"{'' if s['n_reached'] == 0 else ''}")
            axa.text(val * 1.25, y[i] + off, lbl, va="center", ha="left",
                     fontsize=7.5, color=C.INK if reached else C.MUTED)
        u = V[vn]["schemes"]["uniform"]; g = V[vn]["schemes"]["loguniform"]
        if g["first_hit_median"]:
            base = u["first_hit_median"] if u["n_reached"] else budget
            ratios.append(base / g["first_hit_median"])

    axa.set_xscale("log")
    axa.set_xlim(200, budget * 26)
    axa.set_yticks(y); axa.set_yticklabels([SHORT[v] for v in names], fontsize=8)
    axa.set_ylim(-0.7, n - 0.3)
    axa.set_xlabel("candidate parameter sets drawn before the first success "
                   "(count, log scale)")
    axa.grid(axis="y", visible=False)
    axa.legend(handles=[Patch(facecolor=UNI, label="uniform across the interval "
                                                   "(as in the archived R859C exploration)"),
                        Patch(facecolor=LOG, label="log-uniform across the decades "
                                                   "(the scheme this paper recommends)")],
               loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
               fontsize=7.5, borderaxespad=0.0)
    axa.set_title("a   candidates needed to find a compensating configuration",
                  loc="left", color=C.INK, pad=20)

    for i, vn in enumerate(names):
        for scheme, col, off in (("uniform", UNI, +h / 2), ("loguniform", LOG, -h / 2)):
            s = V[vn]["schemes"][scheme]
            axb.barh(y[i] + off, s["viable_median_pct"], height=h, color=col,
                     linewidth=0, zorder=3)
            axb.plot([s["viable_min_pct"], s["viable_max_pct"]], [y[i] + off] * 2,
                     "-", color=C.INK, linewidth=1.0, zorder=5)
            axb.text(s["viable_median_pct"] + 1.2, y[i] + off,
                     f"{s['viable_median_pct']:.0f}%", va="center", ha="left",
                     fontsize=7.5, color=C.INK)
    axb.set_yticks(y)
    axb.set_yticklabels([SHORT[v] for v in names], fontsize=8)
    axb.tick_params(labelleft=False)
    axb.set_ylim(-0.7, n - 0.3)
    axb.set_xlim(0, 72)
    axb.set_xlabel("draws producing a cell that fires at all (%)")
    axb.grid(axis="y", visible=False)
    axb.set_title("b   why: where the draws land", loc="left", color=C.INK, pad=20)

    if ratios:
        lo_r, hi_r = min(ratios), max(ratios)
        print(f"improvement factor, uniform -> log-uniform: "
              f"{lo_r:,.0f}x to {hi_r:,.0f}x "
              f"(a lower bound wherever uniform never reached the target)")

    fig.suptitle("Drawing across the decades, not across the interval, is what "
                 "makes the search affordable", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.004,
             f"Differentiable model, the R859C search pipeline's parameter box. "
             f"{R['repeats']} independent "
             f"repeats per bar, each with its own seed "
             f"({R['seed0']}–{R['seed0']+R['repeats']-1}), {budget:,} draws per "
             f"repeat; bars are the median across repeats and the\n"
             f"horizontal line is the full spread. Success = reaching within "
             f"{R['margin_pct']:g} percentage points of the best value known for "
             f"that variant, scored on spike count across all 35 injected-current "
             f"levels. A hatched bar means the target was never\n"
             f"reached inside the budget in any repeat, and is drawn at the budget "
             f"— so those bars are lower bounds on the true cost, not measurements "
             f"of it.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.115, 1, 0.94))
    paths = save(fig, os.path.join(OUT, "sampling_scheme_comparison"))
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "sampling_scheme_comparison.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "scheme", "seed", "draws_to_first_success",
                    "reached_within_budget", "viable_draws_pct",
                    "best_similarity_reached_pct", "budget_per_repeat",
                    "target_similarity_pct"])
        for vn in names:
            for scheme in ("uniform", "loguniform"):
                s = V[vn]["schemes"][scheme]
                for j, fh in enumerate(s["first_hit_per_seed"]):
                    w.writerow([SHORT[vn].replace("\n", " "), scheme,
                                R["seed0"] + j,
                                fh if fh is not None else "",
                                "yes" if fh is not None else "no",
                                s["viable_pct_per_seed"][j],
                                s["best_reached_per_seed"][j], budget,
                                round(V[vn]["target_pct"], 4)])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
