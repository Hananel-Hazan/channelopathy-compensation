"""Throughput of the differentiable model versus gradient-descent success.

Two panels:

  a  candidate parameter sets evaluated per second, on the stated unit, across both
     pathways (NEURON and the differentiable model) and both graphics cards.  The two
     fused rates are the five-launch medians in figures/throughput_repeats.csv
     (34,279 and 63,649 sets/s); the single launches are in that file's
     original_single_run column.
  b  for each variant configuration, gradient descent under the voltage
     mean-squared-error loss against direct search over the same parameter box,
     with a rule at 0% marking "no better than no treatment".

All plotted values are literals in this script; nothing is measured here.

Colour: panel a is one series (a single measure across configurations), so it uses a
neutral colour.  Panel b's two marks are methods, not cells, so they do not use the
colours reserved for wild type and variant; the pair used remains distinguishable
(dE 39.4) under simulated colour-blindness.

Writes figures/throughput_vs_gradient_success.pdf/.png and
figures/throughput_vs_gradient_success.csv.
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")

GRADIENT = "#7B3294"      # method, not a cell
DIRECT = "#E69F00"

# ---- panel a: (label, sets per second, group, row label written to the CSV) --
THROUGHPUT = [
    ("NEURON, one processor core",            1.1,     "non-differentiable", "neuron_one_core"),
    ("NEURON, 72-core node",                  16.4,    "non-differentiable", "neuron_72_core_node"),
    ("differentiable, batched, RTX 2070",     543.0,   "differentiable",     "rtx2070_batched_unfused"),
    ("differentiable, fused, RTX 2070",       34279.4, "differentiable",     "rtx2070_batched_fused"),
    ("differentiable, fused, RTX 4070 Ti",    63649.2, "differentiable",     "rtx4070ti_batched_fused"),
]

# ---- panel b: (variant, gradient descent %, direct search %) -----------------
SUCCESS = [
    ("sodium conductance\nheld at 140",              94.23, 96.15),
    ("potassium conductance\nheld at 20",           -87.50, 93.97),
    ("leak conductance\nheld at 0.30",             -123.38, 95.02),
    ("sodium activation\nshifted +6.1 mV (R859C)",    1.74, 97.60),
]


def main():
    apply_style()
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.6, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ================= panel a =================================================
    labels = [r[0] for r in THROUGHPUT]
    vals = np.array([r[1] for r in THROUGHPUT])
    y = np.arange(len(vals))[::-1]          # first row at the top

    axa.barh(y, vals, height=0.6, color=C.MUTED, linewidth=0, zorder=3)
    axa.set_xscale("log")
    axa.set_xlim(0.5, 2.6e5)
    axa.set_yticks(y)
    axa.set_yticklabels(labels)
    axa.set_ylim(-0.75, len(vals) - 0.25)
    axa.set_xlabel("candidate parameter sets evaluated per second (log scale)")
    axa.grid(axis="y", visible=False)

    for yi, v in zip(y, vals):
        axa.text(v * 1.35, yi, f"{v:,.0f}" if v >= 100 else f"{v:,.1f}",
                 va="center", ha="left", fontsize=8.5, color=C.INK)

    # separate the two pathways with a rule and label them
    split = 1.5   # between the two NEURON rows and the three differentiable rows
    axa.axhline(split, color=C.GRID, linewidth=1.0, zorder=2)
    # (the row labels already name the pathway, so no group label is added)
    axa.set_title("a   what the fast forward model buys", loc="left", color=C.INK)

    # ================= panel b =================================================
    names = [r[0] for r in SUCCESS]
    g = np.array([r[1] for r in SUCCESS])
    d = np.array([r[2] for r in SUCCESS])
    yb = np.arange(len(names))[::-1]

    axb.axvspan(-140, 0, color=C.MUTED, alpha=0.07, linewidth=0, zorder=0)
    axb.axvline(0, color=C.MUTED, linewidth=1.2, zorder=2)

    for yi, gi, di in zip(yb, g, d):
        axb.plot([gi, di], [yi, yi], "-", color=C.GRID, linewidth=1.6, zorder=3)
    axb.plot(g, yb, "o", color=GRADIENT, markersize=8, zorder=5,
             markeredgecolor="white", markeredgewidth=0.8,
             label="gradient descent, voltage MSE loss")
    axb.plot(d, yb, "D", color=DIRECT, markersize=7, zorder=5,
             markeredgecolor="white", markeredgewidth=0.8,
             label="direct search, same box")

    axb.set_yticks(yb)
    axb.set_yticklabels(names)
    axb.set_ylim(-0.75, len(names) - 0.25)
    axb.set_xlim(-140, 132)
    axb.set_xticks([-100, -50, 0, 50, 100])
    axb.set_xlabel("wild-type firing pattern restored (%)")
    axb.grid(axis="y", visible=False)

    axb.text(-4, -0.62, "← no better than no treatment", fontsize=7.5,
             color=C.MUTED, ha="right", va="center")
    for yi, gi in zip(yb, g):
        if gi < 0:
            axb.text(gi, yi + 0.30, f"{gi:.1f}%", fontsize=7.5, color=GRADIENT,
                     ha="center", va="bottom")
    axb.text(1.74, yb[3] + 0.30, "1.7%", fontsize=7.5, color=GRADIENT,
             ha="center", va="bottom")

    axb.legend(loc="upper left", fontsize=8)
    axb.set_title("b   what the gradient does not buy", loc="left", color=C.INK)

    fig.suptitle("The differentiable model earns its place through throughput, "
                 "not through descent", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.225,
             "Unit, both panels: one candidate parameter set = one conductance "
             "triple across 35 injected-current levels, 300 ms each, 0.01 ms "
             "integration step.\n"
             "a  The non-differentiable rate includes summary statistics that the "
             "differentiable rate does not, so the comparison flatters the "
             "differentiable side. No scaling projection may be drawn from it: "
             "across the two\n"
             "    accelerator generations throughput rose 1.86× against a 5.37× "
             "difference in rated arithmetic throughput.\n"
             "b  Gradient descent under the voltage "
             "mean-squared-error loss against direct search over the same parameter "
             "box, scored on spike count across all 35 levels.\n"
             "    Reaching the compensating configuration in 1 case of 4, and "
             "prescribing a configuration worse than no treatment in 2.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.235, 1, 0.94))

    paths = save(fig, os.path.join(OUT, "throughput_vs_gradient_success"))
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "throughput_vs_gradient_success.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "row", "series", "value", "units", "id"])
        for lab, v, grp, lid in THROUGHPUT:
            w.writerow(["a", lab, grp, v, "candidate parameter sets per second", lid])
        for nm, gi, di in SUCCESS:
            n = nm.replace("\n", " ")
            w.writerow(["b", n, "gradient descent, voltage MSE loss", gi,
                        "% of wild-type firing pattern restored", "gradient_descent_vs_direct_search"])
            w.writerow(["b", n, "direct search, same box", di,
                        "% of wild-type firing pattern restored", "gradient_descent_vs_direct_search"])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
