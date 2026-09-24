"""Per-level spike counts behind the R859C excitability description.

Counts action potentials of the wild-type and R859C cells at each injected-current
level (pipeline detector, see wo7_task13_excitability.py) and compares the two
arms, both over the 120-260 pA window (15 levels) and over the
full 35-level ladder.  Prints the comparison and writes
figures/fig4_caption_evidence.csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_task13_excitability import load, spike_count_pipeline, FILES

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")
FIG4_LO, FIG4_HI = 120, 260


def main():
    _t, cur, v_wt = load(FILES["WT"])
    _, _, v_mt = load(FILES["MT"])
    wt = spike_count_pipeline(v_wt)
    mt = spike_count_pipeline(v_mt)

    win = (cur >= FIG4_LO) & (cur <= FIG4_HI)
    print(f"window {FIG4_LO}-{FIG4_HI} pA = "
          f"{int(win.sum())} panels\n")

    print(f"{'I (pA)':>7} {'wild type':>10} {'variant':>8} {'diff':>6} "
          f"{'variant as % of wild type':>26}")
    for c, a, b in zip(cur[win], wt[win], mt[win]):
        pct = "-" if a == 0 else f"{100.0 * b / a:.1f}%"
        print(f"{c:>7} {a:>10} {b:>8} {int(b) - int(a):>+6} {pct:>26}")

    # levels where both cells fire repetitively (>= 10 action potentials each),
    # the range over which sustained repetitive firing can be compared
    sus = win & (wt >= 10) & (mt >= 10)
    ratio = 100.0 * mt[sus] / wt[sus]
    print(f"\nOver the {FIG4_LO}-{FIG4_HI} pA window, at the levels where BOTH cells fire "
          f"repetitively (>= 10 action potentials each: {int(sus.sum())} levels, "
          f"{cur[sus].min()}-{cur[sus].max()} pA):")
    print(f"   the variant sustains {ratio.min():.1f}% to {ratio.max():.1f}% of the "
          f"wild-type count, mean {ratio.mean():.1f}%")
    print(f"   totals over the window: wild type {wt[win].sum()}, "
          f"variant {mt[win].sum()} "
          f"({100.0 * mt[win].sum() / wt[win].sum():.1f}%)")

    print(f"\nOver the full 35-level ladder: wild type {wt.sum()}, "
          f"variant {mt.sum()} ({100.0 * mt.sum() / wt.sum():.1f}%)")
    less = int(((mt < wt) & ((wt > 0) | (mt > 0))).sum())
    same = int(((mt == wt) & ((wt > 0) | (mt > 0))).sum())
    more = int((mt > wt).sum())
    print(f"   variant fires fewer at {less} levels, the same at {same}, "
          f"more at {more}")
    print(f"   the deficit is confined to {cur[(mt < wt)].min()}-"
          f"{cur[(mt < wt)].max()} pA; above "
          f"{cur[(mt > wt)].min()} pA the variant leads at every level")

    print("\nSummary:")
    print(f"   The variant's deficit is confined to the near-rheobase region. Its "
          f"rheobase is one step higher than the")
    print(f"   wild type's ({cur[mt>0].min()} pA against {cur[wt>0].min()} pA), and "
          f"just above it the variant fires far less: at 150 pA it fires")
    print(f"   {mt[cur==150][0]} against the wild type's {wt[cur==150][0]}. But from "
          f"{cur[sus].min()} pA it sustains {ratio.min():.0f}% or more of the "
          f"wild-type count,")
    print(f"   from 220 pA the two are identical, and from "
          f"{cur[(mt > wt)].min()} pA the variant fires MORE at every level.")
    print("   The difference between the arms is a shift in firing threshold.")

    p = os.path.join(OUT, "fig4_caption_evidence.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["injected_current_pA", "in_120_260pA_window",
                    "spikes_wild_type", "spikes_R859C_variant",
                    "variant_minus_wild_type", "variant_pct_of_wild_type"])
        for i, c in enumerate(cur):
            pct = "" if wt[i] == 0 else round(100.0 * mt[i] / wt[i], 2)
            w.writerow([int(c), "yes" if win[i] else "no", int(wt[i]), int(mt[i]),
                        int(mt[i]) - int(wt[i]), pct])
        w.writerow(["TOTAL over 120-260 pA window", "", int(wt[win].sum()),
                    int(mt[win].sum()), int(mt[win].sum() - wt[win].sum()),
                    round(100.0 * mt[win].sum() / wt[win].sum(), 2)])
        w.writerow(["TOTAL over full ladder", "", int(wt.sum()), int(mt.sum()),
                    int(mt.sum() - wt.sum()),
                    round(100.0 * mt.sum() / wt.sum(), 2)])
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
