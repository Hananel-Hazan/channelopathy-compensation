"""Kv7.2 D212G restoration panel.

Spike count against injected current across the stimulus ladder for the reference
cell, the untreated variant, and the best compensated variant found by the
conductance search.  The mutated M-current is not in the search space: it keeps
the variant's channel mechanism and its unchanged maximal conductance, and only
the sodium, delayed-rectifier potassium and leak conductances are adjusted.

Input: experiments/wo6/phase2/wo6b_taskD2_search.json (2,000 candidates drawn
log-uniformly around the baseline conductances; those rejected as stiff or
divergent are excluded).

Restoration % = (1 - best distance / untreated distance) x 100, where distance is
the summed |spike-count difference| from the reference over the ladder.  Because
the measure is a count, one action potential is worth 100 / untreated distance
percentage points, which sets the precision of the result.

Writes figures/kv7_2_restoration_ladder.{pdf,png,csv} and
figures/kv7_2_restoration_summary.json.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P2 = os.path.join(REPO, "experiments", "wo6", "phase2")
OUT = os.path.join(REPO, "figures")

# baseline conductances (mho/cm2) the search perturbs around, as in
# experiments/wo6/phase2/wo6b_lib.py
BASE = {"gna": 0.045, "gkdr": 0.02, "g_pas": 0.75 / 28000.0}


def main():
    s = json.load(open(os.path.join(P2, "wo6b_taskD2_search.json")))
    cur = np.array(s["currents_nA"])
    ref = np.array(s["wild_type_counts"])
    var = np.array(s["mutant_counts"])
    d_untreated = s["d_untreated"]

    ok = [c for c in s["candidates"] if not c.get("rejected")
          and c.get("similarity") is not None]
    sim = np.array([c["similarity"] for c in ok])
    best = ok[int(np.argmax(sim))]
    comp = np.array(best["counts"])
    quantum = 100.0 / d_untreated

    print(f"{len(s['candidates'])} candidates, {len(ok)} scored, "
          f"{len(s['candidates']) - len(ok)} rejected")
    print(f"ladder (nA): {list(cur)}")
    print(f"reference   : {list(ref)}  total {ref.sum()}")
    print(f"variant     : {list(var)}  total {var.sum()}")
    print(f"compensated : {list(comp)}  total {comp.sum()}")
    print(f"untreated distance {d_untreated}; best distance {best['distance']}")
    print(f"restoration = (1 - {best['distance']}/{d_untreated}) x 100 = "
          f"{best['similarity']:.4f}%")
    print(f"one action potential is worth {quantum:.2f} percentage points, so the "
          f"supported precision is +/- {quantum/2:.1f} pp\n")
    ratios = {k: best['cond'][k] / BASE[k] for k in BASE}
    print("prescription, as a multiple of each baseline conductance:")
    for k, v in ratios.items():
        print(f"   {k:<6} x{v:.3f}   ({best['cond'][k]:.6g} against baseline "
              f"{BASE[k]:.6g})")
    for th in (80, 70, 60, 50):
        print(f"   candidates >= {th}%: {(sim >= th).sum()} of {len(ok)}")

    apply_style()
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.13})

    ax.plot(cur, ref, "-o", color=C.WT, label="reference (wild type)",
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax.plot(cur, var, "-s", color=C.VARIANT, label="variant (Kv7.2 D212G), untreated",
            markersize=4.0, markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    ax.plot(cur, comp, "--^", color=C.COMPENSATED,
            label="variant, compensated", markersize=4.4,
            markeredgecolor="white", markeredgewidth=0.7, zorder=6)
    ax.set_ylabel("action potentials in the 500 ms step (count)")
    ax.set_ylim(-1.5, 30)
    ax.legend(loc="upper left")
    ax.set_title("a   firing across the ladder", loc="left", color=C.INK)

    ax.text(0.985, 0.06,
            f"restoration {best['similarity']:.1f}%  "
            f"(distance {best['distance']} of {d_untreated};\n"
            f"one action potential = {quantum:.1f} percentage points)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=C.INK)

    axd.axhline(0, color=C.MUTED, linewidth=0.8, zorder=2)
    axd.plot(cur, var - ref, "-s", color=C.VARIANT, markersize=4.0,
             label="untreated − reference", zorder=4)
    axd.plot(cur, comp - ref, "--^", color=C.COMPENSATED, markersize=4.4,
             label="compensated − reference", zorder=5)
    axd.set_xlabel("injected current (nA)")
    axd.set_ylabel("difference from\nreference (count)")
    axd.legend(loc="upper right", ncol=2, fontsize=7.5)
    axd.set_title("b   what the intervention closes", loc="left", color=C.INK)

    fig.suptitle("Kv7.2 D212G: adjusting only the accessible conductances "
                 "restores the reference firing pattern",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.004,
             f"Multi-compartment CA1 pyramidal model, ModelDB 118986. "
             f"{len(ok):,} of {len(s['candidates']):,} candidates scored "
             f"({len(s['candidates']) - len(ok)} rejected as stiff or divergent), "
             f"drawn log-uniformly.\n"
             f"The mutated M-current is not in the search space; its maximal "
             f"conductance is identical in both arms (gbar = 0.0001 mho/cm² at "
             f"line 14 of both kmtwt.mod and kmquad.mod).\n"
             f"Prescription: sodium ×{ratios['gna']:.2f}, delayed-rectifier "
             f"potassium ×{ratios['gkdr']:.2f}, leak ×{ratios['g_pas']:.2f}. "
             f"Totals over the ladder: reference {ref.sum()}, untreated variant "
             f"{var.sum()}, compensated {comp.sum()} action potentials.",
             ha="left", va="top", fontsize=7.5, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.085, 1, 0.955))

    paths = save(fig, os.path.join(OUT, "kv7_2_restoration_ladder"))
    print("wrote " + ", ".join(paths))

    p = os.path.join(OUT, "kv7_2_restoration_ladder.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["injected_current_nA", "spikes_reference", "spikes_variant",
                    "spikes_compensated", "variant_minus_reference",
                    "compensated_minus_reference"])
        for i in range(len(cur)):
            w.writerow([cur[i], int(ref[i]), int(var[i]), int(comp[i]),
                        int(var[i] - ref[i]), int(comp[i] - ref[i])])
        w.writerow(["TOTAL", int(ref.sum()), int(var.sum()), int(comp.sum()),
                    int(d_untreated), int(best["distance"])])
    print(f"wrote {p}")

    with open(os.path.join(OUT, "kv7_2_restoration_summary.json"), "w") as f:
        json.dump({
            "restoration_pct": round(best["similarity"], 4),
            "restoration_reported": f"{best['similarity']:.1f}%",
            "precision_percentage_points_per_action_potential": round(quantum, 4),
            "distance_best": best["distance"], "distance_untreated": d_untreated,
            "prescription_multiples_of_baseline": {k: round(v, 4)
                                                   for k, v in ratios.items()},
            "prescription_absolute": best["cond"],
            "baselines": BASE,
            "counts": {"reference": ref.tolist(), "variant": var.tolist(),
                       "compensated": comp.tolist()},
            "currents_nA": cur.tolist(),
            "candidates_scored": len(ok),
            "candidates_rejected": len(s["candidates"]) - len(ok),
            "candidates_at_or_above": {str(t): int((sim >= t).sum())
                                       for t in (80, 70, 60, 50)},
            "source": "experiments/wo6/phase2/wo6b_taskD2_search.json",
        }, f, indent=2)
    print(f"wrote {os.path.join(OUT, 'kv7_2_restoration_summary.json')}")


if __name__ == "__main__":
    main()
