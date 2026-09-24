"""Plot the stability window of the compensating configuration.

Efficacy against perturbation magnitude, where the perturbation is a fraction of
each optimized conductance, in three modes: one conductance at a time, joint
independent draws (median and 5th percentile), and the exhaustive set of sign
corners (worst combination of directions).  The 80 %-of-unperturbed criterion is
marked.  Perturbation is applied only to the pharmacologically accessible
conductances, never to the mutated channel, which is not being dosed.

Panel a is the multi-compartment model (ModelDB 118986); panel b is the
single-compartment differentiable model on the same axes.

Two different quantities should not be confused: the two models' sensitivity to
the same kinetic shift (the single-compartment model loses 459 of 470 action
potentials where the multi-compartment model loses 38 of 498, roughly an order of
magnitude), and the ratio of their stability windows shown here (about two to
three times).

Inputs (stored results; nothing is simulated here):
  multi-compartment  experiments/wo6/phase2/wo6b_taskE_stability_<host>.json
                     (two result files, pooled)
  single-compartment experiments/wo5/task11d_summary.json
Outputs, in figures/:
  stability_window_fractional.{pdf,png}, stability_window_fractional.csv,
  stability_window_summary.json
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
W5 = os.path.join(REPO, "experiments", "wo5")
OUT = os.path.join(REPO, "figures")

MODES = [("one_at_a_time", "one conductance at a time", C.M, "-", "o"),
         ("joint_median", "joint draws, median", C.WT, "-", "s"),
         ("joint_p5", "joint draws, 5th percentile", C.N, "--", "^"),
         ("worst_case", "worst combination of directions", C.VARIANT, "-", "D")]


def multi_compartment():
    """Reduce the stored per-draw records to one row per perturbation magnitude,
    as experiments/wo6/phase2/wo6b_analyse.py does for the stability data.

    For each magnitude: the minimum over one-at-a-time perturbations, the median
    and 5th percentile of joint draws, and the minimum over sign corners and
    one-at-a-time perturbations (worst case).  Returns magnitudes in percent,
    the rows keyed by fractional magnitude, and the metadata of the last file."""
    recs = []
    for h in ("biohours", "hoursone"):
        d = json.load(open(os.path.join(P2, f"wo6b_taskE_stability_{h}.json")))
        recs += d["results"]
        meta = d
    deltas = sorted({r["delta"] for r in recs})
    rows = {}
    for dl in deltas:
        sub = [r for r in recs if r["delta"] == dl]

        def eff(mode):
            return np.array([r["efficacy_pct"] for r in sub
                             if r["mode"] == mode and r["efficacy_pct"] is not None])
        one, joint, worst = eff("one_at_a_time"), eff("joint"), eff("worst_case")
        rows[dl] = dict(
            one_at_a_time=float(np.nanmin(one)) if one.size else np.nan,
            joint_median=float(np.nanmedian(joint)) if joint.size else np.nan,
            joint_p5=float(np.nanpercentile(joint, 5)) if joint.size else np.nan,
            worst_case=float(np.nanmin(np.concatenate([worst, one]))),
            n_joint=int(joint.size), n_one=int(one.size), n_worst=int(worst.size))
    return [100 * d for d in deltas], rows, meta


def first_at_or_below(mags, vals, thr):
    for m, v in zip(mags, vals):
        if v <= thr:
            return m
    return None


def main():
    mags_m, rows_m, meta = multi_compartment()
    print(f"multi-compartment: optimum {meta['optimum']}, "
          f"similarity at optimum {meta['similarity_at_optimum']:.2f} %, "
          f"untreated distance {meta['d_untreated']}")
    quantum = 100.0 / meta["d_untreated"] / meta["similarity_at_optimum"] * 100
    print(f"one action potential is worth {quantum:.1f} points of efficacy, so "
          f"these curves are quantized in roughly {quantum:.0f}-point steps\n")

    print(f"{'delta':>7} " + " ".join(f"{m[1][:14]:>15}" for m in MODES))
    for d in mags_m:
        r = rows_m[d / 100]
        print(f"{d:>6.0f}% " + " ".join(f"{r[m[0]]:>15.0f}" for m in MODES))

    d5 = json.load(open(os.path.join(W5, "task11d_summary.json")))
    cases = {c["name"]: c for c in d5["cases"]}
    sc_name = "potassium variant, g_K fixed at 20"
    sc = cases[sc_name]
    mags_s = sc["mags_pct"]
    print(f"\nsingle-compartment comparator: {sc_name}")
    print(f"   optimum {['%.5g' % x for x in sc['optimum']]}, "
          f"similarity {sc['similarity_optimum']:.2f} %, "
          f"untreated distance {sc['d_untreated']}")

    print(f"\nsmallest perturbation at which efficacy first falls to or below 80 %:")
    print(f"   {'mode':<34} {'multi-compartment':>18} {'single-compartment':>19}")
    tbl = {}
    for key, lbl, *_ in MODES:
        vm = [rows_m[d / 100][key] for d in mags_m]
        skey = {"one_at_a_time": "one", "joint_median": "joint_median",
                "joint_p5": "joint_p5", "worst_case": "worst"}[key]
        vs = sc["curves"][skey]
        a = first_at_or_below(mags_m, vm, 80)
        b = first_at_or_below(mags_s, vs, 80)
        tbl[key] = (a, b)
        ra = f"{a:.0f} %" if a else "> 20 %"
        rb = f"{b:.0f} %" if b else "> 50 %"
        print(f"   {lbl:<34} {ra:>18} {rb:>19}")

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)

    for ax, mags, getter, title, note in (
            (axa, mags_m, lambda k: [rows_m[d / 100][k] for d in mags_m],
             "a   multi-compartment model (ModelDB 118986)",
             "multi-compartment NEURON model"),
            (axb, mags_s, lambda k: sc["curves"][
                {"one_at_a_time": "one", "joint_median": "joint_median",
                 "joint_p5": "joint_p5", "worst_case": "worst"}[k]],
             "b   single-compartment differentiable model",
             "single-compartment model")):
        ax.axhspan(-160, 80, color=C.MUTED, alpha=0.06, linewidth=0, zorder=0)
        ax.axhline(80, color=C.INK, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
        ax.axhline(0, color=C.MUTED, linewidth=0.9, zorder=3)
        for key, lbl, col, ls, mk in MODES:
            ax.plot(mags, getter(key), ls, color=col, marker=mk, markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.5,
                    linewidth=1.5, label=lbl)
        ax.set_xlim(0, 21)
        ax.set_ylim(-160, 118)
        ax.set_xlabel("perturbation of each optimized conductance (± %)")
        ax.set_title(title, loc="left", color=C.INK)
        ax.text(20.6, 84, "80 % of unperturbed efficacy", ha="right", va="bottom",
                fontsize=7.5, color=C.INK)
        ax.text(20.6, -152, note, ha="right", va="bottom", fontsize=7.5,
                color=C.MUTED)
    axa.set_ylabel("efficacy retained (% of unperturbed)")
    axa.legend(loc="lower left", fontsize=7.5)

    fig.suptitle("Stability window: how much dosing error the compensating "
                 "configuration tolerates", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.004,
             f"Perturbation is a fraction of each optimized conductance, applied "
             f"only to the pharmacologically accessible ones — never to the mutated "
             f"channel, which is not being dosed. Shaded region is below the "
             f"80 % criterion;\n"
             f"below the solid rule the treated cell is worse than no treatment at "
             f"all. Multi-compartment: {meta['draws']} joint draws per magnitude, "
             f"one-at-a-time and sign corners exhaustive, "
             f"{meta['similarity_at_optimum']:.1f} % efficacy at the optimum.\n"
             f"Single-compartment comparator is the potassium variant with g_K held "
             f"at 20. These curves are quantized: one action potential is "
             f"worth about {quantum:.0f} points of efficacy on the multi-compartment "
             f"ladder.",
             ha="left", va="top", fontsize=7, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.115, 1, 0.94))
    # "ModelDB 118986" in panel a's title is an accession number, not a
    # quantity; suppress the style check's report for it
    paths = save(fig, os.path.join(OUT, "stability_window_fractional"),
                 allow=("ModelDB 118986",))
    print("\nwrote " + ", ".join(paths))

    p = os.path.join(OUT, "stability_window_fractional.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "perturbation_pct", "mode",
                    "efficacy_retained_pct_of_unperturbed"])
        for d in mags_m:
            for key, lbl, *_ in MODES:
                w.writerow(["multi-compartment (ModelDB 118986)", d, lbl,
                            round(rows_m[d / 100][key], 4)])
        skey = {"one_at_a_time": "one", "joint_median": "joint_median",
                "joint_p5": "joint_p5", "worst_case": "worst"}
        for i, d in enumerate(mags_s):
            for key, lbl, *_ in MODES:
                w.writerow(["single-compartment differentiable", d, lbl,
                            round(sc["curves"][skey[key]][i], 4)])
    print(f"wrote {p}")

    with open(os.path.join(OUT, "stability_window_summary.json"), "w") as f:
        json.dump({
            "criterion": "efficacy at or below 80 % of unperturbed",
            "first_perturbation_at_or_below_80pct": {
                lbl: {"multi_compartment_pct": tbl[key][0],
                      "single_compartment_pct": tbl[key][1]}
                for key, lbl, *_ in MODES},
            "multi_compartment": {
                "model": "ModelDB 118986, CA1 pyramidal, Kv7.2 D212G",
                "similarity_at_optimum_pct": meta["similarity_at_optimum"],
                "d_untreated": meta["d_untreated"],
                "joint_draws_per_magnitude": meta["draws"],
                "efficacy_points_per_action_potential": round(quantum, 3),
                "curves": {str(d): rows_m[d / 100] for d in mags_m}},
            "single_compartment": {
                "case": sc_name, "similarity_at_optimum_pct": sc["similarity_optimum"],
                "d_untreated": sc["d_untreated"], "mags_pct": mags_s,
                "curves": sc["curves"]}}, f, indent=2)
    print(f"wrote {os.path.join(OUT, 'stability_window_summary.json')}")


if __name__ == "__main__":
    main()
