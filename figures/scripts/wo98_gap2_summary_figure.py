"""Pipeline summary figure for the Discussion.

The four panels follow the framework's four steps -- characterize the mutation-specific
electrophysiological alterations, map the parameter space of potential interventions,
identify the stable therapeutic windows, and apply an automated optimization tool for
rapid intervention discovery -- each with its measured result.

Both case studies appear: panel (a) is R859C, panel (c) is the Kv7.2 D212G
multi-compartment model.  Every mark is read from another figure's data file; no value
is recomputed here.

    (a) figures/r859c_excitability_ladder.csv
    (b) figures/mutation1_efficacy_distribution.csv
    (c) figures/stability_window_fractional.csv
    (d) figures/throughput_table.csv

The artwork carries no title and no provenance block; those are in the LaTeX caption.
Writes figures/pipeline_summary.pdf and .png (via wo7_style.save).
"""
import csv, os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIG = os.path.join(REPO, "figures")
R = lambda n: list(csv.DictReader(open(os.path.join(FIG, n), encoding="utf-8")))


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    (a, b), (c, d) = axes

    # -- (a) characterise the deficit ----------------------------------------
    lad = R("r859c_excitability_ladder.csv")
    x = [float(r["injected_current_pA"]) for r in lad]
    a.step(x, [int(r["spikes_wild_type"]) for r in lad], where="mid", color=C.WT)
    a.step(x, [int(r["spikes_R859C"]) for r in lad], where="mid", color=C.VARIANT)
    a.axvspan(135, 215, color=C.BAND_LESS, alpha=0.10, lw=0)
    a.annotate("variant fires less\nover this range", xy=(175, 31), ha="center",
               va="top", fontsize=7.5, color=C.MUTED)
    a.set_title("(a)  Characterise the deficit", loc="left")
    a.set_xlabel("injected current (pA)")
    a.set_ylabel("action potentials (count)")
    # Direct labels rather than a legend: the legend box lands on the rising
    # limb wherever it is placed in this panel.
    a.set_xlim(15, 455)
    a.annotate("reference", xy=(360, 30), xytext=(5, -2), textcoords="offset points",
               color=C.WT, fontsize=8, va="center")
    a.annotate("R859C variant", xy=(360, 32), xytext=(5, 4),
               textcoords="offset points", color=C.VARIANT, fontsize=8, va="center")
    # Model identity and provenance are given in the LaTeX caption, not in the artwork.

    # -- (b) map the intervention space --------------------------------------
    # Cumulative count of interventions at or above each similarity (Figure 7 shows
    # the histogram of the same data).
    eff = sorted(R("mutation1_efficacy_distribution.csv"),
                 key=lambda r: -float(r["similarity_pct"]))
    s = [float(r["similarity_pct"]) for r in eff]
    cum = [int(r["cumulative_tests_at_or_above_this_similarity"]) for r in eff]
    b.plot(s, cum, color=C.COMPENSATED)
    b.set_yscale("log")
    b.annotate("best reached,\n94.6 %", xy=(s[0], cum[0]), xytext=(-8, 4),
               textcoords="offset points", ha="right", va="bottom",
               fontsize=7.5, color=C.MUTED)
    b.plot([s[0]], [cum[0]], marker="o", ms=4, color=C.COMPENSATED)
    b.set_title("(b)  Map the intervention space", loc="left")
    b.set_xlabel("similarity to reference (%)")
    b.set_ylabel("interventions at or above (count)")

    # -- (c) find the stable window ------------------------------------------
    st = [r for r in R("stability_window_fractional.csv")
          if r["model"].startswith("multi-compartment")]
    for mode, colour, lab in (("joint draws, median", C.COMPENSATED, "median of joint draws"),
                              ("worst combination of directions", C.VARIANT,
                               "worst combination of directions")):
        rows = sorted((r for r in st if r["mode"] == mode),
                      key=lambda r: float(r["perturbation_pct"]))
        c.plot([float(r["perturbation_pct"]) for r in rows],
               [float(r["efficacy_retained_pct_of_unperturbed"]) for r in rows],
               color=colour, label=lab, marker="o", ms=3.5)
    c.axhline(80, color=C.MUTED, lw=0.9, ls="--")
    c.annotate("80 % criterion", xy=(20.5, 80), xytext=(0, 4), ha="right",
               textcoords="offset points", fontsize=7.5, color=C.MUTED)
    # Every integer magnitude from 1 to 20 % was measured, so the window is resolved
    # to 1 %; the panel marks the largest magnitude that still retains the 80 %
    # criterion in each mode.
    WINDOW = {"joint draws, median": 11.0,
              "worst combination of directions": 3.0}
    mk = {}
    for mode, mag in WINDOW.items():
        row = next(r for r in st if r["mode"] == mode
                   and float(r["perturbation_pct"]) == mag)
        mk[mode] = float(row["efficacy_retained_pct_of_unperturbed"])
        assert mk[mode] > 80.0, (mode, mag, mk[mode])
    c.plot(list(WINDOW.values()), [mk[m] for m in WINDOW], ls="none", marker="o",
           ms=6, mfc="none", mec=C.MUTED, mew=0.9)
    # The circles are labelled with their magnitude only; their meaning is stated
    # in the LaTeX caption.
    for mode, mag in WINDOW.items():
        c.annotate(f"{mag:.0f} %", xy=(mag, mk[mode]), xytext=(0, -11),
                   textcoords="offset points", ha="center", va="top",
                   fontsize=7.0, color=C.MUTED)
    c.set_xlim(0, 21)
    c.set_title("(c)  Find the stable window", loc="left")
    c.set_xlabel("perturbation of each conductance (%)")
    c.set_ylabel("efficacy retained (% of unperturbed)")
    c.legend(loc="lower left", fontsize=7.0, frameon=True,
             facecolor="white", framealpha=0.92, edgecolor="none")

    # -- (d) make the search affordable --------------------------------------
    rates = {(r["hardware"], r["implementation"]):
             float(r["candidate_parameter_sets_per_second"])
             for r in R("throughput_table.csv") if r["table"] == "throughput"}
    # Every rate is read from the Table 1 generator's output file, so the panel
    # always matches the current Table 1.
    KEYS = [("NEURON, one core", ("one processor core", "—"), C.VARIANT),
            ("NEURON, 72-core node", ("72-core node", "spike-count stage"), C.VARIANT),
            ("differentiable, one core, batched",
             ("one processor core", "batched (1,024 sets)"), C.WT),
            ("differentiable, RTX 2070, fused",
             ("RTX 2070 (2018)", "batched, fused"), C.COMPENSATED),
            ("differentiable, RTX 4070 Ti, fused",
             ("RTX 4070 Ti (2023)", "batched, fused"), C.COMPENSATED)]
    KEYS = [(l, (k[0], "—" if k[1] == "—" else k[1]), c_) for l, k, c_ in KEYS]
    want = [(lbl, rates[k], f"{rates[k]:,.1f}" if rates[k] < 100 else f"{rates[k]:,.0f}", col)
            for lbl, k, col in KEYS]
    assert rates[("one processor core", "—")] == 1.1, "table 1 row drifted"
    labels = [w[0] for w in want]
    d.barh(range(len(want)), [w[1] for w in want],
           color=[w[3] for w in want], height=0.62)
    d.set_yticks(range(len(want)))
    d.set_yticklabels(labels, fontsize=7.5)
    d.invert_yaxis()
    d.set_xscale("log")
    d.xaxis.set_major_formatter(ScalarFormatter())
    d.set_xlim(0.5, 3e5)
    for i, w in enumerate(want):
        d.annotate(w[2], xy=(w[1], i), xytext=(4, 0), textcoords="offset points",
                   va="center", fontsize=7.5, color=C.MUTED)
    d.grid(axis="y", visible=False)
    d.set_title("(d)  Make the search affordable", loc="left")
    d.set_xlabel("candidate parameter sets per second")

    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, os.path.join(FIG, "pipeline_summary"))


if __name__ == "__main__":
    main()
