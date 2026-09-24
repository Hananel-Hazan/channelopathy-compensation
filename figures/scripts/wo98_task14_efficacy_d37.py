"""Case Study 1 (R859C): the efficacy distribution of every intervention tested.

Reads the stored per-test score array written by the original R859C search pipeline,
    external_data/sumary_explor_result100K.mutationFix.pbz2

Similarity scale, as in the original pipeline:
    similarity % = (1 - candidate distance / untreated variant's own distance) * 100
with everything below -100% discarded.

The denominator is the untreated variant's own spike-count distance to wild type.  It is
not stored in the saved candidate database, so it is re-measured and read here from
figures/similarity_denominator.json.  The plotted scale uses 37, the separation between
the search's own in-process reference and the variant, computed by
similarity_denominator_d37.py from the two ladders simulated in NEURON; 38, the separation of the archived reference ladders under
second-order integration, is reported as an alternative.  The summary JSON carries both.

Writes, in figures/:
    mutation1_efficacy_distribution.csv       one row per achievable spike-count distance
    mutation1_efficacy_summary.json           summary statistics for both denominators
    mutation1_efficacy_distribution.pdf/.png  the figure
"""
import bz2
import csv
import json
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCORES = os.path.join(REPO, "external_data", "sumary_explor_result100K.mutationFix.pbz2")
OUT = os.path.join(REPO, "figures")
DEN_SRC = os.path.join(REPO, "figures")   # where similarity_denominator.json is read from

# column layout of the stored rows
COL = {"g_leak": 0, "g_K": 1, "g_Na": 2, "spike_count_dist": 3, "dtw_dist": 4}


def load_scores():
    with bz2.BZ2File(SCORES, "r") as f:
        return pickle.load(f)


def stats(sim, denom_label):
    keep = sim >= -100.0
    s = sim[keep]
    q = np.percentile(s, [10, 25, 50, 75, 90, 99])
    return {
        "denominator": denom_label,
        "tests_simulated": int(sim.size),
        "tests_plotted": int(keep.sum()),
        "plotted_fraction_pct": round(100.0 * keep.mean(), 4),
        "max_similarity_pct": round(float(sim.max()), 4),
        "count_at_100pct": int((sim >= 99.999).sum()),
        "count_ge_97pct": int((sim >= 97).sum()),
        "count_ge_85pct": int((sim >= 85).sum()),
        "count_ge_60pct": int((sim >= 60).sum()),
        "count_ge_60pct_share": round(100.0 * (sim >= 60).mean(), 4),
        "median_plotted_pct": round(float(q[2]), 4),
        "p10_plotted_pct": round(float(q[0]), 4),
        "p25_plotted_pct": round(float(q[1]), 4),
        "p75_plotted_pct": round(float(q[3]), 4),
        "p90_plotted_pct": round(float(q[4]), 4),
        "p99_plotted_pct": round(float(q[5]), 4),
        "min_similarity_pct": round(float(sim.min()), 4),
        "count_below_floor": int((~keep).sum()),
    }


def main():
    with open(os.path.join(DEN_SRC, "similarity_denominator.json")) as f:
        den = json.load(f)
    # the plotted scale is the one the search itself divided by
    DENOM = float(den["computed_from_search_ladders"])     # 37, the search's own reference
    DENOM_ALT = float(den["value_pipeline_detector"])      # 38, the archived ladders

    sr = load_scores()
    print(f"selection criteria stored: {list(sr.keys())}\n")

    summary = {}
    for k1 in sr:
        cand = sr[k1]["score"]["candidates"]
        groups = sorted(cand)
        rows = np.concatenate([cand[i] for i in groups], axis=0)
        n_mtwt = len(sr[k1]["Number of MT-WT candidates"])
        n_mtmt = len(sr[k1]["Number of MT-MT candidates"])
        print(f"[{k1}] {len(groups)} groups, {rows.shape[0]:,} intervention tests, "
              f"MT-WT {n_mtwt}, MT-MT {n_mtmt}, interventions {n_mtwt*n_mtmt:,}")
        summary[k1] = {"groups": len(groups), "tests": int(rows.shape[0]),
                       "MT_WT": n_mtwt, "MT_MT": n_mtmt,
                       "interventions": n_mtwt * n_mtmt}

    # ---- the spike-count criterion (the plotted distribution) ----------------
    k1 = "Spike Count"
    cand = sr[k1]["score"]["candidates"]
    rows = np.concatenate([cand[i] for i in sorted(cand)], axis=0)
    dist = rows[:, COL["spike_count_dist"]]

    sim = (1.0 - dist / DENOM) * 100.0
    sim_alt = (1.0 - dist / DENOM_ALT) * 100.0
    st = stats(sim, f"{DENOM:g} (computed from the search ladders)")
    st_alt = stats(sim_alt, f"{DENOM_ALT:g} (re-measured, pipeline detector)")

    print(f"\n--- {k1} criterion, denominator {DENOM:g} ---")
    for k, v in st.items():
        print(f"  {k:<28} {v}")
    print(f"\n--- same, denominator {DENOM_ALT:g} (the alternative) ---")
    for k in ("tests_plotted", "plotted_fraction_pct", "max_similarity_pct",
              "count_ge_60pct", "median_plotted_pct"):
        print(f"  {k:<28} {st_alt[k]}")
    d = 100.0 * abs(st_alt["max_similarity_pct"] - st["max_similarity_pct"]) / \
        st["max_similarity_pct"]
    print(f"  => switching 37 -> 38 moves the maximum by {d:.3f}% relative")

    # ---- the time-warping criterion, reported but not plotted ---------------
    k2 = "Dynamic Time Warping"
    rows2 = np.concatenate([sr[k2]["score"]["candidates"][i]
                            for i in sorted(sr[k2]["score"]["candidates"])], axis=0)
    print(f"\n--- {k2} criterion: {rows2.shape[0]:,} tests simulated "
          f"(denominator not recoverable from the stored data) ---")

    # ---- binned distribution, written beside the figure ---------------------
    # The spike-count distance is an INTEGER number of misplaced action potentials,
    # so similarity is quantized in steps of 100/37 = 2.7027 percentage points.
    # Binning on a 1% grid would alias it; the natural bin is one integer distance.
    keep = sim >= -100.0
    dist_kept = dist[keep].astype(int)
    levels = np.arange(int(dist_kept.min()), int(dist_kept.max()) + 1)
    counts = np.array([(dist_kept == d).sum() for d in levels], dtype=int)
    sim_levels = (1.0 - levels / DENOM) * 100.0
    step_pct = 100.0 / DENOM

    csv_path = os.path.join(OUT, "mutation1_efficacy_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spike_count_distance_to_wild_type", "similarity_pct",
                    "n_intervention_tests", "share_of_plotted_pct",
                    "cumulative_tests_at_or_above_this_similarity"])
        order = np.argsort(-sim_levels)          # best similarity first
        cum = 0
        for i in order:
            cum += int(counts[i])
            w.writerow([int(levels[i]), round(float(sim_levels[i]), 4),
                        int(counts[i]),
                        round(100.0 * counts[i] / counts.sum(), 6), cum])
    print(f"\nwrote {csv_path}  ({len(levels)} achievable distances, "
          f"quantum {step_pct:.4f} percentage points)")

    with open(os.path.join(OUT, "mutation1_efficacy_summary.json"), "w") as f:
        json.dump({"criteria_present": summary,
                   "spike_count_denominator_38": st_alt,
                   "spike_count_denominator_37": st,
                   "dtw_tests_simulated": int(rows2.shape[0]),
                   "source_file": os.path.relpath(SCORES, REPO)}, f, indent=2)
    print(f"wrote {os.path.join(OUT, 'mutation1_efficacy_summary.json')}")

    # ---- figure --------------------------------------------------------------
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # Linear, not log: every bar lies within a single order of magnitude
    # (5,358 to 60,790 tests), so a log axis would flatten the only structure
    # the figure has to show.
    ax.bar(sim_levels, counts, width=step_pct * 0.82, color=C.VARIANT,
           linewidth=0, zorder=3)
    ax.set_xlim(-108, 108)
    top = counts.max() * 1.42
    ax.set_ylim(0, top)
    ax.set_xticks(np.arange(-100, 101, 25))
    ax.set_xlabel("similarity to wild type (%)   "
                  "— 0% = untreated variant, 100% = wild type")
    ax.set_ylabel("intervention tests (count)\n"
                  f"one bar per achievable value, {step_pct:.2f} pp apart")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _p: f"{int(v):,}"))

    # reference lines at 0% and at the -100% floor, labelled in the headroom
    ax.axvline(0, color=C.MUTED, linewidth=1.1, zorder=4)
    ax.text(-2.5, top * 0.985, "untreated variant (0%)", fontsize=7.5,
            color=C.MUTED, ha="right", va="top")
    ax.axvline(-100, color=C.MUTED, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
    ax.annotate(f"−100% floor\n{st['count_below_floor']:,} tests\ndiscarded below",
                xy=(-100, top * 0.62), xytext=(-88, top * 0.90),
                fontsize=7.5, color=C.MUTED, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=C.MUTED, linewidth=0.8))

    # the maximum, direct-labelled
    mx = st["max_similarity_pct"]
    ax.axvline(mx, color=C.WT, linewidth=1.1, zorder=4)
    ax.annotate(f"best reached {mx:.2f}%\nno test reaches 97%, none reaches 100%",
                xy=(mx, counts[np.argmax(sim_levels)] * 1.10),
                xytext=(mx - 6, top * 0.80), fontsize=8, color=C.WT,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=C.WT, linewidth=0.8))

    fig.suptitle("Case Study 1 (R859C): efficacy of every intervention tested",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.925,
             f"{st['tests_plotted']:,} plotted of {st['tests_simulated']:,} "
             f"simulated · median {st['median_plotted_pct']:.2f}% · "
             f"{st['count_ge_60pct']:,} tests ({st['count_ge_60pct_share']:.1f}%) "
             f"exceed 60% — there is no 60% ceiling",
             ha="left", va="top", fontsize=8.5, color=C.INK)
    fig.text(0.005, 0.005,
             "Spike-count selection criterion. 1,050 MT-WT candidates × 37 MT-MT "
             "configurations = 38,850 interventions,\n"
             "each applied to each of the 37 configurations. Both the additive and "
             "subtractive directions were attempted;\n"
             "the subtractive direction yielded no physically valid test in this run, "
             "so 1,437,450 tests were simulated.\n"
             "Similarity = (1 − candidate distance ÷ 37) × 100.",
             ha="left", va="top", fontsize=7.5, color=C.MUTED)

    paths = save(fig, os.path.join(OUT, "mutation1_efficacy_distribution"))
    print("wrote " + ", ".join(paths))


if __name__ == "__main__":
    main()
