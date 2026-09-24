"""Dynamic-time-warping (DTW) similarity denominator for the R859C search.

The original search pipeline scores a candidate intervention by

    similarity % = (1 - candidate_DTW / denominator) * 100

where candidate_DTW is the summed DTW distance between the wild-type traces and
the candidate's traces, and the denominator is the same quantity computed for the
untreated variant:

    sum over the 35 current levels of |DTW(wild-type trace, untreated variant trace)|

This script computes the denominator from the two archived NEURON trace files,
using `dtaidistance.dtw.distance_fast` with `use_pruning=True` as the pipeline
does.  It then:
  * cross-checks it against the candidate database
    (external_data/sumary_explor_result.100K.pbz2), including the alternative
    "topology" scale that divides by the appended worst-case row instead;
  * converts the stored intervention scores
    (external_data/sumary_explor_result100K.mutationFix.pbz2) to similarity
    percentages for both selection criteria.

In the stored intervention-score rows the DTW distance is column 4; the pipeline
deletes the spike-count column before computing similarity, so it reads the DTW
distance from column 3.

Writes figures/dtw_denominator.json and figures/dtw_denominator_per_level.csv.
Requires the `dtaidistance` package.
"""
import bz2
import csv
import json
import os
import pickle
import time

import numpy as np
from dtaidistance import dtw

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
FILES = {"WT": os.path.join(ARCH, "APthreshold WT-2005 - log every 10th data point.txt"),
         "MT": os.path.join(ARCH, "APthreshold R859C - log every 10th data point.txt")}
SCORES = os.path.join(REPO, "external_data", "sumary_explor_result100K.mutationFix.pbz2")
DB = os.path.join(REPO, "external_data", "sumary_explor_result.100K.pbz2")
OUT = os.path.join(REPO, "figures")


def load(path):
    """Read a tab-separated trace file: returns t (ms), currents (pA), and
    voltages (mV) as a C-contiguous float64 array of shape (n_currents, n_time),
    the layout dtw.distance_fast requires."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        f.readline()
        rows = [[float(p) for p in line.rstrip("\n").split("\t") if p != ""]
                for line in f if line.strip()]
    data = np.array(rows)
    currents = np.array([int(h.split()[0]) for h in header[1:] if h.strip()])
    return data[:, 0], currents, np.ascontiguousarray(data[:, 1:].T, dtype=np.double)


def main():
    t, cur, v_wt = load(FILES["WT"])
    _, cur2, v_mt = load(FILES["MT"])
    assert (cur == cur2).all()
    print(f"dtaidistance {dtw.__name__} loaded")
    print(f"traces: {v_wt.shape[0]} levels x {v_wt.shape[1]} samples, "
          f"t = {t[0]:.1f}..{t[-1]:.1f} ms\n")

    # ---- the denominator, computed as the pipeline's compute_DTW does ---------
    t0 = time.time()
    per_level = np.empty(v_wt.shape[0])
    for i in range(v_wt.shape[0]):
        per_level[i] = abs(dtw.distance_fast(np.ascontiguousarray(v_wt[i]),
                                             np.ascontiguousarray(v_mt[i]),
                                             use_pruning=True))
    denom = float(per_level.sum())
    print(f"per-level DTW(wild type, untreated variant):")
    for c, d in zip(cur, per_level):
        print(f"   {c:>4} pA   {d:12.4f}")
    print(f"\n  DENOMINATOR = sum over 35 levels = {denom:.4f}"
          f"   ({time.time() - t0:.1f} s)")

    # ---- cross-check against the candidate database ------------------------
    with bz2.BZ2File(DB, "r") as f:
        db = pickle.load(f)
    best_mtwt = float(db["Dynamic Time Warping"]["MT"]["WT"][:-1, 3].min())
    ref_row = float(db["Dynamic Time Warping"]["MT"]["WT"][-1, 3])
    print(f"\n  cross-checks against the candidate database:")
    print(f"    best MT-WT candidate DTW distance      {best_mtwt:12.4f}"
          f"   -> similarity {(1 - best_mtwt / denom) * 100:7.3f}%")
    print(f"    appended worst-ever row (topology denom){ref_row:12.4f}")
    print(f"    ratio worst-ever / untreated            {ref_row / denom:7.3f}x")

    # ---- apply it to the stored intervention scores -------------------------
    with bz2.BZ2File(SCORES, "r") as f:
        sr = pickle.load(f)

    results = {}
    for crit in ("Dynamic Time Warping", "Spike Count"):
        cand = sr[crit]["score"]["candidates"]
        rows = np.concatenate([cand[i] for i in sorted(cand)], axis=0)
        d = rows[:, 4]                       # column 4 = DTW distance to wild type
        sim = (1.0 - d / denom) * 100.0
        keep = sim >= -100.0
        s = sim[keep]
        q = np.percentile(s, [10, 25, 50, 75, 90, 99])
        res = {
            "criterion": crit,
            "tests_simulated": int(sim.size),
            "tests_plotted": int(keep.sum()),
            "plotted_fraction_pct": round(100.0 * keep.mean(), 4),
            "max_similarity_pct": round(float(sim.max()), 4),
            "min_similarity_pct": round(float(sim.min()), 4),
            "count_at_100pct": int((sim >= 99.999).sum()),
            "count_ge_995pct": int((sim >= 99.5).sum()),
            "count_ge_97pct": int((sim >= 97).sum()),
            "count_ge_85pct": int((sim >= 85).sum()),
            "count_ge_60pct": int((sim >= 60).sum()),
            "median_plotted_pct": round(float(q[2]), 4),
            "p10_plotted_pct": round(float(q[0]), 4),
            "p90_plotted_pct": round(float(q[4]), 4),
            "p99_plotted_pct": round(float(q[5]), 4),
        }
        results[crit] = res
        print(f"\n--- DTW similarity, {crit} selection criterion ---")
        for k, v in res.items():
            print(f"   {k:<24} {v}")

    # ---- compare the efficacy scale with the topology scale -----------------
    a = db["Dynamic Time Warping"]["MT"]["WT"]
    dd = a[:-1, 3]
    print(f"\n--- the two DTW scales, on the candidate table "
          f"({dd.size:,} rows) ---")
    for name, den in (("efficacy  (divide by untreated variant)", denom),
                      ("topology  (divide by appended worst row)", ref_row)):
        sim = (1 - dd / den) * 100
        print(f"   {name}: best {sim.max():7.3f}%   "
              f">=99.5%: {int((sim >= 99.5).sum()):>7,}   "
              f">=97%: {int((sim >= 97).sum()):>7,}")

    out = {
        "measure": "Dynamic Time Warping",
        "quantity": "observation_summary_statistics_MT['Dynamic Time Warping WT']",
        "definition": "sum over the 35 injected-current levels of "
                      "dtw.distance_fast(wild-type trace, untreated variant trace), "
                      "use_pruning=True",
        "value": round(denom, 4),
        "per_level": {int(c): round(float(d), 4) for c, d in zip(cur, per_level)},
        "source_lines": "R859C search pipeline DTW scoring",
        "traces": {k: os.path.relpath(v, REPO) for k, v in FILES.items()},
        "cross_checks": {
            "best_MT_WT_candidate_distance": round(best_mtwt, 4),
            "best_MT_WT_candidate_similarity_pct": round((1 - best_mtwt / denom) * 100, 4),
            "appended_worst_row_topology_denominator": round(ref_row, 4),
            "ratio_topology_to_efficacy": round(ref_row / denom, 4),
        },
        "intervention_scores": results,
    }
    p = os.path.join(OUT, "dtw_denominator.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")

    p = os.path.join(OUT, "dtw_denominator_per_level.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["injected_current_pA", "dtw_wildtype_vs_untreated_variant"])
        for c, d in zip(cur, per_level):
            w.writerow([int(c), round(float(d), 6)])
        w.writerow(["TOTAL", round(denom, 6)])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
