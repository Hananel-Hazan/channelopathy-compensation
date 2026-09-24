"""Describe the firing patterns found by step4_full_run.py, and spot-check the parallel run.

Reads wo97_patterns.npz and reference_ladder.npy (written by step4_full_run.py),
figures/wo97_best_value_configurations.csv, and the archived wild-type ladder in
Neuron_Test/R859C/.  Prints, for each pattern, its size, its differences from the
reference ladder and the conductance range it spans; re-evaluates 200 random
configurations in this single process and compares them with the parallel run.
Writes wo97_patterns_described.json next to this script.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wo97_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
d = np.load(os.path.join(HERE, "wo97_patterns.npz"))
counts, dist, stored, gt = d["counts"], d["distance"], d["stored"], d["ground_truth_index"]
wt = np.load(os.path.join(HERE, "reference_ladder.npy"))
arch = L.archived("APthreshold WT-2005 - log every 10th data point.txt")
cur = L.CURRENTS
rows = list(csv.DictReader(open(os.path.join(REPO, "figures/wo97_best_value_configurations.csv"))))
gl = np.array([float(r["g_leak"]) for r in rows]); gk = np.array([float(r["g_K"]) for r in rows])
gn = np.array([float(r["g_Na"]) for r in rows])

pat, inv, mult = np.unique(counts, axis=0, return_inverse=True, return_counts=True)
order = np.argsort(-mult)
print(f"{len(counts):,} configurations, all with stored distance 2: {bool((stored==2).all())}")
print(f"all re-simulated distances equal 2: {bool((dist==2).all())}")
print(f"group sizes sum to the total: {int(mult.sum())}\n")
print(f"wild-type reference, the exploration's own in-process run: total {wt.sum()}")
print(f"archived wild-type text file                            : total {arch.sum()}")
print(f"the two differ only at {[cur[i] for i in np.nonzero(wt-arch)[0]]} pA "
      f"({[int(wt[i]) for i in np.nonzero(wt-arch)[0]]} against "
      f"{[int(arch[i]) for i in np.nonzero(wt-arch)[0]]})\n")

print("THE SIX PATTERNS")
print(f"{'#':>2} {'configs':>8} {'share':>7} {'total':>6} {'dist to ref':>11} {'dist to archived WT':>20}  differences from the reference ladder")
for rank, k in enumerate(order):
    df = pat[k] - wt
    nz = np.nonzero(df)[0]
    txt = ", ".join(f"{cur[i]} pA {int(df[i]):+d}" for i in nz)
    print(f"{rank:>2} {mult[k]:>8,} {100.0*mult[k]/len(counts):>6.2f}% {pat[k].sum():>6} "
          f"{int(np.abs(pat[k]-wt).sum()):>11} {int(np.abs(pat[k]-arch).sum()):>20}  {txt}")

print("\nconductance spread inside each pattern (the degeneracy the measurement is about)")
print(f"{'#':>2} {'configs':>8} {'g_leak min-max':>26} {'g_K min-max':>24} {'g_Na min-max':>24} {'ground truths':>14}")
for rank, k in enumerate(order):
    m = inv == k
    print(f"{rank:>2} {int(m.sum()):>8,} "
          f"{gl[m].min():>11.6g}-{gl[m].max():<11.6g}  {gk[m].min():>10.5g}-{gk[m].max():<10.5g}  "
          f"{gn[m].min():>10.5g}-{gn[m].max():<10.5g}  {len(set(gt[m].tolist())):>10}")

# ---- spot check: 200 configurations re-evaluated in fresh processes ----------
print("\nspot check: 200 configurations re-evaluated, each in a process that has done"
      "\nnothing else, against what the 32-worker run recorded")
idx = np.random.default_rng(20260907).choice(len(rows), 200, replace=False)
h = L.start()
wt2, _ = L.exploration_wt(h)
assert np.array_equal(wt2, wt), "reference ladder differs from the parallel run"
bad = 0
for i in idx:
    r = rows[i]
    c, _ = L.exploration_mt(h, (float(r["g_leak"]), float(r["g_K"]), float(r["g_Na"])))
    if not np.array_equal(c, counts[i]):
        bad += 1
print(f"  {200-bad} of 200 identical, {bad} differ -- {'PASSED' if bad==0 else 'FAILED'}")

json.dump({"spot_check_n": 200, "spot_check_mismatches": int(bad),
           "reference_ladder_total": int(wt.sum()),
           "archived_ladder_total": int(arch.sum()),
           "reference_vs_archived_differing_levels_pA": [cur[i] for i in np.nonzero(wt-arch)[0]],
           "patterns": [{"rank": r, "configurations": int(mult[k]),
                         "share_pct": round(100.0*mult[k]/len(counts), 3),
                         "ladder_total": int(pat[k].sum()),
                         "distance_to_reference": int(np.abs(pat[k]-wt).sum()),
                         "distance_to_archived_wild_type": int(np.abs(pat[k]-arch).sum()),
                         "difference_from_reference": {str(cur[i]): int((pat[k]-wt)[i])
                                                       for i in np.nonzero(pat[k]-wt)[0]},
                         "g_leak_range": [float(gl[inv==k].min()), float(gl[inv==k].max())],
                         "g_K_range": [float(gk[inv==k].min()), float(gk[inv==k].max())],
                         "g_Na_range": [float(gn[inv==k].min()), float(gn[inv==k].max())],
                         "ground_truths": sorted(map(int, set(gt[inv==k].tolist())))}
                        for r, k in enumerate(order)]},
          open(os.path.join(HERE, "wo97_patterns_described.json"), "w"), indent=1)
print("\nwrote wo97_patterns_described.json")
