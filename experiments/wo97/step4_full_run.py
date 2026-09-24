"""Re-simulate the 10,780 best-value configurations of the R859C search on the NEURON
model and count how many distinct per-level firing patterns they produce.

Protocol: the "exploration" protocol of wo97_lib.py (the one the original search ran,
loading Neuron_Test/R859C/neuron.hoc).  Each worker process loads NEURON once,
simulates the wild-type reference ladder once in its own process, then evaluates its
share of the configurations.  Every evaluation rebuilds all 35 cells (h.initMT()), so
no state crosses between configurations.

Reads figures/wo97_best_value_configurations.csv (columns g_leak, g_K, g_Na,
stored_spike_count_distance, ground_truth_index).  First checks that every
configuration reproduces its stored spike-count distance; only if all do, it counts
the distinct firing patterns.

Writes reference_ladder.npy, wo97_patterns.npz and wo97_measurement.json next to this
script and prints the measurement.

Usage:  python step4_full_run.py [--workers N] [--limit N]
"""
import argparse, csv, os, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(REPO, "figures/wo97_best_value_configurations.csv")

_H = None
_L = None
_WT = None
_TRIPLES = None


def init(triples):
    global _H, _L, _WT, _TRIPLES
    sys.path.insert(0, HERE)
    import wo97_lib as L
    _L = L
    _H = L.start()
    _WT, _ = L.exploration_wt(_H)
    _TRIPLES = triples


def reference():
    return _WT


def work(i):
    c, _ = _L.exploration_mt(_H, _TRIPLES[i])
    return i, c.astype(np.int16), int(np.abs(_WT - c).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV)))
    if a.limit:
        rows = rows[:a.limit]
    triples = [(float(r["g_leak"]), float(r["g_K"]), float(r["g_Na"])) for r in rows]
    stored = np.array([int(r["stored_spike_count_distance"]) for r in rows])
    gt = np.array([int(r["ground_truth_index"]) for r in rows])
    n = len(rows)
    print(f"{n:,} configurations, {a.workers} worker processes")

    # the wild-type reference ladder each worker will use, recorded once here
    sys.path.insert(0, HERE)
    ref_path = os.path.join(HERE, "reference_ladder.npy")

    counts = np.zeros((n, 35), dtype=np.int16)
    dist = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(triples,)) as p:
        np.save(ref_path, p.apply(reference))
        for i, c, d in p.imap_unordered(work, range(n), chunksize=16):
            counts[i] = c; dist[i] = d; done += 1
            if done % 1000 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{n:,}  {el:.0f} s elapsed, {el/done*(n-done):.0f} s left",
                      flush=True)
    el = time.time() - t0
    print(f"finished in {el:.0f} s ({el/n*1000:.0f} ms per configuration)\n")

    out = os.path.join(HERE, "wo97_patterns.npz")
    np.savez_compressed(out, counts=counts, distance=dist, stored=stored,
                        ground_truth_index=gt)
    print("wrote", out)

    # ---- reproduction check: every configuration must return the stored distance
    agree = int((dist == stored).sum())
    print(f"\nREPRODUCTION: {agree:,} of {n:,} return the stored distance "
          f"of 2 exactly -- {'PASSED' if agree == n else 'FAILED'}")
    if agree != n:
        bad = np.where(dist != stored)[0]
        vals, cts = np.unique(dist[bad], return_counts=True)
        print(f"  {len(bad):,} disagree; distances returned: "
              + ", ".join(f"{v} x{c:,}" for v, c in zip(vals, cts)))
        return

    # ---- the measurement -----------------------------------------------------
    import wo97_lib as L                      # reference ladder, for describing patterns
    wt = np.load(os.path.join(HERE, "reference_ladder.npy"))
    pat, inv, mult = np.unique(counts, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-mult)
    cur = L.CURRENTS
    print("\nTHE MEASUREMENT")
    print(f"  distinct per-level firing patterns : {len(pat):,}")
    print(f"  largest group sharing one pattern  : {mult.max():,} "
          f"({100.0*mult.max()/n:.1f} % of the {n:,})")
    print(f"  patterns occurring only once       : {int((mult==1).sum()):,}")
    print(f"  median group size                  : {int(np.median(mult))}")
    print(f"\n  every pattern, largest group first."
          f"  'differs from wild type at' lists (injected current in pA, spikes above"
          f" or below the wild-type count):")
    for k in order:
        d = pat[k] - wt
        nz = np.nonzero(d)[0]
        where = ", ".join(f"{cur[i]} pA {int(d[i]):+d}" for i in nz)
        print(f"    {mult[k]:>7,} configurations ({100.0*mult[k]/n:5.2f} %)  differs from wild type at {where}")
    print("\n  patterns per ground truth:")
    for g in sorted(set(gt.tolist())):
        m = gt == g
        sub = np.unique(inv[m])
        print(f"    ground truth {g:>2}: {int(m.sum()):>6,} configurations, "
              f"{len(sub):>3} distinct patterns  (pattern ids {sorted(map(int, sub))})")
    json_out = os.path.join(HERE, "wo97_measurement.json")
    import json
    json.dump({
        "configurations": int(n),
        "wild_type_reference_ladder": [int(x) for x in wt],
        "wild_type_reference_total": int(wt.sum()),
        "currents_pA": cur,
        "distinct_patterns": int(len(pat)),
        "largest_group": int(mult.max()),
        "largest_group_share_pct": round(100.0*mult.max()/n, 3),
        "patterns_occurring_once": int((mult == 1).sum()),
        "multiplicity_distribution": sorted((int(m) for m in mult), reverse=True),
        "patterns": [{"count": int(mult[k]), "ladder": [int(x) for x in pat[k]],
                      "difference_from_wild_type": {str(cur[i]): int((pat[k]-wt)[i])
                                                    for i in np.nonzero(pat[k]-wt)[0]}}
                     for k in order],
        "per_ground_truth": {str(g): {"configurations": int((gt == g).sum()),
                                      "distinct_patterns": int(len(np.unique(inv[gt == g])))}
                             for g in sorted(set(gt.tolist()))},
    }, open(json_out, "w"), indent=1)
    print("\nwrote", json_out)


if __name__ == "__main__":
    main()
