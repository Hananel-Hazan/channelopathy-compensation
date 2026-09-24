"""Compensating-conductance search on the NEURON Kv7.2 D212G CA1 cell model.

This is the R859C search pipeline's analysis, run on the Kv7.2 case study so the two are
comparable: sample candidate conductance sets from a box, score every one on
action-potential count across a ladder of injected currents, and express the score on the
pipeline's percentage scale, where 100 % is the wild type and 0 % is the untreated
mutant.

**Where this departs from the R859C pipeline, and why.**

1. **Twelve injected-current levels, not thirty-five, and not evenly spaced.** One 500 ms
   run of this 140-section cell under variable-step integration costs about 15 seconds,
   against microseconds for the single-compartment differentiable model; thirty-five
   levels would cost 9 minutes per candidate. A uniform 0.1-1.0 nA ladder separates
   mutant from wild type by only 15 action potentials, so the percentage scale would move
   in steps of 6.7 points. The separation sits between 0.44 and 0.52 nA, just above the
   wild type's rheobase, so the ladder is refined there and left coarse elsewhere.
2. **Log-uniform sampling, not uniform.** On the single-compartment model
   (wo6_taskB.py), a million candidates drawn uniformly across the pipeline's box fail
   to come within 5 points of the best known value in six of seven cases, while
   log-uniform draws reach it in hundreds to tens of thousands. At 15 seconds a run,
   uniform sampling is not affordable.
3. **A box of ×/÷100 on the two gated conductances and ×/÷10 on the leak**, against the
   pipeline's ×/÷5000 and ×/÷50. Four decades rather than seven: on a multi-compartment
   cell the far corners are stiff, and a stiff configuration costs minutes without being
   a plausible therapy. Every rejected candidate is counted and reported.
4. **A wall-clock guard per run.** A configuration that has not integrated 500 ms of
   model time in `--run-budget` seconds is rejected and counted, so one stiff draw cannot
   stall a shard.

The mutated channel carries the M-current, and it is **not** in the search space: the
intervention is over the pharmacologically accessible conductances only.

The work can be split across machines by index range (`--index-from`, `--index-to`), and
every candidate's conductances come from a generator seeded by its own index alone, so
any split gives exactly the same set of candidates as running the whole range in one
place.  Shards are merged with `wo6b_analyse.py merge`.
"""
import argparse, json, os, platform, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000, help="total candidates across all shards")
ap.add_argument("--index-from", type=int, default=0,
                help="first candidate index for this machine; candidate i is drawn from a "
                     "generator seeded by i alone, so any split of the range gives the "
                     "same candidates as running the whole range in one place")
ap.add_argument("--index-to", type=int, default=-1, help="one past the last; -1 means --n")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--run-budget", type=float, default=90.0, help="seconds per current level")
ap.add_argument("--seed", type=int, default=20260805)
ap.add_argument("--out", default="wo6b_taskD2_search.json")
args = ap.parse_args()

# Twelve injected-current levels.  A uniform 0.1-1.0 nA ladder separates the two cells by only 15 action potentials, so the percentage scale moves in
# steps of 6.7 points.  The separation is concentrated between 0.44 and 0.52 nA -- just
# above the wild type's rheobase, where the mutant already fires a burst -- so the ladder
# is refined there and kept coarse elsewhere, which raises the untreated distance without
# raising the cost much.  Measured: the 0.30-1.00 ladder at 0.05 steps gives 21.
CURRENTS = [0.30, 0.40, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70, 0.85, 1.00]
BOX = {"gna": 100.0, "gkdr": 100.0, "g_pas": 10.0}          # multiplicative half-width


def draw(i, defaults):
    """Candidate i, log-uniform in the box, from a generator seeded by i alone."""
    r = np.random.default_rng(args.seed + i)
    return {k: float(defaults[k] * np.exp(r.uniform(-np.log(BOX[k]), np.log(BOX[k]))))
            for k in BOX}


def worker(idx_list, wid):
    """One process: load the model once, then score its share, writing as it goes.

    Each candidate is appended to this worker's own file and flushed immediately, rather
    than held in memory and passed back through a queue at the end.  If a worker dies (for
    example when the machine runs out of memory at start-up), a queue would leave the
    parent blocked waiting for results that never arrive; with per-worker files a worker
    that dies costs only its own remaining share, and the parent reports it.
    """
    import wo6b_lib as L
    L.load()
    defaults = {"gna": L.DEF["gna"], "gkdr": L.DEF["gkdr"], "g_pas": L.G_PAS_DEF}
    with open(f"{args.out}.w{wid}.jsonl", "w") as fh:
        for i in idx_list:
            cond = draw(i, defaults)
            t0 = time.time()
            counts = L.curve_guarded(CURRENTS, True, budget=args.run_budget, **cond)
            fh.write(json.dumps({"i": int(i), "cond": cond,
                                 "counts": [int(c) for c in counts],
                                 "seconds": time.time() - t0,
                                 "rejected": bool(np.any(np.asarray(counts) < 0))}) + "\n")
            fh.flush()


def main():
    import multiprocessing as mp
    import wo6b_lib as L
    HOST = platform.node()
    L.load()
    defaults = {"gna": L.DEF["gna"], "gkdr": L.DEF["gkdr"], "g_pas": L.G_PAS_DEF}

    # the two references, measured here rather than assumed
    t0 = time.time()
    wt = L.curve(CURRENTS, False, **defaults)
    mt = L.curve(CURRENTS, True, **defaults)
    d_untreated = int(np.abs(mt - wt).sum())
    print(f"machine {HOST} | candidates {args.index_from}-{(args.n if args.index_to < 0 else args.index_to)-1} of {args.n} | {len(CURRENTS)} levels {CURRENTS[0]}-{CURRENTS[-1]} nA")
    print(f"wild type   {list(wt)}  total {wt.sum()}")
    print(f"mutant      {list(mt)}  total {mt.sum()}")
    print(f"untreated distance {d_untreated} spikes over the ladder "
          f"({100.0*d_untreated/max(wt.sum(),1):.1f} % of the wild type's total) "
          f"[references took {time.time()-t0:.0f} s]\n", flush=True)

    hi = args.n if args.index_to < 0 else args.index_to
    mine = list(range(args.index_from, hi))
    chunks = [mine[k::args.workers] for k in range(args.workers)]
    chunks = [c for c in chunks if c]
    procs = [mp.Process(target=worker, args=(c, k)) for k, c in enumerate(chunks)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    secs = time.time() - t0

    got, dead = [], []
    for k, c in enumerate(chunks):
        path = f"{args.out}.w{k}.jsonl"
        rows = []
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path) if l.strip()]
        got.extend(rows)
        if len(rows) < len(c):
            dead.append({"worker": k, "assigned": len(c), "returned": len(rows),
                         "exitcode": procs[k].exitcode})
    if dead:
        print(f"WARNING: {len(dead)} of {len(procs)} workers did not finish their share: "
              f"{dead}", flush=True)

    for r in got:
        if r["rejected"]:
            r["distance"] = None
            r["similarity"] = None
        else:
            d = int(np.abs(np.array(r["counts"]) - wt).sum())
            r["distance"] = d
            r["similarity"] = float(L.similarity(d, d_untreated))
    got.sort(key=lambda r: r["i"])
    ok = [r for r in got if not r["rejected"]]
    best = max(ok, key=lambda r: r["similarity"]) if ok else None

    OUT = {"host": HOST, "index_from": args.index_from,
           "index_to": (args.n if args.index_to < 0 else args.index_to), "n_total": args.n,
           "currents_nA": CURRENTS, "box": BOX, "seed": args.seed,
           "run_budget_s": args.run_budget,
           "wild_type_counts": [int(x) for x in wt], "mutant_counts": [int(x) for x in mt],
           "d_untreated": d_untreated, "seconds": secs,
           "n_evaluated": len(got), "n_rejected": sum(1 for r in got if r["rejected"]),
           "n_assigned": len(mine), "workers_incomplete": dead,
           "candidates": got}
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)

    print(f"{len(got)} candidates, {OUT['n_rejected']} rejected, {secs:.0f} s wall clock "
          f"on {len(procs)} processes ({secs*len(procs)/max(len(got),1):.0f} core-seconds "
          f"each)")
    if best:
        print(f"best similarity {best['similarity']:.2f} %  at "
              + ", ".join(f"{k} x{best['cond'][k]/defaults[k]:.3f}" for k in BOX))
        print(f"  counts {best['counts']}  against wild type {list(wt)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
