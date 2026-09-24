"""Stability window of the compensating optimum, measured on the NEURON Kv7.2 model.

The same stability measurement as for the single-compartment differentiable model
(experiments/wo5/task11c_stability.py), repeated on the multi-compartment CA1 model, to
check whether the window found on the simpler model carries over.

**Protocol.** Perturbation is fractional: each optimized conductance is multiplied by
(1 + delta).  It is applied only to the pharmacologically accessible conductances -- here
sodium, delayed-rectifier potassium and leak -- and never to the mutated channel, which
carries the M-current and is not being dosed.  Efficacy is the spike-count similarity at
the perturbed point divided by the similarity at the optimum, as a percentage.  Three
modes are reported: one conductance at a time (both signs), the joint median and 5th
percentile over random draws, and the worst case over the corners of the perturbation
cube.

The single-compartment measurement uses 1,000 random draws per perturbation size.  Here
a draw costs one NEURON run per current level, about 150 seconds of processor time, so 1,000 draws per
size would be 250 processor-hours; this uses `--draws` (default 200) and reports the 5th
percentile alongside the median.  The one-at-a-time and worst-case modes are exhaustive
and are not sampled.

The optimum is the best candidate in the search result file (wo6b_taskD2_search.py).
Evaluations can be split across machines with `--shard`/`--shards`; the shard files are
summarised by `wo6b_analyse.py taskE`.
"""
import argparse, json, itertools, os, platform, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--search", default="wo6b_taskD2_search.json",
                help="the search result file; its best candidate is the optimum")
ap.add_argument("--deltas", default="1,2,3,5,10,15,20", help="percentages")
ap.add_argument("--draws", type=int, default=200, help="random joint draws per size")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--run-budget", type=float, default=90.0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--shards", type=int, default=1)
ap.add_argument("--seed", type=int, default=20260806)
ap.add_argument("--out", default="wo6b_taskE_stability.json")
args = ap.parse_args()

KEYS = ["gna", "gkdr", "g_pas"]
DELTAS = [float(x) / 100.0 for x in args.deltas.split(",")]


def build_jobs(opt):
    """Every perturbed configuration to evaluate, tagged with the mode it belongs to."""
    jobs = []
    for d in DELTAS:
        for k in KEYS:                                        # one at a time, both signs
            for s in (+1, -1):
                c = dict(opt)
                c[k] = opt[k] * (1 + s * d)
                jobs.append({"mode": "one_at_a_time", "delta": d, "key": k, "sign": s,
                             "cond": c})
        for signs in itertools.product((+1, -1), repeat=len(KEYS)):   # worst case: corners
            c = {k: opt[k] * (1 + s * d) for k, s in zip(KEYS, signs)}
            jobs.append({"mode": "worst_case", "delta": d, "signs": list(signs), "cond": c})
        r = np.random.default_rng(args.seed + int(d * 10000))         # joint random draws
        for j in range(args.draws):
            f = r.uniform(-d, d, len(KEYS))
            c = {k: opt[k] * (1 + f[i]) for i, k in enumerate(KEYS)}
            jobs.append({"mode": "joint", "delta": d, "draw": j, "cond": c})
    return jobs


def worker(jobs, currents, wid):
    """Writes each evaluation to its own file and flushes, so a worker that dies at
    start-up costs only its own share rather than deadlocking the parent."""
    import wo6b_lib as L
    L.load()
    fh = open(f"{args.out}.w{wid}.jsonl", "w")
    out = []
    for job in jobs:
        counts = L.curve_guarded(currents, True, budget=args.run_budget, **job["cond"])
        rec = {k: v for k, v in job.items() if k != "cond"}
        rec["cond"] = job["cond"]
        rec["counts"] = [int(c) for c in counts]
        rec["rejected"] = bool(np.any(np.asarray(counts) < 0))
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        out.append(rec)
    fh.close()


def main():
    import multiprocessing as mp
    import wo6b_lib as L
    HOST = platform.node()
    S = json.load(open(args.search))
    currents = S["currents_nA"]
    wt = np.array(S["wild_type_counts"])
    d_untr = S["d_untreated"]
    ok = [c for c in S["candidates"] if not c["rejected"]]
    best = max(ok, key=lambda c: c["similarity"])
    opt = {k: best["cond"][k] for k in KEYS}
    sim_opt = best["similarity"]
    print(f"machine {HOST} | optimum from {args.search}: similarity {sim_opt:.2f} %")
    print("  " + ", ".join(f"{k} = {opt[k]:.6g}" for k in KEYS))
    print(f"  perturbation sizes {[int(d*100) for d in DELTAS]} %, "
          f"{args.draws} joint draws each\n", flush=True)

    jobs = build_jobs(opt)
    mine = [j for n, j in enumerate(jobs) if n % args.shards == args.shard]
    chunks = [mine[k::args.workers] for k in range(args.workers)]
    chunks = [c for c in chunks if c]
    procs = [mp.Process(target=worker, args=(c, currents, k))
             for k, c in enumerate(chunks)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    secs = time.time() - t0
    got, dead = [], []
    for k, c in enumerate(chunks):
        path = f"{args.out}.w{k}.jsonl"
        rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
        got.extend(rows)
        if len(rows) < len(c):
            dead.append({"worker": k, "assigned": len(c), "returned": len(rows)})
    if dead:
        print(f"WARNING: {len(dead)} workers did not finish their share: {dead}", flush=True)

    for r in got:
        if r["rejected"]:
            r["efficacy_pct"] = None
        else:
            d = int(np.abs(np.array(r["counts"]) - wt).sum())
            sim = float(L.similarity(d, d_untr))
            r["similarity"] = sim
            r["efficacy_pct"] = 100.0 * sim / sim_opt

    OUT = {"host": HOST, "shard": args.shard, "shards": args.shards,
           "optimum": opt, "similarity_at_optimum": sim_opt, "currents_nA": currents,
           "wild_type_counts": [int(x) for x in wt], "d_untreated": d_untr,
           "deltas_pct": [d * 100 for d in DELTAS], "draws": args.draws,
           "seconds": secs, "results": got}
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)
    print(f"{len(got)} evaluations, {sum(1 for r in got if r['rejected'])} rejected, "
          f"{secs:.0f} s on {len(procs)} processes")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
