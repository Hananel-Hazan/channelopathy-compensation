"""Stability sweep of wo6b_taskE_stability.py, with every evaluation in a fresh process.

NEURON carries solver state from one run to the next inside a process, so a spike count
evaluated in a long-lived worker can depend on what that same process evaluated before it,
i.e. on evaluation order.  This script removes that dependence: each configuration is
evaluated in its own child process, forked from a worker that has never loaded NEURON, and
the child imports wo6b_lib, loads the model, runs the current ladder and sends the counts
back.  Job list, seeds, sharding, reduction and output layout are the same as in
wo6b_taskE_stability.py; see that file for the perturbation protocol (fractional
perturbation of the accessible conductances; one-at-a-time, joint random and worst-case
modes; efficacy relative to the optimum).

Process handling:
  1. `_start()` retries a fork that fails with ENOMEM/EAGAIN, at 20 s intervals for up to
     10 minutes, and logs each retry to stderr; a fork that still fails raises.  With many
     workers each forking a NEURON child, the kernel can transiently refuse to fork for
     want of memory.
  2. The parent does not import wo6b_lib (and through it NEURON) before forking the
     workers; it imports it afterwards, for the reduction only.  So a worker carries no
     NEURON state, and each child does the import and the model load itself.
Each worker holds one NEURON child at a time, so `--workers N` means N concurrent NEURON
processes.
"""
import argparse, errno, json, itertools, os, platform, sys, time
import numpy as np


def _start(p, where, retries=30, wait=20.0):
    """p.start(), retried when the kernel refuses to fork for want of memory. Any other
    error, or exhaustion of the retries, raises."""
    for n in range(retries + 1):
        try:
            p.start()
            return
        except OSError as e:
            if e.errno not in (errno.ENOMEM, errno.EAGAIN) or n == retries:
                raise
            print(f"[{where}] fork refused ({e.strerror}); retry {n + 1}/{retries} in {wait:.0f} s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)

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


def _fresh_eval(cond, currents, budget, conn):
    """Runs in a child process; this configuration is the only thing it ever evaluates."""
    import wo6b_lib as L
    L.load()
    conn.send([int(c) for c in L.curve_guarded(currents, True, budget=budget, **cond)])
    conn.close()


def worker(jobs, currents, wid):
    """Writes each evaluation to its own file and flushes, so a worker that dies at
    start-up costs only its own share rather than deadlocking the parent.

    This worker never imports NEURON. Each evaluation is forked into a clean child
    (a fresh process as far as NEURON is concerned) and its counts are read back."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    fh = open(f"{args.out}.w{wid}.jsonl", "w")
    out = []
    for job in jobs:
        r_end, w_end = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_fresh_eval, args=(job["cond"], currents, args.run_budget, w_end))
        _start(p, f"worker {wid}"); w_end.close()
        if r_end.poll(args.run_budget * len(currents) + 300):
            counts = r_end.recv()
        else:                                   # child hung or died: reject, as the guard does
            counts = [-1] * len(currents)
        p.join(30)
        if p.is_alive():
            p.kill(); p.join()
        rec = {k: v for k, v in job.items() if k != "cond"}
        rec["cond"] = job["cond"]
        rec["counts"] = [int(c) for c in counts]
        rec["rejected"] = bool(np.any(np.asarray(counts) < 0))
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        out.append(rec)
    fh.close()


def main():
    import multiprocessing as mp
    # wo6b_lib (and NEURON) is imported AFTER the workers have run, below, so that the
    # forked workers carry no NEURON state.
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
    for k, p in enumerate(procs):
        _start(p, f"parent, worker {k}")
        time.sleep(1.0)                     # stagger the first round of model loads
    for p in procs:
        p.join()
    secs = time.time() - t0
    import wo6b_lib as L                    # for L.similarity in the reduction only
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
           "seconds": secs,
           "reset": "one fresh forked process per evaluation, worker never loads NEURON",
           "results": got}
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)
    print(f"{len(got)} evaluations, {sum(1 for r in got if r['rejected'])} rejected, "
          f"{secs:.0f} s on {len(procs)} processes")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
