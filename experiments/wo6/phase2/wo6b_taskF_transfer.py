"""Do compensating configurations found on the differentiable model transfer to NEURON?

Candidate parameters from the fast differentiable model can be used to seed the region of
parameter space simulated in a detailed model.  This tests whether such seeds land
anywhere useful, for the one mutation both models share: the Kv7.2 D212G
potassium-activation shift.

**What is transferred.** A direct search of the differentiable model's own box for the
potassium +4.7 mV variant (experiments/wo5/task12_topology.py) reached 99.35 % of the
wild-type firing pattern, with 45 configurations tied at the minimum distance and 3,353
in the top set (experiments/wo5/task12_grid_4070.json).  Those configurations are taken
from that file.

**How it is transferred.** The two models do not share units or wild-type values: the
differentiable model is a single compartment with (gNa, gK, gl) = (120, 36, 0.03) in
mS/cm^2, while the NEURON cell is 140 sections with (gna, gkdr, g_pas) =
(0.045, 0.02, 2.68e-5) and its own axonal multipliers and dendritic gradients.  An
absolute conductance carries no meaning across that gap.  What does carry is the
**prescription** -- "reduce sodium to 74 % of wild type, raise delayed-rectifier
potassium to 226 %, cut leak to 17 %" -- so each configuration is applied as the ratio to
its own model's wild type.  This ratio mapping is an assumption, not a derived result.

**What it is compared against.** The direct search of the NEURON model itself
(wo6b_taskD2_search.py).  Transfer efficiency is the similarity the transferred
configuration reaches in NEURON divided by the similarity NEURON's own search reaches, on
the same current ladder and the same percentage scale.
"""
import argparse, json, os, platform, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--search", default="wo6b_taskD2_search.json")
# the differentiable model's direct-search result
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap.add_argument("--grid", default=os.path.join(REPO, "experiments", "wo5", "task12_grid_4070.json"))
ap.add_argument("--variant", default="potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)")
ap.add_argument("--n-transfer", type=int, default=24,
                help="how many of the differentiable model's top configurations to carry over")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--run-budget", type=float, default=90.0)
ap.add_argument("--out", default="wo6b_taskF_transfer.json")
args = ap.parse_args()

KEYS = ["gna", "gkdr", "g_pas"]          # NEURON, in the order the grid file uses


def worker(jobs, currents, wid):
    """Writes each evaluation to its own file and flushes, so a worker that dies at
    start-up costs only its own share rather than deadlocking the parent."""
    import wo6b_lib as L
    L.load()
    fh = open(f"{args.out}.w{wid}.jsonl", "w")
    out = []
    for job in jobs:
        counts = L.curve_guarded(currents, True, budget=args.run_budget, **job["cond"])
        rec = dict(job)
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
    wt_counts = np.array(S["wild_type_counts"])
    d_untr = S["d_untreated"]
    ok = [c for c in S["candidates"] if not c["rejected"]]
    neuron_best = max(ok, key=lambda c: c["similarity"])

    G = json.load(open(args.grid))["variants"][args.variant]
    base = G["baseline"]                                   # (gNa, gK, gl) of the fast model
    pts = G["top_points"]
    # spread the picks across the top set rather than taking 24 neighbours of one point
    step = max(1, len(pts) // args.n_transfer)
    picks = pts[::step][:args.n_transfer]

    L.load()
    defaults = {"gna": L.DEF["gna"], "gkdr": L.DEF["gkdr"], "g_pas": L.G_PAS_DEF}
    jobs = []
    for n, p in enumerate(picks):
        ratio = [p[j] / base[j] for j in range(3)]
        cond = {KEYS[j]: defaults[KEYS[j]] * ratio[j] for j in range(3)}
        jobs.append({"n": n, "fast_model_point": p, "ratio": ratio, "cond": cond})

    print(f"machine {HOST} | transferring {len(jobs)} configurations from the "
          f"differentiable model")
    print(f"  its best similarity there: {G['best_similarity_pct']:.2f} % "
          f"({G['n_tied_at_min']} tied at the minimum, {G['top_n']} in the top set)")
    print(f"  NEURON's own best from the direct search: {neuron_best['similarity']:.2f} %")
    print(f"  example prescription: sodium x{jobs[0]['ratio'][0]:.3f}, "
          f"potassium x{jobs[0]['ratio'][1]:.3f}, leak x{jobs[0]['ratio'][2]:.3f}\n",
          flush=True)

    chunks = [jobs[k::args.workers] for k in range(args.workers)]
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
            r["similarity"] = None
            r["transfer_efficiency_pct"] = None
        else:
            d = int(np.abs(np.array(r["counts"]) - wt_counts).sum())
            sim = float(L.similarity(d, d_untr))
            r["similarity"] = sim
            r["transfer_efficiency_pct"] = 100.0 * sim / neuron_best["similarity"]
    got.sort(key=lambda r: r["n"])
    good = [r for r in got if not r["rejected"]]
    best_t = max(good, key=lambda r: r["similarity"]) if good else None

    OUT = {"host": HOST, "variant": args.variant, "currents_nA": currents,
           "fast_model_baseline": base,
           "fast_model_best_similarity_pct": G["best_similarity_pct"],
           "neuron_best_similarity_pct": neuron_best["similarity"],
           "neuron_best_cond": neuron_best["cond"],
           "wild_type_counts": [int(x) for x in wt_counts], "d_untreated": d_untr,
           "seconds": secs, "transferred": got}
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=1)

    print(f"{len(got)} transferred, {sum(1 for r in got if r['rejected'])} rejected, "
          f"{secs:.0f} s")
    if best_t:
        sims = [r["similarity"] for r in good]
        print(f"best transferred similarity in NEURON {best_t['similarity']:.2f} % "
              f"(median {np.median(sims):.2f} %, worst {min(sims):.2f} %)")
        print(f"transfer efficiency, best: {best_t['transfer_efficiency_pct']:.1f} % of "
              f"what searching NEURON directly achieves")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
