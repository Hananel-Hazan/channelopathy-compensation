"""Compare uniform and log-uniform sampling of the conductance box.

For each channel variant and each sampling scheme (uniform or log-uniform across
the decades of the conductance range), this script runs several independent
random-search repeats, each with its own seed, and records per repeat:
  * the number of draws until the first candidate whose similarity is within
    `--margin` percentage points of the best value known for that variant;
  * the fraction of draws producing a cell that fires at all (candidates that
    never spike are discarded by the search pipeline);
  * the best similarity reached.
The spread across seeds gives the error bars for the comparison.

The conductance box is [WT/5000, WT*5000] for gNa and gK and [WT/50, WT*50] for
the leak conductance, as in the original search pipeline.

The known best value per variant is read from wo7_topology_sweep.json (written by
wo7_task23_topology_sweep.py); fixed fall-back values are used if that file is
missing.  Output: wo7_sampling.json next to this script.

Every simulator call is padded to one constant batch shape.  Without that,
torch.compile exceeds its recompile limit and silently falls back to eager
execution, which is orders of magnitude slower.

Requires a CUDA GPU.
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

for _a in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _a):
        setattr(torch._dynamo.config, _a, 64)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W                                              # noqa: E402

DEV = "cuda"

VARIANTS = {
    "sodium activation +6.1 mV (R859C, ModelDB 87585)":           dict(shift=6.1, shift_n=0.0),
    "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)": dict(shift=0.0, shift_n=4.7),
    "potassium activation +9.4 mV (twice the measured shift)":     dict(shift=0.0, shift_n=9.4),
    "combined: sodium +6.1 mV and potassium +4.7 mV":              dict(shift=6.1, shift_n=4.7),
}

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--steps-per-call", type=int, default=20)
ap.add_argument("--budget", type=int, default=200_000, help="draws per repeat")
ap.add_argument("--repeats", type=int, default=8)
ap.add_argument("--seed0", type=int, default=20260826)
ap.add_argument("--margin", type=float, default=5.0,
                help="percentage points below the known best that count as success")
ap.add_argument("--best", default=os.path.join(REPO, "figures", "scripts", "wo7_topology_sweep.json"),
                help="where the known best value per variant comes from")
ap.add_argument("--out", default=os.path.join(REPO, "figures", "scripts", "wo7_sampling.json"))
args = ap.parse_args()

HOST = platform.node()
I1 = torch.tensor(np.arange(W.N_CUR) * W.I_STEP + W.I_LOW, dtype=torch.float32, device=DEV)
LO = np.array([W.WT[0] / 5000, W.WT[1] / 5000, W.WT[2] / 50])
HI = np.array([W.WT[0] * 5000, W.WT[1] * 5000, W.WT[2] * 50])


def spikes_of(P, shift, chunk, shift_n=0.0):
    N = P.shape[0]
    out = np.empty((N, W.N_CUR), dtype=np.int16)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N); nb = e - s0
        blk = P[s0:e]
        if nb < chunk:
            blk = np.vstack([blk, np.repeat(blk[-1:], chunk - nb, axis=0)])
        rep = np.repeat(blk, W.N_CUR, axis=0)
        g = [torch.tensor(rep[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I1.repeat(chunk),
                                   shift=float(shift), device=DEV,
                                   steps_per_call=args.steps_per_call,
                                   shift_n=float(shift_n))
        out[s0:e] = st["spikes"].cpu().numpy().reshape(chunk, W.N_CUR)[:nb]
    return out


print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}", flush=True)
print(f"{args.repeats} independent repeats per (variant, scheme), "
      f"{args.budget:,} draws each, seeds {args.seed0}..{args.seed0+args.repeats-1}",
      flush=True)
print(f"success = within {args.margin:g} percentage points of the best value known "
      f"for that variant\n", flush=True)

sc_wt = spikes_of(np.array([W.WT]), 0.0, args.chunk)[0].astype(int)
KNOWN = {}
try:
    kb = json.load(open(args.best))["variants"]
    KNOWN = {k: v["best_similarity_pct"] for k, v in kb.items()}
    print(f"known best values read from {args.best}\n", flush=True)
except Exception as e:
    print(f"could not read known best values ({e}); "
          f"using fixed fall-back values\n", flush=True)
    KNOWN = {"sodium activation +6.1 mV (R859C, ModelDB 87585)": 97.60,
             "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)": 99.35,
             "potassium activation +9.4 mV (twice the measured shift)": 99.05,
             "combined: sodium +6.1 mV and potassium +4.7 mV": 95.28}

RESULT = {"host": HOST, "gpu": torch.cuda.get_device_properties(0).name,
          "budget_per_repeat": args.budget, "repeats": args.repeats,
          "seed0": args.seed0, "margin_pct": args.margin,
          "box_lo": LO.tolist(), "box_hi": HI.tolist(),
          "wt_spikes_total": int(sc_wt.sum()), "variants": {}}

for vname, spec in VARIANTS.items():
    t0 = time.time()
    d_untr = int(np.abs(spikes_of(np.array([W.WT]), spec["shift"], args.chunk,
                                  spec["shift_n"])[0].astype(int) - sc_wt).sum())
    target = KNOWN[vname] - args.margin
    entry = {"d_untreated": d_untr, "known_best_pct": KNOWN[vname],
             "target_pct": target, "schemes": {}}
    print(f"--- {vname}\n    untreated distance {d_untr}, "
          f"target >= {target:.2f} %", flush=True)

    for scheme in ("uniform", "loguniform"):
        first_hits, viable, bests = [], [], []
        for r in range(args.repeats):
            seed = args.seed0 + r
            rng = np.random.default_rng(seed)
            drawn = 0; hit = None; n_alive = 0; best = -np.inf
            while drawn < args.budget:
                nb = min(args.chunk, args.budget - drawn)
                if scheme == "uniform":
                    P = rng.uniform(LO, HI, size=(nb, 3))
                else:
                    P = np.exp(rng.uniform(np.log(LO), np.log(HI), size=(nb, 3)))
                sc = spikes_of(P, spec["shift"], args.chunk,
                               spec["shift_n"]).astype(int)
                alive = sc.sum(axis=1) > 0
                n_alive += int(alive.sum())
                d = np.abs(sc - sc_wt[None, :]).sum(axis=1)
                sim = W.similarity(d, d_untr)
                sim = np.where(alive, sim, -np.inf)
                if sim.size:
                    best = max(best, float(np.nanmax(sim)))
                if hit is None:
                    idx = np.where(sim >= target)[0]
                    if idx.size:
                        hit = drawn + int(idx[0]) + 1
                drawn += nb
            first_hits.append(hit)
            viable.append(n_alive / drawn * 100.0)
            bests.append(best)
        reached = [h for h in first_hits if h is not None]
        entry["schemes"][scheme] = {
            "first_hit_per_seed": first_hits,
            "viable_pct_per_seed": [round(v, 4) for v in viable],
            "best_reached_per_seed": [round(b, 4) for b in bests],
            "n_reached": len(reached), "n_repeats": args.repeats,
            "first_hit_median": float(np.median(reached)) if reached else None,
            "first_hit_min": min(reached) if reached else None,
            "first_hit_max": max(reached) if reached else None,
            "viable_median_pct": float(np.median(viable)),
            "viable_min_pct": float(np.min(viable)),
            "viable_max_pct": float(np.max(viable))}
        s = entry["schemes"][scheme]
        fh = (f"{s['first_hit_median']:,.0f} "
              f"[{s['first_hit_min']:,}-{s['first_hit_max']:,}]"
              if reached else f"not reached in {args.budget:,}")
        print(f"    {scheme:<11} first hit {fh:<28} "
              f"reached {len(reached)}/{args.repeats} seeds   "
              f"viable {s['viable_median_pct']:5.1f} % "
              f"[{s['viable_min_pct']:.1f}-{s['viable_max_pct']:.1f}]", flush=True)

    entry["seconds"] = round(time.time() - t0, 1)
    RESULT["variants"][vname] = entry
    print(f"    {entry['seconds']:.0f} s\n", flush=True)

with open(args.out, "w") as f:
    json.dump(RESULT, f, indent=2)
print(f"wrote {args.out}", flush=True)
