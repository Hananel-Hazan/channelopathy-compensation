"""Cost of direct search: how many random candidate parameter sets must be evaluated.

Measures how many candidate parameter sets must be evaluated before a random search finds
a compensating configuration, and what that costs at each pathway's measured throughput.

Method.  For each of the seven cases, draw a large random sample from the parameter box
of the NEURON search pipeline (each free conductance from wild type / 5000 to wild type
x 5000; the leak conductance from / 50 to x 50) and score every draw on the functional
measure.  The number of draws needed to reach a given quality is then a property of the
distribution rather than of one lucky run: if a fraction p of the box meets the target,
the number of draws to the first success is geometric, with median log(2) / -log(1-p).
The median first-hit position over many independent shuffles of the sample is reported
alongside it as an empirical check.

Two sampling schemes are measured, because the choice strongly affects the hit rate:
  uniform      -- what the NEURON search pipeline does
  log-uniform  -- the same box, draws spread evenly across decades

Usage:  python wo6_taskB.py [--samples 2000000] [--cases 1,2,...] [--out FILE]
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W

# torch.compile gives up after `recompile_limit` distinct traces of the same function and
# falls back to running it one operation at a time, which is 200 times slower here.  The
# limit is reached partway through the seven cases -- each kinetic shift and each final
# partial chunk is a new trace.  Raising the limit changes no result; it only stops the
# timings from measuring the fallback rather than the model.
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.accumulated_recompile_limit = 2048

DEV = "cuda"
# measured throughputs, candidate parameter sets (35 current levels each) per second
RATE_GPU_2070 = 34350.0     # fused GPU simulator, RTX 2070
RATE_GPU_4070 = 63205.0     # fused GPU simulator, RTX 4070 Ti
RATE_NEURON_CORE = 1.0      # NEURON, per processor core

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=2_000_000)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--shuffles", type=int, default=200)
ap.add_argument("--cases", default="",
                help="comma-separated 1-based case numbers; empty means all seven. "
                     "Each case draws from its own generator seeded identically "
                     "(20260804), so any partition of the seven across machines gives "
                     "the same numbers as running them together.")
ap.add_argument("--out", default="wo6_taskB.json")
args = ap.parse_args()

HOST = platform.node()
ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
I1 = torch.tensor(ALLC, dtype=torch.float32, device=DEV)
LO = np.array([W.WT[0] / 5000, W.WT[1] / 5000, W.WT[2] / 50])
HI = np.array([W.WT[0] * 5000, W.WT[1] * 5000, W.WT[2] * 50])

CASES = [
    ("conductance, g_Na fixed 140",    (140., 36., .03), [0, 1, 1], 0.0, 0.0, 96.15),
    ("conductance, g_K fixed 20",      (120., 20., .03), [1, 0, 1], 0.0, 0.0, 93.97),
    ("conductance, g_leak fixed 0.30", (120., 36., .30), [1, 1, 0], 0.0, 0.0, 95.02),
    ("kinetic, sodium +6.1 mV",        (120., 36., .03), [1, 1, 1], 6.1, 0.0, 97.60),
    ("kinetic, potassium +4.7 mV",     (120., 36., .03), [1, 1, 1], 0.0, 4.7, 99.35),
    ("kinetic, potassium +9.4 mV",     (120., 36., .03), [1, 1, 1], 0.0, 9.4, 99.05),
    ("kinetic, both shifts",           (120., 36., .03), [1, 1, 1], 6.1, 4.7, 95.28),
]


def spikes(P, sm, sn, chunk):
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.empty((P.shape[0], W.N_CUR), np.int16)
    for s0 in range(0, P.shape[0], chunk):
        e = min(s0 + chunk, P.shape[0]); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I1.repeat(nb), shift=float(sm),
                                   shift_n=float(sn), device=DEV, steps_per_call=20)
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out.astype(int)


if args.cases:
    want = [int(x) for x in args.cases.split(",")]
    CASES = [CASES[i - 1] for i in want]
else:
    want = list(range(1, len(CASES) + 1))

SC_WT = spikes([W.WT], 0.0, 0.0, 64)[0]
print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}")
print(f"{args.samples:,} draws per case per scheme; wild type fires {SC_WT.sum()}")
print(f"cases {want}\n")

OUT = {"host": HOST, "samples": args.samples, "case_numbers": want,
       "rates": {"rtx2070_fused": RATE_GPU_2070, "rtx4070ti_fused": RATE_GPU_4070,
                 "neuron_per_core": RATE_NEURON_CORE},
       "cases": []}

for name, untr, mask, sm, sn, best in CASES:
    rec = {"case": name, "best_known": best, "schemes": {}}
    d_untr = int(np.abs(spikes([untr], sm, sn, 64)[0] - SC_WT).sum())
    rec["d_untreated"] = d_untr
    free = [j for j in range(3) if mask[j]]
    print(f"--- {name}   (untreated distance {d_untr}, best known {best:.2f} %)", flush=True)

    for scheme in ("uniform", "log-uniform"):
        rng = np.random.default_rng(20260804)
        t0 = time.time()
        P = np.repeat(np.array(untr, float)[None, :], args.samples, axis=0)
        for j in free:
            P[:, j] = (rng.uniform(LO[j], HI[j], args.samples) if scheme == "uniform"
                       else np.exp(rng.uniform(np.log(LO[j]), np.log(HI[j]), args.samples)))
        sc = spikes(P, sm, sn, args.chunk)
        alive = sc.sum(axis=1) > 0
        d = np.abs(sc - SC_WT[None, :]).sum(axis=1)
        sim = W.similarity(d, d_untr)
        secs = time.time() - t0

        row = {"seconds": secs, "best_found": float(sim.max()),
               "spiking_fraction": float(alive.mean()), "thresholds": {}}
        for tol in (1.0, 2.0, 5.0):
            thr = best - tol
            hit = sim >= thr
            p = float(hit.mean())
            first = None
            if hit.any():
                firsts = []
                for s in range(args.shuffles):
                    r2 = np.random.default_rng(1000 + s)
                    pos = np.where(hit[r2.permutation(args.samples)])[0]
                    firsts.append(int(pos[0]) + 1 if pos.size else args.samples)
                first = float(np.median(firsts))
            row["thresholds"][f"within {tol:.0f} points"] = {
                "threshold_similarity": thr, "hit_fraction": p,
                "median_draws_theory": (float(np.log(2) / -np.log1p(-p))
                                        if 0 < p < 1 else (1.0 if p >= 1 else None)),
                "observed_median_first_hit": first}
        rec["schemes"][scheme] = row

        print(f"    {scheme:<12} spiking {row['spiking_fraction']*100:5.1f} %  "
              f"best found {sim.max():6.2f} %  [{secs:.0f} s]", flush=True)
        for tol in (1.0, 2.0, 5.0):
            t = row["thresholds"][f"within {tol:.0f} points"]
            if t["hit_fraction"] > 0:
                print(f"      within {tol:.0f} pt (>= {t['threshold_similarity']:6.2f} %): "
                      f"{t['hit_fraction']*100:9.5f} % of the box, median "
                      f"{t['observed_median_first_hit']:,.0f} draws to first hit")
            else:
                print(f"      within {tol:.0f} pt (>= {t['threshold_similarity']:6.2f} %): "
                      f"NOT reached in {args.samples:,} draws")
    OUT["cases"].append(rec)
    print(flush=True)

print("=== what the draws cost, by pathway (log-uniform sampling, within 5 points) ===")
print(f"{'case':<32} {'draws':>12} {'RTX 2070':>11} {'RTX 4070 Ti':>12} "
      f"{'NEURON 1 core':>14} {'NEURON 72 cores':>16}")
for rec in OUT["cases"]:
    t = rec["schemes"]["log-uniform"]["thresholds"]["within 5 points"]
    n = t["observed_median_first_hit"]
    if not n:
        print(f"{rec['case']:<32} {'not reached':>12}")
        continue
    print(f"{rec['case']:<32} {n:>12,.0f} {n/RATE_GPU_2070:>10.2f}s "
          f"{n/RATE_GPU_4070:>11.2f}s {n/RATE_NEURON_CORE/3600:>13.2f}h "
          f"{n/RATE_NEURON_CORE/72/3600:>15.2f}h")

with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1)
print(f"\n-> {args.out}")
