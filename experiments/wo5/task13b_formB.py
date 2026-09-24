"""Compensation of a kinetic sodium variant on the differentiable model.

The variant is a depolarizing shift of the sodium activation (`wo5_lib.alphaM`).  For
each shift in SHIFTS two configurations are fitted:

  - the mutated channel's gating kinetics are held at their variant values and all
    three conductances, including the mutated channel's own g_Na, are searched;
  - the same, but with g_Na also held at its wild-type value (two conductances free).

Each fit is against the wild type across eight injected-current levels, with six
random restarts per point and the best kept.  Results are scored on all 35 levels by
the fraction of the MSE gap closed and by spike-count similarity, and the range of
shifts over which each configuration reaches 95 / 90 / 80 % is reported.

The shift matching the R859C variant is read off the two NEURON mechanisms of
ModelDB 87585:
  ichanWT2005.mod   minf half-activation -27.4 mV, slope factor 5.40 mV
  ichanR859C1.mod   minf half-activation -21.3 mV, slope factor 7.26 mV
  => a depolarizing shift of +6.1 mV.

Writes task13b_summary.json to the current directory.
"""
import json, os, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W
import wo5_fit as F

DEV = "cuda"
R859C = 6.1
SHIFTS = [-5.0, -2.0, 1.0, 2.0, 3.0, 4.0, 5.0, R859C, 8.0, 10.0, 14.0, 20.0]
GROUP = 3                 # shifts fitted at once, to bound activation memory
N_RESTART = 6            # GROUP x 2 configurations x N_RESTART = 36 problems x 8
                        # currents = 288 elements -- the same batch shape as
                        # task11c_stability.py, so a cached Inductor graph is reused
ITERS = 200
FORMS = {"kinetics fixed, all three conductances free": [True, True, True],
         "kinetics fixed AND g_Na fixed, two free":     [False, True, True]}

t00 = time.time()
FITC = F.FIT_CURRENTS
ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
tgt_fit = F.target_traces_multi(FITC)
I1 = torch.tensor(ALLC, dtype=torch.float32, device=DEV)


def spikes_of(P, shift):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    blk = np.repeat(P, W.N_CUR, axis=0)
    g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
    st = W.simulate_stats_fast(g[0], g[1], g[2], I1.repeat(P.shape[0]),
                               shift=float(shift), device=DEV)
    return st["spikes"].cpu().numpy().reshape(P.shape[0], W.N_CUR).astype(int)


sc_wt = spikes_of([W.WT], 0.0)[0]
print(f"fit on {len(FITC)} currents {list(FITC)}, "
      f"evaluate on all {W.N_CUR}")
print(f"wild-type spike counts: total {sc_wt.sum()}")
print(f"the R859C-equivalent kinetic shift is +{R859C} mV (measured from the mod files)\n")

rng = np.random.default_rng(20260803)
res = {k: {} for k in FORMS}

for gi in range(0, len(SHIFTS), GROUP):
    grp = SHIFTS[gi:gi + GROUP]
    starts, masks, shifts_v, tag = [], [], [], []
    for s in grp:
        for label, m in FORMS.items():
            fi = [j for j in range(3) if m[j]]
            for r in range(N_RESTART):
                p = np.array(W.WT, dtype=float)
                if r > 0:
                    for j in fi:
                        p[j] = p[j] * 10 ** rng.uniform(-0.6, 0.6)
                starts.append(p); masks.append(m); shifts_v.append(s)
                tag.append((label, s))
    t0 = time.time()
    r = F.fit_batch_multi(np.array(starts), np.array(masks), tgt_fit, FITC,
                          shift=np.array(shifts_v), iters=ITERS)
    print(f"  shifts {grp}: {len(starts)} fits in {time.time()-t0:.1f} s", flush=True)
    for label, s in set(tag):
        sel = [i for i, t in enumerate(tag) if t == (label, s)]
        b = sel[int(np.argmin(r["best_loss"][sel]))]
        res[label][s] = dict(loss=float(r["best_loss"][b]),
                             params=list(map(float, r["best_params"][b])),
                             n_restart=N_RESTART)

# untreated variant: wild-type conductances with the variant kinetics
L_untr = {s: float(F.losses_multi([W.WT], tgt_fit, FITC, shift=s)[0]) for s in SHIFTS}
d_untr = {s: int(np.abs(spikes_of([W.WT], s)[0] - sc_wt).sum()) for s in SHIFTS}

print(f"\n{'shift mV':>9} {'untreated MSE':>14} {'untreated spk d':>16} | "
      f"{'B gap':>8} {'B spk sim':>10} | {'A gap':>8} {'A spk sim':>10}")
OUT = {"shifts": SHIFTS, "r859c_shift": R859C, "wt_spikes": sc_wt.tolist(),
       "untreated_mse": L_untr, "untreated_spike_distance": d_untr, "forms": {}}
for label in FORMS:
    OUT["forms"][label] = {}
for s in SHIFTS:
    row = []
    for label in FORMS:
        e = res[label][s]
        gap = float(W.gap_closed(e["loss"], L_untr[s]))
        d = int(np.abs(spikes_of([e["params"]], s)[0] - sc_wt).sum())
        sim = float(W.similarity(d, d_untr[s])) if d_untr[s] else float("nan")
        e.update(gap=gap, spike_distance=d, spike_similarity=sim)
        OUT["forms"][label][str(s)] = e
        row += [gap, sim]
    print(f"{s:>9.1f} {L_untr[s]:>14.2f} {d_untr[s]:>16} | "
          f"{row[0]:>7.2f}% {row[1]:>9.2f}% | {row[2]:>7.2f}% {row[3]:>9.2f}%")

print(f"\n{'':>9} converged conductances (Na, K, l)")
for s in SHIFTS:
    for label in FORMS:
        p = res[label][s]["params"]
        print(f"{s:>9.1f} {label[:6]}: ({p[0]:8.3f}, {p[1]:7.3f}, {p[2]:.5f})")


def envelope(label, key, thr):
    ok = [s for s in SHIFTS if res[label][s][key] >= thr]
    return (min(ok), max(ok)) if ok else None


print("\n=== envelope of kinetic shift over which compensation succeeds ===")
for label in FORMS:
    for key, nm in (("gap", "mean squared error"), ("spike_similarity", "spike count")):
        line = f"  {label[:6]} on {nm:<19}: "
        for thr in (95, 90, 80):
            e = envelope(label, key, thr)
            line += f">={thr}%: " + (f"[{e[0]:+.1f},{e[1]:+.1f}] mV   " if e else "none   ")
        print(line)
    e = res[label][R859C]
    print(f"          at the measured R859C shift +{R859C} mV: "
          f"{e['gap']:.2f} % of the gap closed, "
          f"{e['spike_similarity']:.2f} % spike-count similarity\n")

with open("task13b_summary.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(f"total {time.time()-t00:.1f} s  -> task13b_summary.json")
