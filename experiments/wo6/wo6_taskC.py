"""Optimization failure or objective failure?

In two conductance cases (g_K held fixed, g_leak held fixed), gradient descent on the
voltage mean squared error ends up WORSE than no treatment on the functional measure.
There are two possible explanations, and they need different remedies: the optimizer may
have failed to minimise its objective, or it may have minimised its objective perfectly
well and the objective may simply have been the wrong one.

The two are distinguished by watching both quantities at once.  This script runs Adam on
voltage mean squared error, across eight injected currents, from the untreated variant,
and records, every few iterations, BOTH the objective being minimised AND the functional
score (spike-count similarity across all 35 current levels, on the search pipeline's
similarity scale) that no part of the optimizer can see.

  objective falls, functional score falls   -> OBJECTIVE failure. Fix the objective.
  objective does not fall                   -> OPTIMIZATION failure. Fix the optimizer.

The g_Na case, in which gradient descent succeeds, is included as a control.

Usage:  python wo6_taskC.py [--iters 200] [--every 10] [--out FILE]
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo6"))
import wo5_lib as W
import wo6_lib as O

DEV = "cuda"
LRS = {"Na": 0.5, "K": 0.15, "l": 0.001}
ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--every", type=int, default=10)
ap.add_argument("--out", default="wo6_taskC.json")
args = ap.parse_args()

HOST = platform.node()
FITC = np.array([0.0, 3.0, 6.0, 7.0, 8.5, 11.0, 14.0, 17.0])
ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
C = len(FITC)
I_ALL = torch.tensor(ALLC, dtype=torch.float32, device=DEV)

CASES = [
    ("g_Na fixed at 140  (this one SUCCEEDED)", (140., 36., .03), [False, True,  True ]),
    ("g_K fixed at 20    (this one failed)",    (120., 20., .03), [True,  False, True ]),
    ("g_leak fixed at 0.30 (this one failed)",  (120., 36., .30), [True,  True,  False]),
]


def hard_spikes(P):
    P = np.asarray(P, float).reshape(-1, 3)
    blk = np.repeat(P, W.N_CUR, axis=0)
    g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
    st = W.simulate_stats_fast(g[0], g[1], g[2], I_ALL.repeat(P.shape[0]),
                               device=DEV, steps_per_call=20)
    return st["spikes"].cpu().numpy().reshape(P.shape[0], W.N_CUR).astype(int)


SC_WT = hard_spikes([W.WT])[0]
D_UNTR = [int(np.abs(hard_spikes([c[1]])[0] - SC_WT).sum()) for c in CASES]

with torch.no_grad():
    g = [torch.full((C,), float(x), dtype=torch.float32, device=DEV) for x in W.WT]
    TGT = O.traj(g[0], g[1], g[2], torch.tensor(FITC, dtype=torch.float32, device=DEV),
                 grad=False).unsqueeze(1)

print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}")
print(f"Adam on voltage mean squared error, {C} injected currents, {args.iters} "
      f"iterations, starting from the untreated variant\n")

starts = np.array([c[1] for c in CASES])
masks = np.array([c[2] for c in CASES])
P = starts.shape[0]
I = torch.tensor(np.tile(FITC, P), dtype=torch.float32, device=DEV)
par = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV, requires_grad=True)
       for j in range(3)]
fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
keep = [torch.tensor(~masks[:, j], device=DEV) for j in range(3)]
opt = torch.optim.Adam([{"params": [par[j]], "lr": LRS[W.NAMES[j]]} for j in range(3)])

hist = {c[0]: {"iter": [], "objective": [], "similarity": [], "params": []} for c in CASES}
last_good = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
n_guard = [0, 0, 0]
t0 = time.time()
for it in range(args.iters + 1):
    gg = [q.unsqueeze(1).expand(P, C).reshape(-1) for q in par]
    v = O.traj(gg[0], gg[1], gg[2], I).view(-1, P, C)
    loss = O.obj_voltage_mse(v, TGT)
    if it % args.every == 0 or it == args.iters:
        pv = np.stack([q.detach().double().cpu().numpy() for q in par], axis=1)
        d = np.abs(hard_spikes(pv) - SC_WT[None, :]).sum(axis=1)
        lv = loss.detach().double().cpu().numpy()
        for ci, c in enumerate(CASES):
            hist[c[0]]["iter"].append(it)
            hist[c[0]]["objective"].append(float(lv[ci]))
            hist[c[0]]["similarity"].append(float(W.similarity(d[ci], D_UNTR[ci])))
            hist[c[0]]["params"].append(list(map(float, pv[ci])))
    if it == args.iters:
        break
    opt.zero_grad(set_to_none=True)
    torch.nan_to_num(loss, nan=1e12, posinf=1e12).sum().backward()
    opt.step()
    with torch.no_grad():
        for j in range(3):
            # Non-finite guard. The explicit integrator can diverge mid-optimization,
            # and once a parameter becomes not-a-number a clamp cannot repair it --
            # every comparison against NaN is false, so clamp_ leaves it NaN and the run
            # is dead from that iteration on. The last finite value is restored instead.
            bad = ~torch.isfinite(par[j])
            if bool(bad.any()):
                n_guard[j] += int(bad.sum())
                par[j].copy_(torch.where(bad, last_good[j], par[j]))
            par[j].clamp_(min=1e-6)
            par[j].copy_(torch.where(keep[j], fixed[j], par[j]))
            last_good[j].copy_(par[j])

for ci, c in enumerate(CASES):
    h = hist[c[0]]
    o, s = np.array(h["objective"]), np.array(h["similarity"])
    print(f"--- {c[0]}   (untreated spike distance {D_UNTR[ci]})")
    print(f"    {'iter':>6} {'voltage MSE':>13} {'functional similarity':>22}")
    for k in range(len(h["iter"])):
        if h["iter"][k] % (args.every * 4) == 0 or k == len(h["iter"]) - 1:
            print(f"    {h['iter'][k]:>6} {o[k]:>13.3f} {s[k]:>21.2f}%")
    fin = np.isfinite(o)
    o_f = o[fin]
    drop = (o[0] - o_f.min()) / o[0] * 100 if (o[0] > 0 and o_f.size) else float("nan")
    print(f"    objective fell by {drop:.1f} % of its starting value "
          f"({o[0]:.2f} -> {o_f.min():.2f}); {int((~fin).sum())} of {o.size} recorded "
          f"iterates were non-finite")
    print(f"    functional similarity {s[0]:+.2f}% -> {s[-1]:+.2f}%  "
          f"(best along the way {s.max():+.2f}%)")
    k_best = int(np.nanargmin(np.where(fin, o, np.inf)))
    print(f"    at the iterate with the LOWEST objective (iteration {h['iter'][k_best]}): "
          f"objective {o[k_best]:.2f}, functional similarity {s[k_best]:+.2f}%")
    verdict = ("OBJECTIVE failure: the optimizer did minimise its loss, and the "
               "configuration it chose is functionally no better"
               if drop > 20 and s[k_best] < 50 else
               "OPTIMIZATION failure: the loss did not fall materially"
               if drop <= 20 else
               "both the loss and the function improved")
    print(f"    -> {verdict}\n")

with open(args.out, "w") as f:
    json.dump({"host": HOST, "d_untreated": D_UNTR, "history": hist,
               "guard_fired": n_guard}, f, indent=1)
print(f"non-finite guard fired {sum(n_guard)} times (per parameter: {n_guard})")
print(f"{time.time()-t0:.0f} s  -> {args.out}")
