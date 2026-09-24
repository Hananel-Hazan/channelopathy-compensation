"""Is the divergence of the explicit integrator the binding constraint?

When Adam runs on the voltage mean squared error, the explicit fixed-step
integrator goes non-finite for most of the parameter values the optimizer
proposes: in wo6_taskC.py the non-finite guard fired 1,407 times over 200
iterations on three problems, and the objective fell only 3 % to 6 % of its
starting value in the two failing cases.  That is an optimization failure, not
an objective failure.

But a guard that restores the last finite value also freezes the parameter, so
the run could be pathological in a way that overstates the problem.  This script
separates the two possibilities by removing the cause rather than patching the
symptom:

  - shrink the learning rate, so the optimizer proposes smaller jumps;
  - shrink the integration step, so the explicit method is stable further out.

If either makes the guard stop firing AND lets the objective fall properly AND
recovers the functional score, the remedy is numerical.  If the objective falls
but the functional score still does not follow, the remedy is a different
objective.

Needs a CUDA graphics card.  Writes wo6_taskC2.json in the current directory.
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
BASE_LRS = np.array([0.5, 0.15, 0.001])          # the per-conductance rates of wo6_taskC.py
ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--ms", type=float, default=300.0)
ap.add_argument("--out", default="wo6_taskC2.json")
args = ap.parse_args()

HOST = platform.node()
FITC = np.array([0.0, 3.0, 6.0, 7.0, 8.5, 11.0, 14.0, 17.0])
C = len(FITC)
I_ALL = torch.tensor(np.arange(W.N_CUR) * W.I_STEP + W.I_LOW, dtype=torch.float32, device=DEV)

CASES = [
    ("g_Na fixed 140",   (140., 36., .03), [False, True,  True ]),
    ("g_K fixed 20",     (120., 20., .03), [True,  False, True ]),
    ("g_leak fixed 0.30",(120., 36., .30), [True,  True,  False]),
]
CONFIGS = [        # label, integration step (ms), learning-rate multiplier
    ("dt 0.01 ms, learning rate x1     (as reported)", 0.01,  1.0),
    ("dt 0.01 ms, learning rate x0.3",                0.01,  0.3),
    ("dt 0.01 ms, learning rate x0.1",                0.01,  0.1),
    ("dt 0.01 ms, learning rate x0.03",               0.01,  0.03),
    ("dt 0.005 ms, learning rate x1",                 0.005, 1.0),
    ("dt 0.005 ms, learning rate x0.1",               0.005, 0.1),
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
print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}")
print(f"{args.ms:.0f} ms per trace, {args.iters} Adam iterations, "
      f"starting from the untreated variant\n")

OUT = {"host": HOST, "d_untreated": D_UNTR, "configs": []}
for label, dt, lrmul in CONFIGS:
    steps = int(round(args.ms / dt))
    with torch.no_grad():
        g = [torch.full((C,), float(x), dtype=torch.float32, device=DEV) for x in W.WT]
        TGT = O.traj(g[0], g[1], g[2], torch.tensor(FITC, dtype=torch.float32, device=DEV),
                     steps=steps, dt=dt, grad=False).unsqueeze(1)

    starts = np.array([c[1] for c in CASES]); masks = np.array([c[2] for c in CASES])
    P = starts.shape[0]
    I = torch.tensor(np.tile(FITC, P), dtype=torch.float32, device=DEV)
    par = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV, requires_grad=True)
           for j in range(3)]
    fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
    last = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
    keep = [torch.tensor(~masks[:, j], device=DEV) for j in range(3)]
    opt = torch.optim.Adam([{"params": [par[j]], "lr": float(BASE_LRS[j] * lrmul)}
                            for j in range(3)])

    n_guard = 0
    n_nonfinite_loss = 0
    best_obj = np.full(P, np.inf); best_p = starts.copy()
    o0 = None
    t0 = time.time()
    for it in range(args.iters):
        gg = [q.unsqueeze(1).expand(P, C).reshape(-1) for q in par]
        v = O.traj(gg[0], gg[1], gg[2], I, steps=steps, dt=dt).view(-1, P, C)
        loss = O.obj_voltage_mse(v, TGT)
        lv = loss.detach().double().cpu().numpy()
        if o0 is None:
            o0 = lv.copy()
        n_nonfinite_loss += int((~np.isfinite(lv)).sum())
        opt.zero_grad(set_to_none=True)
        torch.nan_to_num(loss, nan=1e12, posinf=1e12).sum().backward()
        opt.step()
        with torch.no_grad():
            for j in range(3):
                bad = ~torch.isfinite(par[j])
                if bool(bad.any()):
                    n_guard += int(bad.sum())
                    par[j].copy_(torch.where(bad, last[j], par[j]))
                par[j].clamp_(min=1e-6)
                par[j].copy_(torch.where(keep[j], fixed[j], par[j]))
                last[j].copy_(par[j])
        pv = np.stack([q.detach().double().cpu().numpy() for q in par], axis=1)
        imp = np.isfinite(lv) & (lv < best_obj)
        best_obj[imp] = lv[imp]; best_p[imp] = pv[imp]
    secs = time.time() - t0

    d = np.abs(hard_spikes(best_p) - SC_WT[None, :]).sum(axis=1)
    sims = [float(W.similarity(d[i], D_UNTR[i])) for i in range(P)]
    drops = [float((o0[i] - best_obj[i]) / o0[i] * 100) for i in range(P)]

    OUT["configs"].append(dict(label=label, dt=dt, lr_multiplier=lrmul, steps=steps,
                               guard_fired=n_guard, nonfinite_losses=n_nonfinite_loss,
                               objective_drop_pct=drops, similarity=sims,
                               params=best_p.tolist(), seconds=secs))
    print(f"--- {label}   [{secs:.0f} s]")
    print(f"    non-finite guard fired {n_guard:>6} times; "
          f"{n_nonfinite_loss} of {args.iters*P} loss evaluations were non-finite")
    for i, c in enumerate(CASES):
        print(f"    {c[0]:<20} objective fell {drops[i]:6.1f} %   "
              f"functional similarity {sims[i]:+7.2f} %")
    print(flush=True)

with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1)
print(f"-> {args.out}")
