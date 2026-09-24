"""Does any differentiable objective let gradient descent find the compensating parameters?

Seven cases: three conductance variants (the mutated conductance held fixed, the other
two free) and four kinetic variants (the gating shift held fixed, all three conductances
free).  Each case carries the similarity reached by direct search as a benchmark.

Every objective in wo6_lib.OBJECTIVES is optimized by Adam from several random restarts,
and every result is then scored on ONE common functional measure that no objective sees
during optimization: the spike-count distance to wild type across all 35 injected-current
levels, expressed on the similarity percentage scale of the search pipeline
(`wo5_lib.similarity`).  The optimized objective and the scoring measure are kept
separate.

Usage:  python wo6_taskA.py [--restarts 6] [--iters 200] [--objectives a,b] [--out FILE]
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(REPO, "experiments", "wo5"), os.path.join(REPO, "experiments", "wo6")):
    sys.path.insert(0, _p)
import wo5_lib as W
import wo6_lib as O

DEV = "cuda"
LRS = {"Na": 0.5, "K": 0.15, "l": 0.001}       # Adam learning rate per conductance
CLAMP = 1e-6

# name, untreated (Na,K,l), free mask, sodium shift, potassium shift, direct-search benchmark
CASES = [
    ("conductance, g_Na fixed 140",   (140., 36., .03), [False, True,  True ], 0.0, 0.0, 96.15),
    ("conductance, g_K fixed 20",     (120., 20., .03), [True,  False, True ], 0.0, 0.0, 93.97),
    ("conductance, g_leak fixed 0.30",(120., 36., .30), [True,  True,  False], 0.0, 0.0, 95.02),
    ("kinetic, sodium +6.1 mV",       (120., 36., .03), [True,  True,  True ], 6.1, 0.0, 97.60),
    ("kinetic, potassium +4.7 mV",    (120., 36., .03), [True,  True,  True ], 0.0, 4.7, 99.35),
    ("kinetic, potassium +9.4 mV",    (120., 36., .03), [True,  True,  True ], 0.0, 9.4, 99.05),
    ("kinetic, both shifts",          (120., 36., .03), [True,  True,  True ], 6.1, 4.7, 95.28),
]

ap = argparse.ArgumentParser()
ap.add_argument("--restarts", type=int, default=6)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--steps", type=int, default=W.STEPS_EXT, help="integration steps for the FIT")
ap.add_argument("--objectives", default="")
ap.add_argument("--out", default="wo6_taskA.json")
args = ap.parse_args()

HOST = platform.node()
FITC = np.array([0.0, 3.0, 6.0, 7.0, 8.5, 11.0, 14.0, 17.0])
ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
C = len(FITC)
I_ALL = torch.tensor(ALLC, dtype=torch.float32, device=DEV)


def hard_spikes(P, shift, shift_n, chunk=4096):
    """The pipeline's own spike count, on all 35 levels. Used only for SCORING."""
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.empty((P.shape[0], W.N_CUR), int)
    for s0 in range(0, P.shape[0], chunk):
        e = min(s0 + chunk, P.shape[0]); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I_ALL.repeat(nb),
                                   shift=float(shift), shift_n=float(shift_n),
                                   device=DEV, steps_per_call=20)
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out


SC_WT = hard_spikes([W.WT], 0.0, 0.0)[0]
print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}")
print(f"wild type fires {SC_WT.sum()} action potentials over {W.N_CUR} current levels")
print(f"fit on {C} currents, {args.steps} steps ({args.steps*W.DT_EXT:.0f} ms), "
      f"{args.restarts} restarts, {args.iters} Adam iterations\n")

# untreated variant of each case, for the similarity denominator
D_UNTR = [int(np.abs(hard_spikes([c[1]], c[3], c[4])[0] - SC_WT).sum()) for c in CASES]
for c, d in zip(CASES, D_UNTR):
    print(f"  {c[0]:<34} untreated spike-count distance to wild type {d}")

# wild-type target at the fit currents
with torch.no_grad():
    g = [torch.full((C,), float(x), dtype=torch.float32, device=DEV) for x in W.WT]
    TGT = O.traj(g[0], g[1], g[2], torch.tensor(FITC, dtype=torch.float32, device=DEV),
                 steps=args.steps, grad=False).unsqueeze(1)          # (T, 1, C)
print(f"\ntarget trace {tuple(TGT.shape)}")


def run_objective(name, fn, kw):
    """Fit every case x restart in one batch under this objective."""
    rng = np.random.default_rng(20260804)
    starts, masks, sm, sn, owner = [], [], [], [], []
    for ci, (nm, untr, mask, s_m, s_n, _) in enumerate(CASES):
        for r in range(args.restarts):
            p = np.array(untr, float)
            if r > 0:
                for j in range(3):
                    if mask[j]:
                        p[j] = p[j] * 10 ** rng.uniform(-0.6, 0.6)
            starts.append(p); masks.append(mask); sm.append(s_m); sn.append(s_n)
            owner.append(ci)
    starts = np.array(starts); masks = np.array(masks)
    P = starts.shape[0]
    owner = np.array(owner)

    I = torch.tensor(np.tile(FITC, P), dtype=torch.float32, device=DEV)
    sh_m = torch.tensor(np.repeat(sm, C), dtype=torch.float32, device=DEV)
    sh_n = torch.tensor(np.repeat(sn, C), dtype=torch.float32, device=DEV)
    par = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV, requires_grad=True)
           for j in range(3)]
    fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
    keep = [torch.tensor(~masks[:, j], device=DEV) for j in range(3)]
    opt = torch.optim.Adam([{"params": [par[j]], "lr": LRS[W.NAMES[j]]} for j in range(3)])

    best = np.full(P, np.inf); best_p = starts.copy()
    t0 = time.time()
    for it in range(args.iters):
        gg = [q.unsqueeze(1).expand(P, C).reshape(-1) for q in par]
        v = O.traj(gg[0], gg[1], gg[2], I, steps=args.steps, shift=sh_m, shift_n=sh_n)
        v = v.view(-1, P, C)
        loss = fn(v, TGT, **kw)
        loss = torch.nan_to_num(loss, nan=1e12, posinf=1e12, neginf=1e12)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            for j in range(3):
                par[j].clamp_(min=CLAMP)
                par[j].copy_(torch.where(keep[j], fixed[j], par[j]))
        lv = loss.detach().double().cpu().numpy()
        pv = np.stack([q.detach().double().cpu().numpy() for q in par], axis=1)
        imp = np.isfinite(lv) & (lv < best)
        best[imp] = lv[imp]; best_p[imp] = pv[imp]
    fit_s = time.time() - t0

    # Score EVERY restart on the common functional measure, then report two selection
    # rules, because they answer different questions:
    #   by_objective -- the restart with the lowest value of the objective being
    #                   optimized. This is what a gradient-only pipeline can actually
    #                   pick, because the functional measure is not available to it.
    #   by_function  -- the best restart judged by the functional measure. This treats
    #                   the gradient pathway as a candidate GENERATOR and something else
    #                   as the selector.
    res = []
    for ci, (nm, untr, mask, s_m, s_n, bench) in enumerate(CASES):
        sel = np.where(owner == ci)[0]
        d = np.abs(hard_spikes(best_p[sel], s_m, s_n) - SC_WT[None, :]).sum(axis=1)
        sims = W.similarity(d, D_UNTR[ci])
        b_obj = int(np.argmin(best[sel]))
        b_fun = int(np.argmax(sims))
        res.append(dict(case=nm, benchmark=bench,
                        sim_by_objective=float(sims[b_obj]),
                        sim_by_function=float(sims[b_fun]),
                        params_by_objective=list(map(float, best_p[sel][b_obj])),
                        params_by_function=list(map(float, best_p[sel][b_fun])),
                        all_similarities=list(map(float, sims)),
                        all_objectives=list(map(float, best[sel]))))
    return res, fit_s


chosen = ([k for k in O.OBJECTIVES if any(t in k for t in args.objectives.split(","))]
          if args.objectives else list(O.OBJECTIVES))
OUT = {"host": HOST, "restarts": args.restarts, "iters": args.iters,
       "fit_steps": args.steps, "wt_spikes": SC_WT.tolist(),
       "d_untreated": D_UNTR, "objectives": {}}

for name in chosen:
    fn, kw = O.OBJECTIVES[name]
    print(f"\n=== {name} ===", flush=True)
    try:
        res, secs = run_objective(name, fn, kw)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        OUT["objectives"][name] = {"failed": f"{type(e).__name__}: {str(e)[:300]}"}
        continue
    OUT["objectives"][name] = {"seconds": secs, "results": res}
    print(f"  {'case':<32} {'by objective':>13} {'by function':>12} "
          f"{'direct search':>14} {'shortfall':>10}")
    for r in res:
        print(f"  {r['case']:<32} {r['sim_by_objective']:>12.2f}% "
              f"{r['sim_by_function']:>11.2f}% {r['benchmark']:>13.2f}% "
              f"{r['benchmark']-r['sim_by_objective']:>9.2f}")
    ok_o = sum(1 for r in res if r["sim_by_objective"] >= r["benchmark"] - 5.0)
    ok_f = sum(1 for r in res if r["sim_by_function"] >= r["benchmark"] - 5.0)
    print(f"  -> within 5 points of direct search: {ok_o} of {len(res)} selecting by the "
          f"objective, {ok_f} of {len(res)} selecting by the functional measure"
          f"   [{secs:.0f} s]", flush=True)

with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1)
print(f"\n-> {args.out}")
