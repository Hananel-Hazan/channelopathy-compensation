"""Stability window around a DIRECTLY SEARCHED optimum on the spike-count objective.

No optimizer is used.  For each single-conductance variant (sodium, potassium or leak
conductance held at its variant value) the two free conductances are gridded over the
NEURON pipeline's sampling box (leak baseline/50 to baseline*50, sodium and potassium
baseline/5000 to baseline*5000), log-spaced so the whole box is covered, and every
point is evaluated on the functional objective directly -- spike count at each of 35
injected-current levels, scored on the pipeline's percentage scale
(wo5_lib.similarity).  The best point on the grid is the achievable optimum.

The stability window is then measured around that point with fractional perturbations
(each conductance alone, random joint perturbations, and a 201 x 201 grid for the worst
case within a given magnitude).  The multi-current voltage MSE of the optimum is
reported alongside.

Writes task11d_summary.json to the current directory.
"""
import json, os, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W
import wo5_fit as F

DEV = "cuda"
GRID = 400                     # per free axis -> 160,000 parameter sets per case
t00 = time.time()

CASES = [
    ("sodium variant, g_Na fixed at 140",  "Na", (140.0, 36.0, 0.03)),
    ("potassium variant, g_K fixed at 20", "K",  (120.0, 20.0, 0.03)),
    ("leak variant, g_leak fixed at 0.30", "l",  (120.0, 36.0, 0.30)),
]
SPAN = {"Na": 5000.0, "K": 5000.0, "l": 50.0}      # box: baseline/SPAN to baseline*SPAN

ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
I1 = torch.tensor(ALLC, dtype=torch.float32, device=DEV)


def spikes_of(P, chunk=8192):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    out = np.empty((P.shape[0], W.N_CUR), dtype=np.int16)
    for s0 in range(0, P.shape[0], chunk):
        e = min(s0 + chunk, P.shape[0]); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I1.repeat(nb), device=DEV)
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out


sc_wt = spikes_of([W.WT])[0].astype(int)
print(f"wild-type spike counts total {sc_wt.sum()}, "
      f"{int((sc_wt==0).sum())} silent of {W.N_CUR}\n")


def dist(P):
    return np.abs(spikes_of(P).astype(int) - sc_wt[None, :]).sum(axis=1)


tgt_fit = F.target_traces_multi(F.FIT_CURRENTS)
OUT = {"wt_spikes": sc_wt.tolist(), "grid": GRID, "cases": []}

for name, fz, untr in CASES:
    fi = [j for j, n in enumerate(W.NAMES) if n != fz]
    d_untr = int(dist([untr])[0])
    t0 = time.time()

    axes = [np.geomspace(untr[j] / SPAN[W.NAMES[j]], untr[j] * SPAN[W.NAMES[j]], GRID)
            for j in fi]
    AX, AY = np.meshgrid(axes[0], axes[1], indexing="ij")
    P = np.repeat(np.array(untr, dtype=float)[None, :], GRID * GRID, axis=0)
    P[:, fi[0]] = AX.ravel(); P[:, fi[1]] = AY.ravel()

    D = dist(P)
    b = int(np.argmin(D))
    p_opt = P[b]
    sim_opt = float(W.similarity(D[b], d_untr))
    n_tied = int((D == D[b]).sum())
    mse_opt = float(F.losses_multi([p_opt], tgt_fit, F.FIT_CURRENTS)[0])
    mse_untr = float(F.losses_multi([untr], tgt_fit, F.FIT_CURRENTS)[0])

    print(f"--- {name}")
    print(f"    untreated variant: spike distance {d_untr}, multi-current MSE {mse_untr:.2f}")
    print(f"    grid {GRID}x{GRID} = {GRID*GRID:,} parameter sets over the pipeline's box, "
          f"{time.time()-t0:.1f} s")
    print(f"    BEST reachable: distance {D[b]} -> similarity {sim_opt:.2f}%  "
          f"at (Na, K, l) = ({p_opt[0]:.4f}, {p_opt[1]:.4f}, {p_opt[2]:.5f})")
    print(f"    {n_tied:,} grid points tie at that distance; multi-current MSE there "
          f"{mse_opt:.2f} ({W.gap_closed(mse_opt, mse_untr):.1f}% of the MSE gap closed)")

    # ---- stability window around the directly searched optimum ----------------
    MAGS = np.arange(1, 51) / 100.0
    gn = 201
    fr = np.linspace(-0.5, 0.5, gn)
    GX, GY = np.meshgrid(fr, fr, indexing="ij")
    Gd = np.stack([GX.ravel(), GY.ravel()], axis=1)
    LINF = np.abs(Gd).max(axis=1)
    rng = np.random.default_rng(20260803)
    d1 = np.array([[sg * p if c == cc else 0.0 for c in range(2)]
                   for p in MAGS for cc in range(2) for sg in (+1, -1)])
    dj = (rng.uniform(-1, 1, size=(MAGS.size, 1000, 2)) * MAGS[:, None, None]).reshape(-1, 2)
    allD = np.concatenate([Gd, d1, dj], axis=0)
    Pp = np.repeat(p_opt[None, :], allD.shape[0], axis=0)
    for c, j in enumerate(fi):
        Pp[:, j] = p_opt[j] * (1.0 + allD[:, c])
    rel = W.similarity(dist(Pp), d_untr) / sim_opt * 100.0

    ng, n1 = Gd.shape[0], d1.shape[0]
    g_, o_, j_ = rel[:ng], rel[ng:ng+n1].reshape(MAGS.size, 4), rel[ng+n1:].reshape(MAGS.size, 1000)
    cur = {"worst": [float(g_[LINF <= p + 1e-12].min()) for p in MAGS],
           "one": o_.min(axis=1).tolist(),
           "joint_median": np.median(j_, axis=1).tolist(),
           "joint_p5": np.percentile(j_, 5, axis=1).tolist()}

    print(f"    {'+/- %':>6} | {'one-at-a-time':>14} {'joint median':>13} "
          f"{'joint 5th pct':>14} {'worst case':>11}")
    for k, p in enumerate(MAGS):
        if int(round(p*100)) not in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50):
            continue
        print(f"    {p*100:>5.0f}% | {cur['one'][k]:>13.1f}% {cur['joint_median'][k]:>12.1f}% "
              f"{cur['joint_p5'][k]:>13.1f}% {cur['worst'][k]:>10.1f}%")

    def fb(c, thr):
        b_ = np.where(np.asarray(c) <= thr)[0]
        return None if b_.size == 0 else float(MAGS[b_[0]] * 100)

    thr = {m: [fb(cur[m], x) for x in (95, 90, 80)] for m in cur}
    print("    window (first magnitude at or below the threshold):")
    for m, t in thr.items():
        f = lambda x: (f"{x:.0f}%" if x is not None else ">50%")
        print(f"      {m:<14} <95%: {f(t[0]):>5}   <90%: {f(t[1]):>5}   <80%: {f(t[2]):>5}")
    print()

    OUT["cases"].append(dict(name=name, frozen=fz, untreated=list(untr),
                             d_untreated=d_untr, mse_untreated=mse_untr,
                             optimum=list(map(float, p_opt)), d_optimum=int(D[b]),
                             similarity_optimum=sim_opt, n_tied=n_tied,
                             mse_optimum=mse_opt, curves=cur, thresholds=thr,
                             mags_pct=(MAGS*100).tolist()))

with open("task11d_summary.json", "w") as f:
    json.dump(OUT, f)
print(f"total {time.time()-t00:.1f} s  -> task11d_summary.json")
