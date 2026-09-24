"""Stability window of the fitted optimum under fractional conductance perturbations.

For three single-conductance variants (sodium, potassium or leak conductance held at
its variant value), the remaining two conductances are optimized against the wild type
across a RANGE of injected currents, and the optimum is then perturbed and re-scored.

  fit        8 injected-current levels, 300 ms each, 0.01 ms step, mean squared error
             on the voltage trace, Adam, 200 iterations, 12 random restarts per
             configuration, best restart kept.
  evaluate   all 35 levels -- so the evaluation levels are mostly held out from the fit
             -- by (i) the same multi-current mean squared error and (ii) the spike
             count at each level, scored on the NEURON pipeline's percentage scale
             (wo5_lib.similarity).

Perturbations are FRACTIONAL: parameter -> parameter * (1 + delta), applied only to the
optimized (pharmacologically accessible) conductances: each conductance alone, random
joint perturbations, and a 201 x 201 grid for the worst case within a given magnitude.

Writes task11c_summary.json to the current directory.
"""
import json, os, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W
import wo5_fit as F

DEV = "cuda"
N_RESTART = 12
ITERS = 200
t00 = time.time()

CASES = [
    ("sodium variant, g_Na fixed at 140",  "Na", (140.0, 36.0, 0.03)),
    ("potassium variant, g_K fixed at 20", "K",  (120.0, 20.0, 0.03)),
    ("leak variant, g_leak fixed at 0.30", "l",  (120.0, 36.0, 0.30)),
]

FITC = F.FIT_CURRENTS
ALLC = np.arange(W.N_CUR) * W.I_STEP + W.I_LOW
tgt_fit = F.target_traces_multi(FITC)
print(f"fitting on {len(FITC)} current levels {list(FITC)}")
print(f"evaluating on all {W.N_CUR} levels "
      f"{ALLC[0]:.1f} to {ALLC[-1]:.1f} step {W.I_STEP}\n")

# ---- spike-count machinery, evaluated on all 35 levels -----------------------
I1 = torch.tensor(ALLC, dtype=torch.float32, device=DEV)


def spike_vectors(P, chunk_sets=4096):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    out = np.empty((P.shape[0], W.N_CUR), dtype=np.int64)
    for s0 in range(0, P.shape[0], chunk_sets):
        e = min(s0 + chunk_sets, P.shape[0]); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats(g[0], g[1], g[2], I1.repeat(nb), device=DEV)
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out


sc_wt = spike_vectors([W.WT])[0]


def sc_distance(P):
    return np.abs(spike_vectors(P) - sc_wt[None, :]).sum(axis=1)


# ---- converge each configuration, with restarts ------------------------------
rng = np.random.default_rng(20260803)
starts, masks, owner = [], [], []
for i, (name, fz, untr) in enumerate(CASES):
    fi = [j for j, n in enumerate(W.NAMES) if n != fz]
    for r in range(N_RESTART):
        s = np.array(untr, dtype=float)
        if r > 0:                                   # restart 0 is the untreated variant
            for j in fi:
                s[j] = s[j] * 10 ** rng.uniform(-0.5, 0.5)
        starts.append(s)
        masks.append([n != fz for n in W.NAMES])
        owner.append(i)
starts = np.array(starts); masks = np.array(masks); owner = np.array(owner)

print(f"converging {len(CASES)} configurations x {N_RESTART} restarts "
      f"({ITERS} Adam iterations)...")
t0 = time.time()
fitres = F.fit_batch_multi(starts, masks, tgt_fit, FITC, iters=ITERS)
print(f"  {time.time()-t0:.1f} s\n")

L_untr = F.losses_multi([c[2] for c in CASES], tgt_fit, FITC)
d_untr = sc_distance([c[2] for c in CASES])

opt, gapM, simS, dS = [], [], [], []
print(f"{'configuration':<36} {'gap closed (MSE)':>17} {'spike similarity':>17}  optimum (Na, K, l)")
for i, (name, fz, untr) in enumerate(CASES):
    sel = np.where(owner == i)[0]
    b = sel[np.argmin(fitres["best_loss"][sel])]
    p = fitres["best_params"][b]
    opt.append(p)
    gapM.append(float(W.gap_closed(fitres["best_loss"][b], L_untr[i])))
    d = int(sc_distance([p])[0]); dS.append(d)
    simS.append(float(W.similarity(d, d_untr[i])))
    print(f"{name:<36} {gapM[-1]:>16.2f}% {simS[-1]:>16.2f}%  "
          f"({p[0]:8.3f}, {p[1]:7.3f}, {p[2]:.5f})")
    print(f"{'':<36} untreated: MSE {L_untr[i]:.2f}, spike distance {d_untr[i]} "
          f"-> optimum spike distance {d}")

# ---- perturbation ------------------------------------------------------------
MAGS = np.arange(1, 51) / 100.0
GN = 201
fr = np.linspace(-0.5, 0.5, GN)
GX, GY = np.meshgrid(fr, fr, indexing="ij")
GRID = np.stack([GX.ravel(), GY.ravel()], axis=1)
LINF = np.abs(GRID).max(axis=1)
N_JOINT = 1000


def perturbed(params, fi, deltas):
    P = np.repeat(np.asarray(params, float)[None, :], deltas.shape[0], axis=0)
    for c, j in enumerate(fi):
        P[:, j] = P[:, j] * (1.0 + deltas[:, c])
    return P


OUT = {"cases": [], "mags_pct": (MAGS * 100).tolist()}
for i, (name, fz, untr) in enumerate(CASES):
    fi = [j for j, n in enumerate(W.NAMES) if n != fz]
    p0 = opt[i]
    t0 = time.time()

    d1 = np.array([[sg * p if c == cc else 0.0 for c in range(2)]
                   for p in MAGS for cc in range(2) for sg in (+1, -1)])
    dj = (rng.uniform(-1, 1, size=(MAGS.size, N_JOINT, 2)) * MAGS[:, None, None]
          ).reshape(-1, 2)
    allD = np.concatenate([GRID, d1, dj], axis=0)
    allP = perturbed(p0, fi, allD)

    Lm = F.losses_multi(allP, tgt_fit, FITC)
    relM = W.gap_closed(Lm, L_untr[i]) / gapM[i] * 100.0
    relS = (W.similarity(sc_distance(allP), d_untr[i]) / simS[i] * 100.0
            if simS[i] > 1e-9 else np.full(allP.shape[0], np.nan))

    ng, n1 = GRID.shape[0], d1.shape[0]
    curves = {}
    for tag, rel in (("MSE", relM), ("spike", relS)):
        g_, o_, j_ = rel[:ng], rel[ng:ng+n1].reshape(MAGS.size, 4), rel[ng+n1:].reshape(MAGS.size, N_JOINT)
        curves[tag] = {
            "worst": [float(np.nanmin(g_[LINF <= p + 1e-12])) for p in MAGS],
            "one": np.nanmin(o_, axis=1).tolist(),
            "joint_median": np.nanmedian(j_, axis=1).tolist(),
            "joint_p5": np.nanpercentile(j_, 5, axis=1).tolist()}

    OUT["cases"].append(dict(name=name, frozen=fz, untreated=list(untr),
                             optimum=list(map(float, p0)), gap_mse=gapM[i],
                             sim_spike=simS[i], d_untreated=int(d_untr[i]),
                             d_optimum=int(dS[i]), curves=curves))
    print(f"\n[{name}]  {allP.shape[0]:,} perturbed parameter sets, "
          f"both measures, {time.time()-t0:.1f} s")
    print(f"  {'+/- %':>6} | {'MSE one':>9} {'MSE joint50':>12} {'MSE worst':>10} | "
          f"{'spk one':>9} {'spk joint50':>12} {'spk worst':>10}")
    for k, p in enumerate(MAGS):
        if int(round(p*100)) not in (1, 2, 3, 5, 8, 10, 15, 20, 30, 50):
            continue
        c1, c2 = curves["MSE"], curves["spike"]
        print(f"  {p*100:>5.0f}% | {c1['one'][k]:>8.1f}% {c1['joint_median'][k]:>11.1f}% "
              f"{c1['worst'][k]:>9.1f}% | {c2['one'][k]:>8.1f}% "
              f"{c2['joint_median'][k]:>11.1f}% {c2['worst'][k]:>9.1f}%")


def first_below(curve, thr):
    c = np.asarray(curve, dtype=float)
    b = np.where(c <= thr)[0]
    return None if b.size == 0 else float(MAGS[b[0]] * 100)


print("\n\n=== the stability window: largest perturbation keeping efficacy above ... ===")
print(f"{'configuration':<36} {'measure':<7} {'mode':<13} {'95%':>6} {'90%':>6} {'80%':>6}")
for c in OUT["cases"]:
    for tag in ("MSE", "spike"):
        for mode in ("one", "joint_median", "joint_p5", "worst"):
            t = [first_below(c["curves"][tag][mode], x) for x in (95, 90, 80)]
            c["curves"][tag][mode + "_thr"] = t
            f = lambda x: (f"{x:.0f}%" if x is not None else ">50%")
            print(f"{c['name']:<36} {tag:<7} {mode:<13} "
                  f"{f(t[0]):>6} {f(t[1]):>6} {f(t[2]):>6}")

with open("task11c_summary.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(f"\ntotal {time.time()-t00:.1f} s  -> task11c_summary.json")
