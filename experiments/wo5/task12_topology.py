"""Solution-set topology (band versus plane) for several kinetic variant types.

Solution sets in conductance space can be band-like (elongated along one direction) or
planar (spread over a surface).  A band is automatically also nearly planar, so the
DISCRIMINATING quantity is the share of variation along the SECOND principal direction,
not the first two together.

This script defines several variant types on the differentiable model, sweeps the
conductance space for each, and reports the principal-direction shares of the best
solutions.

Sampling reproduces the NEURON pipeline's scheme:
  - the box  [baseline/50, baseline*50]      for the leak conductance
             [baseline/5000, baseline*5000]  for sodium and potassium
    drawn uniformly (the pipeline's scheme), log-uniformly, or on a log-spaced grid
    (--mode)
  - parameter sets that produce no action potential at any current level are discarded,
    as the pipeline does

Selection is by spike-count distance to the wild type over all 35 injected-current
levels; two clusters are analysed: the top 0.1 % by that distance, and the set tied at
the minimum.

Usage:  python task12_topology.py [--mode grid|uniform|loguniform] [--out FILE]
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W

DEV = "cuda"

# Variant types.  Every one is a KINETIC alteration, because that is what both real case
# studies (R859C and Kv7.2 D212G) are, and because it is the only setting in which the
# topology question has content: if the variant were a change to a CONDUCTANCE and the
# search ranged over all three conductances, the search could simply set that
# conductance back and the "variant" would not be a variant at all.  A conductance
# variant must instead be searched with that conductance held fixed, which confines the
# solution set to a coordinate plane by construction and makes "planar" circular.
#
# Magnitudes are read off the two published mechanisms rather than chosen:
#   sodium    ichanWT2005.mod vs ichanR859C1.mod -- minf half-activation
#             -27.4 -> -21.3 mV, a depolarizing shift of +6.1 mV  (ModelDB 87585)
#   potassium kmtwt.mod vs kmquad.mod -- activation half-point
#             -32.4 -> -27.7 mV, a depolarizing shift of +4.7 mV  (ModelDB 118986)
VARIANTS = {
    "sodium activation +6.1 mV (R859C, ModelDB 87585)":     dict(shift=6.1, shift_n=0.0),
    "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)": dict(shift=0.0, shift_n=4.7),
    "potassium activation +9.4 mV (twice the measured shift)":    dict(shift=0.0, shift_n=9.4),
    "combined: sodium +6.1 mV and potassium +4.7 mV":       dict(shift=6.1, shift_n=4.7),
}

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=5_000_000)
ap.add_argument("--seed-lo", type=int, default=0)
ap.add_argument("--seed-hi", type=int, default=1)
ap.add_argument("--chunk", type=int, default=16384, help="parameter sets per call")
ap.add_argument("--steps-per-call", type=int, default=20,
                help="integration steps per fused block; 100 makes an Inductor graph "
                     "that takes longer to compile than the sweep takes to run")
ap.add_argument("--mode", choices=("uniform", "loguniform", "grid"), default="grid",
                help="uniform reproduces the NEURON pipeline exactly; loguniform covers "
                     "the same box but spends draws evenly across decades; grid is a "
                     "deterministic dense log-spaced lattice over the same box")
ap.add_argument("--grid-n", type=int, default=200, help="points per axis in grid mode")
ap.add_argument("--out", default="task12_summary.json")
args = ap.parse_args()

HOST = platform.node()
I1 = torch.tensor(np.arange(W.N_CUR) * W.I_STEP + W.I_LOW, dtype=torch.float32, device=DEV)
LO = np.array([W.WT[0] / 5000, W.WT[1] / 5000, W.WT[2] / 50])
HI = np.array([W.WT[0] * 5000, W.WT[1] * 5000, W.WT[2] * 50])


def spikes_of(P, shift, chunk, shift_n=0.0):
    """P is (N,3). Returns (N,35) spike counts."""
    N = P.shape[0]
    out = np.empty((N, W.N_CUR), dtype=np.int16)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I1.repeat(nb), shift=float(shift),
                                   device=DEV, steps_per_call=args.steps_per_call,
                                   shift_n=float(shift_n))
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out


def geometry(P):
    """Cluster geometry: normalise each axis by its own
    range, then report the share of variation carried by each principal direction and
    the distance from the best-fit plane as a fraction of the cluster's own size."""
    X = np.asarray(P, dtype=float)
    rng_ = X.max(axis=0) - X.min(axis=0)
    rng_[rng_ == 0] = 1.0
    Xn = X / rng_
    Xc = Xn - Xn.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum() * 100.0
    normal = Vt[2]
    d = np.abs(Xc @ normal)
    extent = np.linalg.norm(Xc, axis=1).max()
    return dict(pc1=float(var[0]), pc2=float(var[1]), pc3=float(var[2]),
                pc12=float(var[0] + var[1]),
                plane_residual_pct=float(d.mean() / extent * 100.0))


print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}")
print(f"sampling box (Na, K, leak): {LO} to {HI}")
print(f"{args.samples:,} samples per variant type, seeds {args.seed_lo}..{args.seed_hi-1}\n")

sc_wt = spikes_of(np.array([W.WT]), 0.0, 64)[0].astype(int)
print(f"wild-type spike counts: total {sc_wt.sum()}, "
      f"{int((sc_wt == 0).sum())} silent levels of {W.N_CUR}\n")

RESULT = {"host": HOST, "mode": args.mode, "gpu": torch.cuda.get_device_properties(0).name,
          "samples_per_variant": args.samples, "wt_spikes": sc_wt.tolist(),
          "variants": {}}

for vname, spec in VARIANTS.items():
    t0 = time.time()
    base = np.array(W.WT)          # the variant carries wild-type conductances; only its
                                   # gating kinetics differ, exactly as both real models do
    d_untr = int(np.abs(spikes_of(base[None, :], spec["shift"], 64,
                                  spec["shift_n"])[0].astype(int) - sc_wt).sum())

    keptP, keptD = [], []
    counts = [0, 0]                # drawn, silent -- a list because `absorb` is defined
                                   # at module scope, where `nonlocal` has no binding

    def absorb(P):
        sc = spikes_of(P, spec["shift"], args.chunk, spec["shift_n"]).astype(int)
        counts[0] += P.shape[0]
        alive = sc.sum(axis=1) > 0                     # the pipeline's filter: drop silent sets
        counts[1] += int((~alive).sum())
        if alive.any():
            keptP.append(P[alive])
            keptD.append(np.abs(sc[alive] - sc_wt[None, :]).sum(axis=1))

    if args.mode == "grid":
        g = args.grid_n
        ax = [np.geomspace(LO[j], HI[j], g) for j in range(3)]
        A, B_, C = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
        allP = np.stack([A.ravel(), B_.ravel(), C.ravel()], axis=1)
        for s0 in range(0, allP.shape[0], args.chunk * 8):
            absorb(allP[s0:s0 + args.chunk * 8])
    else:
        for seed in range(args.seed_lo, args.seed_hi):
            rng = np.random.default_rng(1000 + seed)
            todo = args.samples // (args.seed_hi - args.seed_lo)
            while todo > 0:
                nb = min(args.chunk * 8, todo)
                if args.mode == "uniform":
                    P = rng.uniform(LO, HI, size=(nb, 3))
                else:
                    P = np.exp(rng.uniform(np.log(LO), np.log(HI), size=(nb, 3)))
                absorb(P)
                todo -= nb

    n_drawn, n_silent = counts
    P = np.concatenate(keptP); D = np.concatenate(keptD)
    order = np.argsort(D, kind="stable")
    n_top = max(50, int(round(0.001 * P.shape[0])))
    top = order[:n_top]
    tied = np.where(D == D.min())[0]

    g_top = geometry(P[top])
    g_tied = geometry(P[tied]) if tied.size >= 4 else None
    best_sim = float(W.similarity(D.min(), d_untr)) if d_untr else float("nan")

    RESULT["variants"][vname] = dict(
        baseline=base.tolist(), shift=spec["shift"], shift_n=spec["shift_n"],
        d_untreated=d_untr, mode=args.mode,
        n_drawn=int(n_drawn), n_spiking=int(P.shape[0]), n_silent=int(n_silent),
        best_distance=int(D.min()), n_tied_at_min=int(tied.size),
        best_similarity_pct=best_sim, top_n=int(n_top),
        geom_top=g_top, geom_tied=g_tied,
        top_points=P[top[:2000]].tolist())

    print(f"--- {vname}")
    print(f"    untreated variant spike distance to wild type : {d_untr}")
    print(f"    drawn {n_drawn:,} | spiking {P.shape[0]:,} "
          f"({100*P.shape[0]/n_drawn:.1f} %) | discarded silent {n_silent:,}")
    print(f"    best distance reached {D.min()} ({best_sim:.2f} % similarity), "
          f"{tied.size:,} tied there")
    print(f"    top 0.1 % ({n_top:,} points): "
          f"first {g_top['pc1']:.2f} %, SECOND {g_top['pc2']:.2f} %, "
          f"third {g_top['pc3']:.2f} %, first two {g_top['pc12']:.2f} %, "
          f"plane residual {g_top['plane_residual_pct']:.2f} %")
    if g_tied:
        print(f"    tied at minimum ({tied.size:,}): "
              f"first {g_tied['pc1']:.2f} %, SECOND {g_tied['pc2']:.2f} %, "
              f"third {g_tied['pc3']:.2f} %, plane residual "
              f"{g_tied['plane_residual_pct']:.2f} %")
    print(f"    {time.time()-t0:.1f} s\n")

with open(args.out, "w") as f:
    json.dump(RESULT, f)
print(f"-> {args.out}")
