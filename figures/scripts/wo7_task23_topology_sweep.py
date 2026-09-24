"""Geometry of the solution set as a function of the selection threshold.

For each of four channel variants, every point of a log-spaced grid_n^3 lattice
over the conductance box (gNa, gK, gLeak) is simulated across the 35-level current
ladder.  Candidates that never spike are discarded; the rest are scored by their
integer spike-count distance to wild type.  The spiking candidates are then
selected at a log-spaced sequence of ~40 distance cuts, from the set tied at the
minimum distance up to the loosest 10 % of the spiking candidates, and at each cut
the shape of the selected point cloud is summarised by a principal-direction
decomposition (share of variance on each direction, distance from the best-fit
plane, and which axis carries the first direction), in linear and in log10
conductance space.

Point clouds are also saved at three representative thresholds (tied at the
minimum, top 0.1 %, top 10 %) for the scatter panels.

Output: wo7_topology_sweep.json next to this script, read by
wo7_task23_topology_plot.py and by wo7_task22_sampling_run.py (best value per
variant).  Requires a CUDA GPU.
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

# A ragged final chunk presents torch.compile with a new tensor shape; after its
# default recompile limit (8) is reached it falls back to eager execution, which
# is 20-200x slower.  Two defences: pad every call to one constant shape (in
# spikes_of below), and raise the limit.
for _attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _attr):
        setattr(torch._dynamo.config, _attr, 64)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W

DEV = "cuda"

# Axis order is (sodium, potassium, leak), matching wo5_lib.WT = (120.0, 36.0,
# 0.03) and the box built at LO/HI below.  The baselines are four orders
# of magnitude apart, so a linear decomposition of a set spanning this box tends
# to be dominated by the largest axis; the decomposition is therefore also
# computed in log space, and the axis carrying the first direction is recorded.
AXES = ("sodium", "potassium", "leak")
AXIS_BASELINE = (120.0, 36.0, 0.03)

VARIANTS = {
    "sodium activation +6.1 mV (R859C, ModelDB 87585)":           dict(shift=6.1, shift_n=0.0),
    "potassium activation +4.7 mV (Kv7.2 D212G, ModelDB 118986)": dict(shift=0.0, shift_n=4.7),
    "potassium activation +9.4 mV (twice the measured shift)":     dict(shift=0.0, shift_n=9.4),
    "combined: sodium +6.1 mV and potassium +4.7 mV":              dict(shift=6.1, shift_n=4.7),
}

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--steps-per-call", type=int, default=20)
ap.add_argument("--grid-n", type=int, default=200)
ap.add_argument("--n-thresholds", type=int, default=40)
ap.add_argument("--out", default=os.path.join(REPO, "figures", "scripts", "wo7_topology_sweep.json"))
args = ap.parse_args()

HOST = platform.node()
I1 = torch.tensor(np.arange(W.N_CUR) * W.I_STEP + W.I_LOW, dtype=torch.float32, device=DEV)
LO = np.array([W.WT[0] / 5000, W.WT[1] / 5000, W.WT[2] / 50])
HI = np.array([W.WT[0] * 5000, W.WT[1] * 5000, W.WT[2] * 50])


def spikes_of(P, shift, chunk, shift_n=0.0):
    """P is (N,3) -> (N,35) spike counts.

    Every call is padded to exactly `chunk` parameter sets so the compiled kernel
    only ever sees one tensor shape.  Without this the ragged final chunk of each
    lattice presents a second shape, and with four variants the recompile limit is
    reached and execution silently falls back to eager."""
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


def _decompose(X):
    """Normalise each axis by its own range, then decompose by SVD.  Returns the
    share of variance on each principal direction (%), the mean distance from the
    best-fit plane as a percentage of the cluster's extent, the absolute loading
    of the first principal direction on each axis, and the name of the axis with
    the largest loading (shows whether the decomposition simply tracks the axis
    with the largest magnitude)."""
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
                plane_residual_pct=float(d.mean() / extent * 100.0),
                pc1_loading=[float(abs(x)) for x in Vt[0]],
                # name of the axis that carries the first direction
                pc1_axis=AXES[int(np.argmax(np.abs(Vt[0])))])


def geometry(P):
    """The decomposition of a point cloud P (N,3), computed in both spaces.

    Linear: the decomposition on the conductances themselves (keys without
    suffix).  The search box spans seven decades and the lattice is log-spaced, so
    in linear space almost every point sits within a tiny fraction of the range
    and the variance is carried by the few largest values.  As the selected set
    grows to span more of the box, this can drive the second direction toward
    zero purely as an effect of the coordinate choice.

    Log: the same procedure on log10 of the conductances (keys suffixed `_log`),
    which is the space the box and the lattice are defined in.

    Returns None if there are fewer than 4 points."""
    X = np.asarray(P, dtype=float)
    if X.shape[0] < 4:
        return None
    out = {}
    for tag, M in (("linear", X), ("log", np.log10(X))):
        for k, v in _decompose(M).items():
            out[f"{k}_{tag}" if tag == "log" else k] = v
    return out


print(f"machine {HOST} | {torch.cuda.get_device_properties(0).name}", flush=True)
print(f"box (Na, K, leak): {LO} to {HI}", flush=True)
print(f"lattice {args.grid_n}^3 = {args.grid_n**3:,} points per variant\n", flush=True)

sc_wt = spikes_of(np.array([W.WT]), 0.0, args.chunk)[0].astype(int)
print(f"wild-type spike counts: total {sc_wt.sum()}\n", flush=True)

RESULT = {"host": HOST, "gpu": torch.cuda.get_device_properties(0).name,
          "mode": "grid", "grid_n": args.grid_n, "wt_spikes": sc_wt.tolist(),
          "method": "each axis normalised by its own range, then SVD",
          "variants": {}}

for vname, spec in VARIANTS.items():
    t0 = time.time()
    base = np.array(W.WT)
    d_untr = int(np.abs(spikes_of(base[None, :], spec["shift"], args.chunk,
                                  spec["shift_n"])[0].astype(int) - sc_wt).sum())

    keptP, keptD, counts = [], [], [0, 0]

    def absorb(P):
        sc = spikes_of(P, spec["shift"], args.chunk, spec["shift_n"]).astype(int)
        counts[0] += P.shape[0]
        alive = sc.sum(axis=1) > 0
        counts[1] += int((~alive).sum())
        if alive.any():
            keptP.append(P[alive])
            keptD.append(np.abs(sc[alive] - sc_wt[None, :]).sum(axis=1))

    g = args.grid_n
    ax_ = [np.geomspace(LO[j], HI[j], g) for j in range(3)]
    A, B_, C = np.meshgrid(ax_[0], ax_[1], ax_[2], indexing="ij")
    allP = np.stack([A.ravel(), B_.ravel(), C.ravel()], axis=1)
    for s0 in range(0, allP.shape[0], args.chunk * 8):
        absorb(allP[s0:s0 + args.chunk * 8])
    del allP, A, B_, C

    n_drawn, n_silent = counts
    P = np.concatenate(keptP); D = np.concatenate(keptD)
    del keptP, keptD
    order = np.argsort(D, kind="stable")
    n_spk = P.shape[0]
    tied = int((D == D.min()).sum())

    # The sweep selects by distance cut, not by count.  The score is an integer
    # spike-count distance, so candidates come in large tie blocks; selecting
    # "the best k" would slice arbitrarily through a tie block and make the
    # geometry jump with k.  Selecting every candidate with distance <= d avoids
    # that, and matches how "tied at the minimum" is defined.
    tied_eff = max(tied, 4)
    dvals = np.unique(D)
    cum = np.searchsorted(np.sort(D), dvals, side="right")   # count at each cut
    keep_mask = cum <= max(int(round(0.10 * n_spk)), tied_eff)
    dvals, cum = dvals[keep_mask], cum[keep_mask]
    if dvals.size > args.n_thresholds:                        # thin, log-spaced in k
        # geomspace cannot start at zero, so thin over RANKS 1..n and shift back
        pick = np.unique(np.round(np.geomspace(
            1, dvals.size, args.n_thresholds)).astype(int)) - 1
        pick = pick[(pick >= 0) & (pick < dvals.size)]
        dvals, cum = dvals[pick], cum[pick]

    sweep = []
    for dcut, k in zip(dvals, cum):
        if k < 4:
            continue
        sel = order[:int(k)]                 # order is by D, so this IS D <= dcut
        gm = geometry(P[sel])
        if gm is None:
            continue
        sweep.append(dict(k=int(k), distance_cut=int(dcut),
                          frac_of_spiking_pct=float(100.0 * k / n_spk),
                          frac_of_lattice_pct=float(100.0 * k / n_drawn),
                          similarity_cut_pct=float(W.similarity(dcut, d_untr))
                          if d_untr else float("nan"),
                          **gm))

    # Point clouds at three representative thresholds (at most `cap` points each).
    def cloud(k, cap=3000):
        sel = order[:int(k)]
        if sel.size > cap:
            sel = sel[np.linspace(0, sel.size - 1, cap).astype(int)]
        return P[sel].tolist()

    # The loosest threshold ("top 10 %") is whatever the last distance cut in the
    # sweep admitted.
    k_top = max(50, int(round(0.001 * n_spk)))
    k_hi = int(sweep[-1]["k"]) if sweep else tied_eff
    clouds = {"tied_at_minimum": cloud(tied_eff),
              "top_0.1pct": cloud(k_top),
              "top_10pct": cloud(k_hi)}

    RESULT["variants"][vname] = dict(
        baseline=base.tolist(), shift=spec["shift"], shift_n=spec["shift_n"],
        d_untreated=d_untr, n_drawn=int(n_drawn), n_spiking=int(n_spk),
        n_silent=int(n_silent), best_distance=int(D.min()),
        n_tied_at_min=tied,
        best_similarity_pct=float(W.similarity(D.min(), d_untr)) if d_untr else None,
        k_tied=tied, k_tied_used=int(tied_eff),
        k_top01=int(k_top), k_top10=int(k_hi),
        sweep=sweep, clouds=clouds, seconds=round(time.time() - t0, 1))

    print(f"--- {vname}", flush=True)
    print(f"    untreated distance {d_untr} | spiking {n_spk:,} of {n_drawn:,} "
          f"({100*n_spk/n_drawn:.1f} %)", flush=True)
    print(f"    best distance {D.min()} "
          f"({RESULT['variants'][vname]['best_similarity_pct']:.2f} %), "
          f"{tied:,} tied there", flush=True)
    for lbl, k in (("tied at minimum", tied_eff), ("top 0.1 %", k_top),
                   ("top 10 %", k_hi)):
        gm = geometry(P[order[:int(k)]])
        if gm is None:
            print(f"    {lbl:<16} k={k:>8,}  too few points for a decomposition",
                  flush=True); continue
        print(f"    {lbl:<16} k={k:>8,}", flush=True)
        print(f"        linear  pc1 {gm['pc1']:6.2f}  pc2 {gm['pc2']:6.2f}  "
              f"pc3 {gm['pc3']:6.2f}   plane {gm['plane_residual_pct']:5.2f} %   "
              f"PC1 on {gm['pc1_axis']:<10} "
              f"|Na,K,leak| {gm['pc1_loading'][0]:.2f},"
              f"{gm['pc1_loading'][1]:.2f},{gm['pc1_loading'][2]:.2f}", flush=True)
        print(f"        log     pc1 {gm['pc1_log']:6.2f}  pc2 {gm['pc2_log']:6.2f}  "
              f"pc3 {gm['pc3_log']:6.2f}   plane {gm['plane_residual_pct_log']:5.2f} %   "
              f"PC1 on {gm['pc1_axis_log']:<10} "
              f"|Na,K,leak| {gm['pc1_loading_log'][0]:.2f},"
              f"{gm['pc1_loading_log'][1]:.2f},{gm['pc1_loading_log'][2]:.2f}",
              flush=True)
    print(f"    {RESULT['variants'][vname]['seconds']:.0f} s\n", flush=True)
    del P, D, order

with open(args.out, "w") as f:
    json.dump(RESULT, f)
print(f"wrote {args.out}", flush=True)
