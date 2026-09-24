"""Dimensionality null model for the compensating-set geometry.

The compensating set is a threshold on a scalar over a 3-D parameter space.  The
level set of a scalar function on a 3-D space is generically a 2-D surface, so a
thin shell around it is generically ~2-D and its third principal direction is
generically small.  This script measures the third principal direction of the real
criterion (summed absolute spike-count difference from wild type) and of criteria with
no compensation structure, on the same lattice and at the same selection size.

Method:
  * the same 200^3 geomspaced lattice over the same box as
    figures/scripts/wo7_task23_topology_sweep.py;
  * the same "spiking candidates only" support -- the null is scored on exactly the
    points the real criterion keeps, so support is not what is being compared;
  * two null criteria, both smooth scalars with no compensation structure:
      NULL-PLANE  s(u) = mean of the three normalised log-conductances
                  (planar level sets; the strict dimension-counting bound)
      NULL-CURVED an arbitrary smooth scalar with curved level sets
                  (the generic case)
  * selection matched on COUNT to each row of
    figures/solution_topology_vs_threshold.csv, so both criteria are decomposed on
    sets of identical size;
  * `_decompose` is the function from wo7_task23_topology_sweep.py, unchanged.

The real criterion is recomputed here as well, and its agreement with the published CSV
is written to the output.

Writes figures/wo9_dimensionality_null.json.  Needs a CUDA GPU.
"""
import argparse, csv, json, os, platform, sys, time
import numpy as np
import torch

for _a in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _a):
        setattr(torch._dynamo.config, _a, 64)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
import wo5_lib as W

OUT = os.path.join(REPO, "figures")
DEV = "cuda"
AXES = ("sodium", "potassium", "leak")

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
ap.add_argument("--variants", default="all")
args = ap.parse_args()

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


def _decompose(X):
    """PCA of the range-normalised point cloud (from wo7_task23_topology_sweep.py, unchanged).

    Returns the percentage of variance on each principal component, the mean distance to
    the best-fit plane as a percentage of the cloud's extent, and the axis that dominates PC1."""
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
                plane_residual_pct=float(d.mean() / extent * 100.0),
                pc1_axis=AXES[int(np.argmax(np.abs(Vt[0])))])


def geometry(P):
    X = np.asarray(P, dtype=float)
    if X.shape[0] < 4:
        return None
    out = {}
    for tag, M in (("linear", X), ("log", np.log10(X))):
        for k, v in _decompose(M).items():
            out[f"{k}_{tag}" if tag == "log" else k] = v
    return out


def normalised_log(P):
    """Map the box to the unit cube in log10 coordinates."""
    U = (np.log10(P) - np.log10(LO)) / (np.log10(HI) - np.log10(LO))
    return U


def null_plane(P):
    U = normalised_log(P)
    u0 = normalised_log(np.array([W.WT]))[0]
    return np.abs(U.mean(axis=1) - u0.mean())


def null_curved(P):
    """An arbitrary smooth scalar with curved level sets and no compensation
    structure.  Coefficients are fixed constants chosen once and not tuned."""
    U = normalised_log(P)
    u0 = normalised_log(np.array([W.WT]))[0]

    def f(u):
        return (u[..., 0] + 0.70 * u[..., 1] - 0.40 * u[..., 2]
                + 0.90 * u[..., 0] * u[..., 1] - 0.60 * u[..., 1] ** 2
                + 0.50 * np.sin(3.0 * u[..., 2]))
    return np.abs(f(U) - f(u0))


def main():
    published = {}
    with open(os.path.join(OUT, "solution_topology_vs_threshold.csv")) as f:
        for r in csv.DictReader(f):
            published.setdefault(r["variant"], []).append(r)

    names = list(VARIANTS) if args.variants == "all" else \
        [n for n in VARIANTS if args.variants.lower() in n.lower()]

    sc_wt = spikes_of(np.array([W.WT]), 0.0, args.chunk)[0].astype(int)
    result = {"host": platform.node(),
              "gpu": torch.cuda.get_device_properties(0).name,
              "grid_n": args.grid_n, "wt_total_spikes": int(sc_wt.sum()),
              "box_lo": LO.tolist(), "box_hi": HI.tolist(), "variants": {}}

    g = args.grid_n
    ax_ = [np.geomspace(LO[j], HI[j], g) for j in range(3)]
    A, B_, C = np.meshgrid(ax_[0], ax_[1], ax_[2], indexing="ij")
    allP = np.stack([A.ravel(), B_.ravel(), C.ravel()], axis=1)
    del A, B_, C

    for vname in names:
        spec = VARIANTS[vname]
        t0 = time.time()
        keptP, keptD = [], []
        step = args.chunk * 8
        for s0 in range(0, allP.shape[0], step):
            Pb = allP[s0:s0 + step]
            sc = spikes_of(Pb, spec["shift"], args.chunk, spec["shift_n"]).astype(int)
            alive = sc.sum(axis=1) > 0
            if alive.any():
                keptP.append(Pb[alive])
                keptD.append(np.abs(sc[alive] - sc_wt[None, :]).sum(axis=1))
        P = np.concatenate(keptP); D = np.concatenate(keptD)
        del keptP, keptD
        Dn_plane = null_plane(P)
        Dn_curved = null_curved(P)
        order_real = np.argsort(D, kind="stable")
        order_plane = np.argsort(Dn_plane, kind="stable")
        order_curved = np.argsort(Dn_curved, kind="stable")

        rows = []
        for r in published.get(vname, []):
            k = int(r["candidates_kept"])
            if k < 4 or k > P.shape[0]:
                continue
            rec = {"candidates_kept": k,
                   "published_log_pc3_pct": float(r["log_pc3_pct"]),
                   "published_linear_pc3_pct": float(r["linear_pc3_pct"])}
            for tag, order in (("real", order_real), ("null_plane", order_plane),
                               ("null_curved", order_curved)):
                gm = geometry(P[order[:k]])
                rec[f"{tag}_log_pc3_pct"] = gm["pc3_log"]
                rec[f"{tag}_linear_pc3_pct"] = gm["pc3"]
                rec[f"{tag}_log_pc1_axis"] = gm["pc1_axis_log"]
            rows.append(rec)

        agree = [abs(x["real_log_pc3_pct"] - x["published_log_pc3_pct"]) for x in rows]
        result["variants"][vname] = {
            "n_spiking": int(P.shape[0]),
            "n_lattice": int(allP.shape[0]),
            "seconds": round(time.time() - t0, 1),
            "reproduction_max_abs_diff_log_pc3_vs_published": max(agree) if agree else None,
            "rows": rows,
            "summary": {
                "max_real_log_pc3": max(x["real_log_pc3_pct"] for x in rows),
                "max_null_plane_log_pc3": max(x["null_plane_log_pc3_pct"] for x in rows),
                "max_null_curved_log_pc3": max(x["null_curved_log_pc3_pct"] for x in rows),
                "median_real_log_pc3": float(np.median([x["real_log_pc3_pct"] for x in rows])),
                "median_null_plane_log_pc3": float(np.median([x["null_plane_log_pc3_pct"] for x in rows])),
                "median_null_curved_log_pc3": float(np.median([x["null_curved_log_pc3_pct"] for x in rows])),
            },
        }
        print(f"[{vname}] {result['variants'][vname]['summary']}", flush=True)

    path = os.path.join(OUT, "wo9_dimensionality_null.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
