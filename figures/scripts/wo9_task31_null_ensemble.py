"""Ensemble of random smooth null criteria for the dimensionality test.

`wo9_task31_dimensionality_null.py` uses two null criteria: a planar one and one
arbitrary curved one, whose third-direction share depends on how curved it is.  This
script draws an ENSEMBLE of random smooth criteria on the same lattice and the same
spiking support, and reports the spread of the third principal direction across them,
next to the real criterion, at each selection size in
figures/solution_topology_vs_threshold.csv.

Each ensemble member is a random quadratic-plus-sinusoid scalar on the normalised
log-conductance cube, with coefficients drawn from fixed distributions and a fixed
seed sequence, so the ensemble is reproducible.

One variant only (the R859C sodium-activation shift); wo9_task31_null_ensemble_allvariants.py
runs all four.  Command-line options are those of wo9_task31_dimensionality_null.py, which
parses them on import.

Writes figures/wo9_dimensionality_null_ensemble.json.  Needs a CUDA GPU.
"""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wo9_task31_dimensionality_null as N

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")
VARIANT = "sodium activation +6.1 mV (R859C, ModelDB 87585)"
N_MEMBERS = 24
SEED0 = 20260904


def random_smooth(rng):
    """A random smooth scalar on the unit cube: linear + quadratic + a sinusoid per axis."""
    a = rng.normal(0, 1, 3)
    Q = rng.normal(0, 0.6, (3, 3)); Q = (Q + Q.T) / 2.0
    b = rng.normal(0, 0.5, 3)
    w = rng.uniform(1.0, 4.0, 3)
    ph = rng.uniform(0, 2 * np.pi, 3)

    def f(u):
        lin = u @ a
        quad = np.einsum("ni,ij,nj->n", u, Q, u)
        sin = (b * np.sin(w * u + ph)).sum(axis=1)
        return lin + quad + sin
    return f


def main():
    published = {}
    with open(os.path.join(OUT, "solution_topology_vs_threshold.csv")) as f:
        for r in csv.DictReader(f):
            published.setdefault(r["variant"], []).append(r)
    counts = sorted({int(r["candidates_kept"]) for r in published[VARIANT]
                     if int(r["candidates_kept"]) >= 4})

    spec = N.VARIANTS[VARIANT]
    sc_wt = N.spikes_of(np.array([N.W.WT]), 0.0, N.args.chunk)[0].astype(int)
    g = N.args.grid_n
    ax_ = [np.geomspace(N.LO[j], N.HI[j], g) for j in range(3)]
    A, B_, C = np.meshgrid(ax_[0], ax_[1], ax_[2], indexing="ij")
    allP = np.stack([A.ravel(), B_.ravel(), C.ravel()], axis=1)
    del A, B_, C

    keptP, keptD = [], []
    step = N.args.chunk * 8
    for s0 in range(0, allP.shape[0], step):
        Pb = allP[s0:s0 + step]
        sc = N.spikes_of(Pb, spec["shift"], N.args.chunk, spec["shift_n"]).astype(int)
        alive = sc.sum(axis=1) > 0
        if alive.any():
            keptP.append(Pb[alive]); keptD.append(np.abs(sc[alive] - sc_wt[None, :]).sum(axis=1))
    P = np.concatenate(keptP); D = np.concatenate(keptD)
    del keptP, keptD, allP
    U = N.normalised_log(P)
    u0 = N.normalised_log(np.array([N.W.WT]))

    real_order = np.argsort(D, kind="stable")
    real = {k: N.geometry(P[real_order[:k]])["pc3_log"] for k in counts}

    members = []
    for i in range(N_MEMBERS):
        rng = np.random.default_rng(SEED0 + i)
        f = random_smooth(rng)
        d = np.abs(f(U) - f(u0)[0])
        order = np.argsort(d, kind="stable")
        pc3 = {k: N.geometry(P[order[:k]])["pc3_log"] for k in counts}
        members.append(pc3)
        print(f"  member {i:2d}: median pc3 {np.median(list(pc3.values())):6.3f} %  "
              f"max {max(pc3.values()):6.3f} %", flush=True)

    rows = []
    for k in counts:
        vals = np.array([m[k] for m in members])
        rows.append(dict(candidates_kept=k, real_log_pc3_pct=real[k],
                         null_ensemble_min=float(vals.min()),
                         null_ensemble_p25=float(np.percentile(vals, 25)),
                         null_ensemble_median=float(np.median(vals)),
                         null_ensemble_p75=float(np.percentile(vals, 75)),
                         null_ensemble_max=float(vals.max()),
                         real_below_ensemble_median=bool(real[k] < np.median(vals)),
                         real_rank_in_ensemble=int((vals < real[k]).sum())))
    allreal = np.array([r["real_log_pc3_pct"] for r in rows])
    allmed = np.array([r["null_ensemble_median"] for r in rows])
    rep = dict(variant=VARIANT, n_members=N_MEMBERS, seed0=SEED0,
               n_spiking=int(P.shape[0]), selection_sizes=counts,
               rows=rows,
               summary=dict(
                   real_median_over_thresholds=float(np.median(allreal)),
                   real_max_over_thresholds=float(allreal.max()),
                   ensemble_median_of_medians=float(np.median(allmed)),
                   ensemble_max_over_all=float(max(r["null_ensemble_max"] for r in rows)),
                   ensemble_min_over_all=float(min(r["null_ensemble_min"] for r in rows)),
                   thresholds_where_real_below_ensemble_median=int(sum(
                       r["real_below_ensemble_median"] for r in rows)),
                   n_thresholds=len(rows)))
    path = os.path.join(OUT, "wo9_dimensionality_null_ensemble.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print("\n", json.dumps(rep["summary"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
