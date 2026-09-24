"""The 24-member smooth-criterion null ensemble across all four variant types.

`wo9_task31_null_ensemble.py` runs the ensemble on one variant (the R859C sodium-activation
shift).  This script runs it on every variant in wo9_task31_dimensionality_null.VARIANTS.
Member seeds, member count, criterion family, lattice, spiking support, selection sizes and
decomposition are all the same as in the single-variant script, so the R859C variant
reproduces `figures/wo9_dimensionality_null_ensemble.json` exactly.

Implementation notes:

  1. TRITON FALLBACK.  wo5_lib's fast integrator is torch.compile'd, which needs a
     working triton.  If triton is missing the script falls back to the uncompiled
     path (wo5_lib.simulate_stats_fast takes fused=False) and prints a warning.
     Results are identical; throughput is the unfused rate, so expect roughly an hour
     per variant on an RTX 4070 Ti and about four on an RTX 2070.  Installing triton
     is much faster -- see the banner the script prints.  Override the automatic
     choice with HH_FUSED=1 or HH_FUSED=0.

  2. SUPPORT CACHE.  The expensive step is building the spiking support by
     simulating the 8,000,000-point lattice; the 24 criteria on top of it are
     seconds of array arithmetic.  The support is cached per variant to
     figures/wo9_null_support_<tag>.npz.  Delete the .npz files to force a
     re-simulation.

Output: figures/wo9_dimensionality_null_ensemble_allvariants.json (the single-variant
output file is not overwritten).  Needs a CUDA GPU.

Usage:
    python3 figures/scripts/wo9_task31_null_ensemble_allvariants.py
"""
import csv, functools, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wo9_task31_dimensionality_null as N

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")
N_MEMBERS = 24
SEED0 = 20260904


def _tag(vname):
    return re.sub(r"[^a-z0-9]+", "_", vname.lower()).strip("_")[:48]


def _choose_fused():
    """Return True if the compiled path works, False to run eager.  HH_FUSED overrides."""
    env = os.environ.get("HH_FUSED")
    if env is not None:
        want = env.strip() not in ("0", "false", "no", "")
        print(f"  HH_FUSED={env!r} -> fused={want}", flush=True)
        return want
    try:
        N.spikes_of(np.array([N.W.WT]), 0.0, N.args.chunk)
        print("  compiled (fused) integrator OK", flush=True)
        return True
    except Exception as e:
        name = type(e).__name__
        print("\n" + "=" * 78)
        print("  The compiled integrator is unavailable:", name)
        print("  ", str(e).strip().splitlines()[0][:160])
        print()
        print("  Falling back to the UNCOMPILED path.  Results are identical; the run is")
        print("  much slower (the unfused rate, roughly 2,044 sets/s on the 4070 Ti and")
        print("  543/s on the 2070, against 63,205 and 34,350 fused).  Budget about an")
        print("  hour per variant on the 4070 Ti, four on the 2070.")
        print()
        print("  To get the fast path back, install a triton matching this torch:")
        print("      python3 -c 'import torch; print(torch.__version__)'")
        print("      pip install triton")
        print("  then re-run.  The support cache means a second run is cheap.")
        print("=" * 78 + "\n", flush=True)
        return False


def random_smooth(rng):
    """Same as wo9_task31_null_ensemble.random_smooth -- same draws, same seeds."""
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


def support_for(vname, spec, sc_wt, allP):
    """The spiking support and its spike-count distances, cached to disk."""
    path = os.path.join(OUT, f"wo9_null_support_{_tag(vname)}.npz")
    if os.path.exists(path):
        z = np.load(path)
        print(f"  support loaded from {os.path.basename(path)} "
              f"({z['P'].shape[0]:,} points)", flush=True)
        return z["P"].astype(float), z["D"].astype(int)
    t0 = time.time()
    keptP, keptD = [], []
    step = N.args.chunk * 8
    nblocks = (allP.shape[0] + step - 1) // step
    for bi, s0 in enumerate(range(0, allP.shape[0], step)):
        Pb = allP[s0:s0 + step]
        sc = N.spikes_of(Pb, spec["shift"], N.args.chunk, spec["shift_n"]).astype(int)
        alive = sc.sum(axis=1) > 0
        if alive.any():
            keptP.append(Pb[alive])
            keptD.append(np.abs(sc[alive] - sc_wt[None, :]).sum(axis=1))
        if bi % 20 == 0 or bi == nblocks - 1:
            el = time.time() - t0
            eta = el / (bi + 1) * (nblocks - bi - 1)
            print(f"    block {bi + 1}/{nblocks}  elapsed {el / 60:6.1f} min  "
                  f"eta {eta / 60:6.1f} min", flush=True)
    P = np.concatenate(keptP); D = np.concatenate(keptD)
    np.savez_compressed(path, P=P.astype(np.float32), D=D.astype(np.int32))
    print(f"  support built in {(time.time() - t0) / 60:.1f} min, cached to "
          f"{os.path.basename(path)} ({P.shape[0]:,} points)", flush=True)
    return P, D


def main():
    published = {}
    with open(os.path.join(OUT, "solution_topology_vs_threshold.csv")) as f:
        for r in csv.DictReader(f):
            published.setdefault(r["variant"], []).append(r)

    fused = _choose_fused()
    if not fused:
        N.W.simulate_stats_fast = functools.partial(N.W.simulate_stats_fast, fused=False)

    sc_wt = N.spikes_of(np.array([N.W.WT]), 0.0, N.args.chunk)[0].astype(int)
    g = N.args.grid_n
    ax_ = [np.geomspace(N.LO[j], N.HI[j], g) for j in range(3)]
    A, B_, C = np.meshgrid(ax_[0], ax_[1], ax_[2], indexing="ij")
    allP = np.stack([A.ravel(), B_.ravel(), C.ravel()], axis=1)
    del A, B_, C

    # the 24 criteria are drawn ONCE and reused across variants, so the ensemble is
    # the same control everywhere and the four variants are directly comparable.
    criteria = [random_smooth(np.random.default_rng(SEED0 + i)) for i in range(N_MEMBERS)]

    report = {"n_members": N_MEMBERS, "seed0": SEED0, "grid_n": g,
              "fused": bool(fused), "n_lattice": int(allP.shape[0]), "variants": {}}

    for vname, spec in N.VARIANTS.items():
        print(f"\n[{vname}]", flush=True)
        t0 = time.time()
        counts = sorted({int(r["candidates_kept"]) for r in published[vname]
                         if int(r["candidates_kept"]) >= 4})
        P, D = support_for(vname, spec, sc_wt, allP)
        U = N.normalised_log(P)
        u0 = N.normalised_log(np.array([N.W.WT]))

        counts = [k for k in counts if k <= P.shape[0]]
        real_order = np.argsort(D, kind="stable")
        real = {k: N.geometry(P[real_order[:k]])["pc3_log"] for k in counts}

        members = []
        for i, f in enumerate(criteria):
            d = np.abs(f(U) - f(u0)[0])
            order = np.argsort(d, kind="stable")
            pc3 = {k: N.geometry(P[order[:k]])["pc3_log"] for k in counts}
            members.append(pc3)
            print(f"    member {i:2d}: median {np.median(list(pc3.values())):7.3f} %"
                  f"  max {max(pc3.values()):7.3f} %", flush=True)

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

        member_medians = [float(np.median(list(m.values()))) for m in members]
        allreal = np.array([r["real_log_pc3_pct"] for r in rows])
        allmed = np.array([r["null_ensemble_median"] for r in rows])
        report["variants"][vname] = dict(
            n_spiking=int(P.shape[0]), seconds=round(time.time() - t0, 1),
            selection_sizes=counts, member_medians=member_medians, rows=rows,
            summary=dict(
                real_median_over_thresholds=float(np.median(allreal)),
                real_max_over_thresholds=float(allreal.max()),
                ensemble_member_median_min=float(min(member_medians)),
                ensemble_member_median_max=float(max(member_medians)),
                ensemble_median_of_medians=float(np.median(allmed)),
                ensemble_max_over_all=float(max(r["null_ensemble_max"] for r in rows)),
                ensemble_min_over_all=float(min(r["null_ensemble_min"] for r in rows)),
                real_inside_member_median_range=bool(
                    min(member_medians) <= float(np.median(allreal)) <= max(member_medians)),
                members_with_median_under_12pct=int(sum(m < 12.0 for m in member_medians)),
                members_under_12pct_at_every_threshold=int(sum(
                    max(m.values()) < 12.0 for m in members)),
                thresholds_where_real_below_ensemble_median=int(sum(
                    r["real_below_ensemble_median"] for r in rows)),
                n_thresholds=len(rows)))
        print(f"  summary: {json.dumps(report['variants'][vname]['summary'])}", flush=True)

    # pooled over all four variants: the range of ensemble member medians, and whether
    # each variant's real value lies inside it
    lo = min(v["summary"]["ensemble_member_median_min"] for v in report["variants"].values())
    hi = max(v["summary"]["ensemble_member_median_max"] for v in report["variants"].values())
    reals = {k: v["summary"]["real_median_over_thresholds"] for k, v in report["variants"].items()}
    report["pooled_over_variants"] = dict(
        ensemble_member_median_range_pct=[lo, hi],
        real_median_by_variant=reals,
        all_reals_inside_range=bool(all(lo <= r <= hi for r in reals.values())),
        note="Range of the ensemble members' medians, pooled over the four variant "
             "types, and whether each variant's own compensating-set median lies "
             "inside that range.")

    path = os.path.join(OUT, "wo9_dimensionality_null_ensemble_allvariants.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + json.dumps(report["pooled_over_variants"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
