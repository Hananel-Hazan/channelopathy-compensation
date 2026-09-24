"""Merge the shards of the Kv7.2 conductance search and summarise the stored results.

Three jobs, all of them plain arithmetic on the stored result files, no simulation:

  merge   -- join the shards of the search (wo6b_taskD2_search.py, run on separate
             machines) into one file, checking they agree on the references before
             combining anything
  taskD   -- what the search found: best candidate, how the top set is shaped, and how
             many draws were needed to reach each quality level
  taskE   -- the stability window (wo6b_taskE_stability.py): efficacy by perturbation
             size in three modes (one conductance at a time, joint random, worst case),
             and the smallest perturbation at which efficacy first falls to 95 %, 90 %
             and 80 %; writes wo6b_taskE_summary.json

Usage:  python wo6b_analyse.py {merge,taskD,taskE} [--inputs ...] [--stability ...]
"""
import argparse, glob, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["merge", "taskD", "taskE"])
ap.add_argument("--inputs", nargs="*", default=["wo6b_taskD2_search_*.json"])
ap.add_argument("--search", default="wo6b_taskD2_search.json")
ap.add_argument("--stability", nargs="*", default=["wo6b_taskE_stability*.json"])
ap.add_argument("--out", default="wo6b_taskD2_search.json")
args = ap.parse_args()


def merge():
    files = sorted(f for pat in args.inputs for f in glob.glob(pat)
                   if not f.endswith(args.out))
    parts = [json.load(open(f)) for f in files]
    assert parts, "no input files matched"
    ref = parts[0]
    for p in parts[1:]:
        # the references are measured independently on each machine; they must agree,
        # and if they do not, nothing downstream is comparable
        assert p["wild_type_counts"] == ref["wild_type_counts"], "wild-type curves differ"
        assert p["mutant_counts"] == ref["mutant_counts"], "mutant curves differ"
        assert p["currents_nA"] == ref["currents_nA"], "current ladders differ"
    cands = sorted((c for p in parts for c in p["candidates"]), key=lambda c: c["i"])
    idx = [c["i"] for c in cands]
    assert len(set(idx)) == len(idx), "the two shares overlap"
    OUT = dict(ref)
    OUT["hosts"] = [p["host"] for p in parts]
    OUT["seconds_per_host"] = {p["host"]: p["seconds"] for p in parts}
    OUT["candidates"] = cands
    OUT["n_evaluated"] = len(cands)
    OUT["n_rejected"] = sum(1 for c in cands if c["rejected"])
    OUT["index_range_covered"] = [min(idx), max(idx)]
    json.dump(OUT, open(args.out, "w"), indent=1)
    print(f"merged {len(files)} files from {OUT['hosts']}: {len(cands)} candidates, "
          f"indices {min(idx)}-{max(idx)}, {OUT['n_rejected']} rejected")
    print(f"references agree on both machines: wild type {OUT['wild_type_counts']}, "
          f"untreated distance {OUT['d_untreated']}")
    print(f"-> {args.out}")


def taskD():
    S = json.load(open(args.search))
    wt = np.array(S["wild_type_counts"])
    ok = [c for c in S["candidates"] if not c["rejected"]]
    sims = np.array([c["similarity"] for c in ok])
    defaults = {"gna": 0.045, "gkdr": 0.02, "g_pas": 1.0 / (28000.0 / 0.75)}
    print(f"{len(S['candidates'])} candidates, {S['n_rejected']} rejected "
          f"({100*S['n_rejected']/len(S['candidates']):.1f} %), {len(ok)} scored")
    print(f"untreated distance {S['d_untreated']} spikes; one action potential is worth "
          f"{100/S['d_untreated']:.1f} percentage points\n")
    print(f"best {sims.max():.2f} %   median {np.median(sims):.2f} %   "
          f"worst {sims.min():.2f} %")
    for thr in (50, 60, 70, 80, 90):
        n = int((sims >= thr).sum())
        print(f"  at or above {thr:>3} %: {n:>5} of {len(ok)} scored "
              f"({100*n/len(ok):.3f} % of the box) "
              + (f"-> about {np.log(2)/-np.log1p(-n/len(ok)):,.0f} draws to the first hit"
                 if 0 < n < len(ok) else ""))
    top = sorted(ok, key=lambda c: -c["similarity"])[:10]
    print(f"\ntop 10, as multiples of the published values:")
    print(f"  {'similarity':>10} {'sodium':>9} {'potassium':>10} {'leak':>9}   counts")
    for c in top:
        print(f"  {c['similarity']:>9.2f} % {c['cond']['gna']/defaults['gna']:>9.3f} "
              f"{c['cond']['gkdr']/defaults['gkdr']:>10.3f} "
              f"{c['cond']['g_pas']/defaults['g_pas']:>9.3f}   {c['counts']}")
    print(f"  {'wild type':>10}  {'1.000':>9} {'1.000':>10} {'1.000':>9}   {list(wt)}")
    r = np.array([[c["cond"]["gna"]/defaults["gna"], c["cond"]["gkdr"]/defaults["gkdr"],
                   c["cond"]["g_pas"]/defaults["g_pas"]] for c in top])
    print(f"\ntop-10 geometric mean prescription: sodium x{np.exp(np.log(r[:,0]).mean()):.3f}, "
          f"potassium x{np.exp(np.log(r[:,1]).mean()):.3f}, "
          f"leak x{np.exp(np.log(r[:,2]).mean()):.3f}")


def taskE():
    files = sorted(f for pat in args.stability for f in glob.glob(pat))
    parts = [json.load(open(f)) for f in files]
    assert parts, "no stability files matched"
    res = [r for p in parts for r in p["results"]]
    sim_opt = parts[0]["similarity_at_optimum"]
    d_untr = parts[0]["d_untreated"]
    print(f"optimum similarity {sim_opt:.2f} %; one action potential is worth "
          f"{100/d_untr:.1f} points of similarity, i.e. "
          f"{100*(100/d_untr)/sim_opt:.1f} points of efficacy\n")
    deltas = sorted({r["delta"] for r in res})
    rows = {}
    print(f"{'perturbation':>13} {'one at a time':>15} {'joint median':>14} "
          f"{'joint 5th pct':>14} {'worst case':>12} {'rejected':>9}")
    for d in deltas:
        sub = [r for r in res if r["delta"] == d]
        def eff(mode):
            v = [r["efficacy_pct"] for r in sub
                 if r["mode"] == mode and r["efficacy_pct"] is not None]
            return np.array(v) if v else np.array([np.nan])
        one, joint, worst = eff("one_at_a_time"), eff("joint"), eff("worst_case")
        rej = sum(1 for r in sub if r["efficacy_pct"] is None)
        rows[d] = {"one_at_a_time": float(np.nanmin(one)),
                   "joint_median": float(np.nanmedian(joint)),
                   "joint_p5": float(np.nanpercentile(joint, 5)),
                   "worst_case": float(np.nanmin(np.concatenate([worst, one]))),
                   "rejected": rej}
        print(f"{d*100:>12.0f} % {rows[d]['one_at_a_time']:>14.0f} "
              f"{rows[d]['joint_median']:>13.0f} {rows[d]['joint_p5']:>13.0f} "
              f"{rows[d]['worst_case']:>11.0f} {rej:>9}")
    print(f"\nsmallest perturbation at which efficacy first falls to or below:")
    print(f"  {'mode':<22} {'95 %':>7} {'90 %':>7} {'80 %':>7}")
    for mode in ("one_at_a_time", "joint_median", "joint_p5", "worst_case"):
        cells = []
        for thr in (95, 90, 80):
            hit = [d for d in deltas if rows[d][mode] <= thr]
            cells.append(f"{min(hit)*100:.0f} %" if hit else "not reached")
        print(f"  {mode:<22} {cells[0]:>7} {cells[1]:>7} {cells[2]:>7}")
    json.dump({"rows": {str(k): v for k, v in rows.items()},
               "similarity_at_optimum": sim_opt, "files": files},
              open("wo6b_taskE_summary.json", "w"), indent=1)
    print("\n-> wo6b_taskE_summary.json")


{"merge": merge, "taskD": taskD, "taskE": taskE}[args.what]()
