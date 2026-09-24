"""How the 560,230 Dynamic Time Warping tests decompose.

The spike-count arm decomposes cleanly (1,050 MT-WT x 37 MT-MT = 38,850
interventions, each applied to each of 37 configurations = 1,437,450 tests).
The DTW arm does not: 70 x 70 x 70 = 343,000, and 560,230 / 4,900 = 114.33.
This reads the stored score array and reports, for each criterion, the per-group row
counts and the sizes of the MT-WT and MT-MT candidate lists.

Reads external_data/sumary_explor_result100K.mutationFix.pbz2 and writes
figures/wo9_dtw_decomposition.json.  wo9_task33_reconstruct.py reproduces the row counts
from the candidate database.
"""
import bz2, json, os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCORES = os.path.join(REPO, "external_data", "sumary_explor_result100K.mutationFix.pbz2")
OUT = os.path.join(REPO, "figures")


def main():
    with bz2.BZ2File(SCORES, "r") as f:
        sr = pickle.load(f)

    report = {"criteria": {}}
    for k1 in sr:
        node = sr[k1]
        cand = node["score"]["candidates"]
        keys = sorted(cand)
        counts = [int(np.asarray(cand[i]).shape[0]) for i in keys]
        n_mtwt = len(node["Number of MT-WT candidates"])
        n_mtmt = len(node["Number of MT-MT candidates"])
        uniq, cnt = np.unique(counts, return_counts=True)
        report["criteria"][k1] = {
            "top_level_keys": sorted([str(x) for x in node.keys()]),
            "n_groups": len(keys),
            "group_keys_first10": [str(x) for x in keys[:10]],
            "total_rows": int(sum(counts)),
            "rows_per_group_unique": {int(u): int(c) for u, c in zip(uniq, cnt)},
            "rows_per_group_min": int(min(counts)),
            "rows_per_group_max": int(max(counts)),
            "MT_WT_list_len": n_mtwt,
            "MT_MT_list_len": n_mtmt,
            "product_MTWT_x_MTMT": n_mtwt * n_mtmt,
            "product_x_groups": n_mtwt * n_mtmt * len(keys),
            "MT_WT_counts_head": [int(x) for x in
                                  np.asarray(node["Number of MT-WT candidates"])[:10]]
            if np.ndim(node["Number of MT-WT candidates"]) == 1 else None,
            "MT_MT_counts_head": [int(x) for x in
                                  np.asarray(node["Number of MT-MT candidates"])[:10]]
            if np.ndim(node["Number of MT-MT candidates"]) == 1 else None,
        }
        # is total_rows = sum over groups of (MT-WT_g x MT-MT_g)?
        try:
            a = np.asarray(node["Number of MT-WT candidates"], dtype=float)
            b = np.asarray(node["Number of MT-MT candidates"], dtype=float)
            report["criteria"][k1]["sum_a_times_b_elementwise"] = float((a * b).sum())
            report["criteria"][k1]["sum_MT_WT"] = float(a.sum())
            report["criteria"][k1]["sum_MT_MT"] = float(b.sum())
        except Exception as e:
            report["criteria"][k1]["elementwise_note"] = str(e)

    path = os.path.join(OUT, "wo9_dtw_decomposition.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:6000])
    print("\nwrote", path)


if __name__ == "__main__":
    main()
