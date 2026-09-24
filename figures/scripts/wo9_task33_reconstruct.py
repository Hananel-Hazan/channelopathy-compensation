"""Reconstruct the intervention construction from the source candidate database, and
show where the negative-conductance rejection acts.

The construction is transcribed from the original R859C search pipeline, with
num_of_candidates = 70:

  MT-WT set   spike count: rows range(70*15) = range(1,050)
              DTW        : rows range(70)
  MT-MT set   spike count: all rows tied at the minimum distance
              DTW        : rows range(70)
  difference vectors  delta[i,j] = MT_WT[i] - WT_WT[j]  over the SAME index set
                      used for MT-MT  ->  |MT_WT| x |MT_MT| vectors
  tests       for each MT-MT configuration g:  g + delta  AND  g - delta,
              each filtered by  np.sum(t_param < 0, axis=1) == 0

Reads external_data/sumary_explor_result.100K.pbz2 (candidate database) and
external_data/sumary_explor_result100K.mutationFix.pbz2 (scored tests), compares the
reconstructed test counts with the stored ones, and writes
figures/wo9_intervention_construction.json.
"""
import bz2, json, os, pickle
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DB = os.path.join(REPO, "external_data", "sumary_explor_result.100K.pbz2")
SCORED = os.path.join(REPO, "external_data", "sumary_explor_result100K.mutationFix.pbz2")
OUT = os.path.join(REPO, "figures")
NCAND = 70
CRIT = ["Spike Count", "Spike Width avg", "Spike Width std", "Dynamic Time Warping"]


def select(db, k1):
    mtwt = np.where(db[k1]["MT"]["WT"][:, 3] == db[k1]["MT"]["WT"][:, 3].min())[0]
    if k1 == CRIT[0]:
        if NCAND > mtwt.shape[0]:
            mtwt = np.arange(NCAND)
        elif NCAND < mtwt.shape[0]:
            mtwt = np.arange(NCAND * 15)
    elif k1 == CRIT[3]:
        if NCAND != mtwt.shape[0]:
            mtwt = np.arange(NCAND)
    mtmt = np.where(db[k1]["MT"]["MT"][:, 3] == db[k1]["MT"]["MT"][:, 3].min())[0]
    if k1 == CRIT[3] and NCAND != mtmt.shape[0]:
        mtmt = np.arange(NCAND)
    return mtwt, mtmt


def main():
    with bz2.BZ2File(DB, "r") as f:
        db = pickle.load(f)
    with bz2.BZ2File(SCORED, "r") as f:
        scored = pickle.load(f)

    rep = {}
    for k1 in (CRIT[0], CRIT[3]):
        mtwt, mtmt = select(db, k1)
        delta = np.concatenate(
            [db[k1]["MT"]["WT"][i, 0:3] - db[k1]["WT"]["WT"][mtmt, 0:3] for i in mtwt],
            axis=0)
        plus = minus = 0
        per_group = []
        all_pos = bool(np.all(delta > 0))
        for g in range(mtmt.shape[0]):
            base = db[k1]["MT"]["MT"][mtmt[g], 0:3]
            p = int(np.sum(np.sum(base + delta < 0, axis=1) == 0))
            m = int(np.sum(np.sum(base - delta < 0, axis=1) == 0))
            plus += p; minus += m; per_group.append(p + m)
        obs = {int(i): int(np.asarray(scored[k1]["score"]["candidates"][i]).shape[0])
               for i in sorted(scored[k1]["score"]["candidates"])}
        obs_list = [obs[i] for i in sorted(obs)]
        rep[k1] = dict(
            n_MT_WT=int(mtwt.shape[0]), n_MT_MT=int(mtmt.shape[0]),
            n_difference_vectors=int(delta.shape[0]),
            difference_vectors_all_strictly_positive=all_pos,
            n_difference_vectors_positive_in_all_three=int(
                np.sum(np.sum(delta <= 0, axis=1) == 0)),
            max_tests_if_nothing_rejected=int(2 * delta.shape[0] * mtmt.shape[0]),
            admissible_plus_delta=plus, admissible_minus_delta=minus,
            reconstructed_total=int(plus + minus),
            observed_total=int(sum(obs_list)),
            reconstruction_matches_observed=bool(plus + minus == sum(obs_list)),
            per_group_matches_observed=bool(per_group == obs_list),
            rejected_fraction_pct=round(
                100.0 * (1 - (plus + minus) / (2 * delta.shape[0] * mtmt.shape[0])), 4),
            observed_per_group_min=int(min(obs_list)),
            observed_per_group_max=int(max(obs_list)),
        )
    path = os.path.join(OUT, "wo9_intervention_construction.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
