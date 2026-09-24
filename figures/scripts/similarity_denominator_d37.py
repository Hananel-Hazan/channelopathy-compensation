"""The R859C efficacy denominator, computed from the search's own ladders.

The efficacy scale of the R859C search (the paper's Equation 4) divides by the
untreated variant's own spike-count distance to the reference. The search
simulated both ladders in the same NEURON process, with the protocol of
Neuron_Test/R859C/neuron.hoc: 35 injected-current levels, 20-360 pA, and the
solver that file leaves in place (secondorder = 0). This script simulates the
same two ladders with experiments/wo97/wo97_lib.py, counts spikes with the
pipeline's detector, and derives the denominator as the summed absolute
per-level difference in spike count.

The paper's values are a reference total of 497 action potentials, a variant
total of 488, and a denominator of 37. The archived ladders in
Neuron_Test/R859C/, integrated with secondorder = 2, give 498, 490 and 38
instead; wo7_task13_excitability.py measures those.

Requires NEURON and the compiled working directory:
    bash scripts/fetch_modeldb.sh
    python3 experiments/wo97/build_work.py

Writes figures/similarity_denominator_d37.json, which
wo7_task13_excitability.py reads.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo97"))
import wo97_lib as L  # noqa: E402

OUT = os.path.join(REPO, "figures", "similarity_denominator_d37.json")
STORED_REFERENCE = os.path.join(REPO, "experiments", "wo97", "reference_ladder.npy")


def main():
    h = L.start()
    import neuron
    wt, _ = L.exploration_wt(h)
    mt, _ = L.exploration_mt(h)
    wt = [int(x) for x in wt]
    mt = [int(x) for x in mt]
    per_level = [abs(a - b) for a, b in zip(wt, mt)]
    d = int(sum(per_level))

    stored = np.load(STORED_REFERENCE).astype(int).tolist()
    same_as_stored = stored == wt
    print(f"reference ladder: total {sum(wt)}  (matches the stored reference_ladder.npy: {same_as_stored})")
    print(f"variant ladder:   total {sum(mt)}")
    print(f"summed |reference - variant| over {len(wt)} levels: {d}")

    result = {
        "denominator": d,
        "definition": "summed |reference - untreated variant| spike count over the 35 "
                      "injected-current levels, both ladders simulated in one NEURON "
                      "process with the protocol of Neuron_Test/R859C/neuron.hoc "
                      "(secondorder = 0), spikes counted with the pipeline's detector",
        "currents_pA": list(L.CURRENTS),
        "reference_ladder": wt,
        "variant_ladder": mt,
        "abs_difference_per_level": per_level,
        "reference_total_spikes": sum(wt),
        "variant_total_spikes": sum(mt),
        "reference_matches_stored_reference_ladder": same_as_stored,
        "neuron_version": neuron.__version__,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
