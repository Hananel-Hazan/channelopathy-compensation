# Data that is not in this repository

The data behind every figure and Table 1 is in this repository, except two inputs that are too
large for it. Redrawing Figure 7 from the search output, rather than from the plotted
distribution shipped here, needs them. Section 4 of [REPRODUCE.md](REPRODUCE.md) lists the few
numbers whose raw output is not archived.

## The archived R859C candidate databases

| File | Size |
|---|---|
| `sumary_explor_result100K.mutationFix.pbz2` | 62.7 MB |
| `sumary_explor_result.100K.pbz2` | 43.5 MB |

These are bz2-compressed pickles from the original R859C search. They hold the 1,050
mutant-to-wild-type candidates, the 37 mutant-to-mutant configurations, and the per-test scores
of the 38,850 interventions built from them.

Put both files in a directory named `external_data/` at the root of the repository. The scripts
look for them there, and `.gitignore` keeps that directory out of version control.

| Script (`figures/scripts/`) | Produces |
|---|---|
| `wo98_task14_efficacy_d37.py` | Figure 7 and `figures/mutation1_efficacy_summary.json` |
| `wo7_dtw_denominator.py` | the dynamic-time-warping similarity denominator |
| `wo9_task33_dtw_decomposition.py` | the breakdown of the dynamic-time-warping tests |
| `wo9_task33_reconstruct.py` | the construction of the interventions from the candidates |

The data plotted in Figure 7 is already in `figures/mutation1_efficacy_distribution.csv`. The
databases are needed only to recompute that distribution from the search output.

Both databases are deposited on Zenodo with this repository, doi:10.5281/zenodo.22942388, under the
Creative Commons Attribution 4.0 licence, together with the full run logs of the original
search.
