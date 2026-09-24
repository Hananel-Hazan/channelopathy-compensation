# Computational framework for identifying ion channel mutation-compensating interventions

Code and data for the paper by **Hananel Hazan and Michael Levin**, Allen Discovery Center at
Tufts University.

A channelopathy is modelled as a fixed change to the kinetics or conductance of one ion channel,
and a candidate intervention as a multiplier on the maximal conductance of channels that are *not*
mutated. The framework searches the space of those multipliers for configurations that restore the
reference cell's firing, scoring each candidate by spike count across a 35-level injected-current
ladder and by dynamic time warping of the voltage traces. Two kinds of forward model are used:
NEURON models, which carry the two case studies (Nav1.1 R859C and Kv7.2 D212G), and a
differentiable Hodgkin-Huxley model in PyTorch, which is fast enough for large surveys of
conductance space.

## Quick start

Regenerating the figures from the data in this repository needs only NumPy and Matplotlib:

```bash
git clone https://github.com/Hananel-Hazan/channelopathy-compensation channelopathy-compensation && cd channelopathy-compensation
python3 -m pip install -r requirements.txt
cd figures/scripts
python3 wo7_task13_excitability.py        # Figure 13; writes figures/r859c_excitability_ladder.{pdf,png,csv}
```

Every figure script runs the same way. Scripts resolve all paths relative to the repository, so
they can be run from any directory. A figure script will not overwrite an existing PDF or PNG;
delete the old one first. [docs/REPRODUCE.md](docs/REPRODUCE.md) covers the heavier steps
(graphics-card sweeps and NEURON simulations).

## Repository layout

```
figures/                   data files the figures and tables are drawn from (CSV/JSON)
figures/scripts/           figure and table scripts, and the analyses they rely on
experiments/               the simulation experiments that produced the data
  wo2_*.py                 differentiable model: optimizers, loss functions, learning rates
  wo3/                     differentiable model on the GPU: fused and graph-captured integration, throughput logs
  wo5/                     differentiable-model library (wo5_lib.py, wo5_fit.py), stability window,
                           kinetic variants, solution-set survey; bench/ holds the RTX 4070 Ti logs
  wo6/                     differentiable objectives and the cost of direct search
  wo6/phase2/              Kv7.2 D212G case study on the NEURON CA1 model (ModelDB 118986)
  wo97/                    re-simulation of the R859C best-value configurations in NEURON
  wo9/                     differentiable side of the model-bridge sensitivity test
Neuron_Test/R859C/         NEURON protocol files for the R859C case study and its archived voltage traces
mimic_cell_activity/HH.py  the differentiable Hodgkin-Huxley model class used by the wo2 experiments and Figure 11
mimic_cell_activity/fit_snapshots_2020-06-01/   saved images of the original fit behind Figure 11
scripts/fetch_modeldb.sh   downloads the two third-party NEURON models
pipeline/r859c/            the 2020 R859C NEURON search: exhaustive search, genetic algorithms, simulation-based inference
tools/                     a script that renamed keys in four data files (see its docstring)
env/                       conda environments and pip freezes of the machines the results were computed on
docs/                      reproduction notes and data that is not in the repository
```

Script-name prefixes (`wo2_`, `wo7_`, `wo98_`, ...) are the identifiers the experiments were run
under; they are kept so file names match the recorded outputs.

## Which script made which figure

Figure numbers follow the paper as submitted to PLOS Computational Biology (v8.15); [docs/FIGURE_NUMBERING.md](docs/FIGURE_NUMBERING.md) maps them to the preprint. **Script** draws the figure from the data file(s) in **Reads**;
**Measurement** is the simulation that produced that data, where it is separate. Figure scripts
need only NumPy and Matplotlib unless stated.

| # | Content | Script (`figures/scripts/`) | Reads | Measurement |
|---|---|---|---|---|
| 1 | Workflow diagram | — | — | a drawing, not computed |
| 2 | R859C ladder, trace by trace | `wo7_task12b_trace_ladder.py` | archived traces | as Figure 13 |
| 3 | Kv7.2 D212G reproduction | `wo10_kv7_published_spike_times.py` | `experiments/wo6/phase2/wo6b_taskD1_fig6a*`; panel b reads the published spike times off the graph shipped with ModelDB 118986 (`scripts/fetch_modeldb.sh`), and writes them to `figures/kv7_2_published_spike_times_digitised.{csv,json}` | `experiments/wo6/phase2/wo6b_taskD1_fig6a.py` (NEURON) |
| 4 | Kv7.2 stimulus ladder | `wo7_task21d_kv7_ladder_plot.py` | `figures/scripts/wo7_kv7_ladder_traces.{npz,json}` | `figures/scripts/wo7_task21d_kv7_ladder_run.py` (NEURON) |
| 5 | Kv7.2 restoration | `wo7_task21b_restoration.py` | `experiments/wo6/phase2/wo6b_taskD2_search.json` | `experiments/wo6/phase2/wo6b_taskD2_search.py` (NEURON) |
| 6 | Solution topology vs threshold | `wo7_task23_topology_plot.py` | `figures/scripts/wo7_topology_sweep.json` | `figures/scripts/wo7_task23_topology_sweep.py` (GPU) |
| 7 | Efficacy distribution (R859C) | `wo98_task14_efficacy_d37.py` | archived candidate database, see [docs/DATA_EXTERNAL.md](docs/DATA_EXTERNAL.md); plotted data in `figures/mutation1_efficacy_distribution.csv` | the R859C search, `pipeline/r859c/Explore8/` (NEURON; see its README) |
| 8 | Stability window | `wo7_task24_stability.py` | `experiments/wo6/phase2/wo6b_taskE_stability_*.json`, `experiments/wo5/task11d_summary.json` | `experiments/wo6/phase2/wo95_taskE_stability_fresh.py` (NEURON), `experiments/wo5/task11d_direct.py` (GPU) |
| 9 | Log-uniform vs uniform sampling | `wo8_task22_sampling_plot.py` | `figures/scripts/wo7_sampling.json` | `figures/scripts/wo7_task22_sampling_run.py` (GPU) |
| Table 1 | Throughput | `wo7_task25_throughput_table.py` | `figures/throughput_repeats.csv` | `wo9_task4_repeats.py` over five launches per card of `experiments/wo3/wo3_taskC2_clean.py` |
| 10 | Throughput vs gradient success | `wo98_task25_throughput_vs_gradient_medians.py` | literals in the script: the Table 1 rates, and the gradient-descent and direct-search results of `experiments/wo5/task11c_stability.py`, `task11d_direct.py`, `task12_topology.py` and `task13b_formB.py` | as Table 1, and those experiments |
| 11 | Convergence phase plane | `wo8_task12_fig11.py` | runs `mimic_cell_activity/HH.py` through `wo7_hh.py` | needs PyTorch |
| 12 | Pipeline summary | `wo98_gap2_summary_figure.py` | data files of Figures 13, 7, 8 and Table 1 | none |
| 13 | R859C excitability ladder | `wo7_task13_excitability.py` | archived traces in `Neuron_Test/R859C/`, and `figures/similarity_denominator_d37.json` | NEURON, ModelDB 87585 mechanisms; what wrote the archived traces is not identified ([docs/REPRODUCE.md](docs/REPRODUCE.md), section 3) |
| 14 | Dynamic time warping schematic | `make_fig_dtw.py` | — | computes its own warping path |

### Analyses behind numbers quoted in the text

| Script (`figures/scripts/`) | What it computes |
|---|---|
| `wo7_fig4_caption_evidence.py` | per-level spike counts of the two R859C arms |
| `wo7_task21a_reproduction.py` | `figures/kv7_2_reproduction_published_protocol.csv`, the simulated spike times of Figure 3; the figure it also draws shows the downloaded ModelDB screenshot for local comparison and is not the one in the paper |
| `similarity_denominator_d37.py` | the efficacy denominator 37 of Equation 4, from the reference and variant ladders simulated in NEURON |
| `wo7_dtw_denominator.py` | the DTW similarity denominator (needs the external database) |
| `wo9_task11_slow_inactivation.py`, `wo9_task11_neuron_verification.py` | what the R859C mechanism implements, and the same check inside NEURON |
| `wo9_task31_dimensionality_null.py`, `wo9_task31_null_ensemble*.py` | the dimension-counting null model for the solution topology |
| `wo9_task33_dtw_decomposition.py`, `wo9_task33_reconstruct.py` | how the DTW tests and the interventions decompose (needs the external database) |
| `wo9_bridge_sensitivity_{neuron,differentiable}.py` | one-conductance-at-a-time sensitivity in both models |
| `wo9_task4_repeats.py` | the launch-to-launch spread behind Table 1 |
| `experiments/wo97/step4_full_run.py`, `step5_analysis.py` | the six distinct firing patterns among the 10,780 best-value R859C configurations |

## The forward models

**Differentiable Hodgkin-Huxley (PyTorch).** Single compartment, three conductances, forward
Euler. `experiments/wo5/wo5_lib.py` is the library the sweeps use. `mimic_cell_activity/HH.py`
holds the model class that `experiments/wo2_lib.py` and `figures/scripts/wo7_hh.py` load; both
compute the same arithmetic in the same order. The conductance update in the paper's Algorithm 1 is the batched
`fit_grid` function of `experiments/wo2_lib.py`.

**NEURON, Nav1.1 R859C.** ModelDB accession 87585 (Barela et al. 2006), downloaded by
`scripts/fetch_modeldb.sh`. `Neuron_Test/R859C/` holds the 35-level current-ladder protocol
(`neuron.hoc`, 20-360 pA in 10 pA steps) and the two per-arm threshold runs, derived from that
entry's `APthreshold.hoc`, plus two archived voltage traces computed with that entry's
mechanisms.

**NEURON, Kv7.2 D212G.** The multi-compartment CA1 pyramidal model of Miceli et al. (2009),
ModelDB accession 118986, used unmodified and downloaded into `experiments/wo6/phase2/mutant/`.
The compensating search never changes the mutated channel; only the pharmacologically accessible
conductances move.

## Licence

Copyright 2026 Hananel Hazan. The repository carries two licences.

- **Code** (`*.py`, `*.sh`, `*.yml` and `requirements.txt`): the **Apache License 2.0, subject to
  the Tufts Open Source License Rider v.1 – Academic Use Only**. The Rider limits the rights
  granted by the Apache License to academic, non-commercial research. Both texts are in
  [LICENSE](LICENSE).
- **Data** (every `*.csv`, `*.json`, `*.npz`, `*.npy`, `*.log` and `*.out` file and every `*.md` table under
  `figures/` and `experiments/`, the archived traces in `Neuron_Test/R859C/`, the images in
  `mimic_cell_activity/fit_snapshots_2020-06-01/`, and the databases and logs deposited on Zenodo):
  **Creative Commons Attribution 4.0 International**, in [LICENSE-DATA](LICENSE-DATA).

The `.hoc` files derived from ModelDB accession 87585 keep their original attribution.
Third-party material is listed in [NOTICE](NOTICE); the two ModelDB models are not
redistributed here.

## Citing

If you use this code, please cite the paper. [CITATION.cff](CITATION.cff) has the entry.
