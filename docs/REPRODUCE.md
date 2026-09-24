# Reproducing the results

All scripts locate their inputs relative to the repository root, so they can be run from any
working directory. Nothing needs to be installed system-wide.

## 1. Figures and Table 1 from the shipped data (minutes, any machine)

```bash
python3 -m pip install -r requirements.txt       # numpy, matplotlib
cd figures/scripts
python3 wo7_task13_excitability.py               # Figure 13 (reads figures/similarity_denominator_d37.json)
python3 wo7_task12b_trace_ladder.py              # Figure 2
python3 make_fig_dtw.py                          # Figure 14
python3 wo8_task22_sampling_plot.py              # Figure 9
python3 wo7_task21b_restoration.py               # Figure 5
python3 wo7_task21d_kv7_ladder_plot.py           # Figure 4
python3 wo7_task23_topology_plot.py              # Figure 6
python3 wo7_task24_stability.py                  # Figure 8
python3 wo7_task25_throughput_table.py           # Table 1
python3 wo98_task25_throughput_vs_gradient_medians.py   # Figure 10
python3 wo98_gap2_summary_figure.py              # Figure 12
python3 wo7_fig4_caption_evidence.py             # per-level spike counts quoted in the text
```

Figures are written to `figures/` as PDF and PNG. A figure script refuses to overwrite an
existing figure file; delete it to rebuild.

These twelve scripts were run on a clean copy of this repository before release. Every data
file they rewrite (CSV/JSON in `figures/`) came back byte-identical to the shipped copy.
Figure numbers follow the paper as submitted to PLOS Computational Biology; the preprint used
different numbers, and [FIGURE_NUMBERING.md](FIGURE_NUMBERING.md) maps one to the other.

Three figures need more than NumPy and Matplotlib:

| Figure | Needs |
|---|---|
| 3 (`wo10_kv7_published_spike_times.py`) | panel b reads the spike times off the graph shipped inside ModelDB 118986: run `bash scripts/fetch_modeldb.sh` first. The digitised times it writes are already in `figures/kv7_2_published_spike_times_digitised.csv` |
| 7 (`wo98_task14_efficacy_d37.py`) | the archived candidate database, see [DATA_EXTERNAL.md](DATA_EXTERNAL.md); the plotted distribution itself is in `figures/mutation1_efficacy_distribution.csv` |
| 11 (`wo8_task12_fig11.py`) | PyTorch (it runs `mimic_cell_activity/HH.py`) |

## 2. The differentiable-model experiments (graphics card)

These need PyTorch with CUDA. The published runs used PyTorch 2.10.0 built against CUDA 12.8 on
an RTX 2070 and an RTX 4070 Ti; `env/` holds the exact conda environments and pip freezes, and
`env/setup_env.sh` creates an equivalent environment named `hh-compensation`.

| Result | Command |
|---|---|
| Figure 9 data | `python3 figures/scripts/wo7_task22_sampling_run.py` → `figures/scripts/wo7_sampling.json` |
| Figure 6 data | `python3 figures/scripts/wo7_task23_topology_sweep.py` → `figures/scripts/wo7_topology_sweep.json` |
| Topology null model | `figures/scripts/wo9_task31_dimensionality_null.py`, `wo9_task31_null_ensemble.py`, `wo9_task31_null_ensemble_allvariants.py` |
| Table 1 launches | `bash experiments/wo3/wo9_4_run_repeats.sh` (RTX 2070) and `bash experiments/wo5/bench/wo9_4_run_repeats_4070.sh` (RTX 4070 Ti), then `python3 figures/scripts/wo9_task4_repeats.py` |
| Table 1, one processor core | `cd experiments && python3 taskD_cpu_big.py` (CPU only; the recorded output is `experiments/out_taskD_cpu_big_py.log`, see section 4) |
| Table 1, memory traffic (RTX 2070) | `cd experiments/wo3 && python3 wo3_taskC1.py` (output `out_taskC1.log`), then `python3 wo3_taskC1_bandwidth.py` and `python3 wo3_taskC2_profile.py` (output `out_taskC_profile.log`) |
| Model-bridge sensitivity, differentiable side | `python3 figures/scripts/wo9_bridge_sensitivity_differentiable.py` → `experiments/wo9/sens_diff.json` |
| Optimizer and loss comparisons | `cd experiments && python3 wo2_task*.py` (each writes its `task*.npz` to the current directory; the optimizer comparison's printed output is `experiments/out_taskB2.log`) |
| Stability window, single compartment | `cd experiments/wo5 && python3 task11c_stability.py && python3 task11d_direct.py` |
| Kinetic-variant compensation | `cd experiments/wo5 && python3 task13b_formB.py` |
| Differentiable objectives, search cost | `cd experiments/wo6 && python3 wo6_taskA.py`, `wo6_taskB.py`, `wo6_taskC.py`, `wo6_taskC2.py`, `wo6_taskC5.py`, `wo6_taskC5b.py` |

The wo2 and wo5 experiments, and the wo6 ones, write their output files to the current working
directory, so run them from the directory that holds the shipped copies.

`wo9_task4_repeats.py` compares each median with the original single run and exits non-zero
when one differs by more than 5 %. One row is known to: the RTX 4070 Ti cache-resident batch, single
run 107,012.9 against a five-launch median of 100,443.4 (6.1 %). Table 1 reports the median with its
full range, so for that row the script prints a note instead and exits 0; any other row outside 5 %
still fails.

## 3. The NEURON experiments

Install NEURON (`pip install neuron`). The NEURON environment of the published runs has NEURON
9.0.1 (`env/pip-freeze_neuron.txt`); `wo9_task11_neuron_verification.py` was run with 9.0.2, as
its output `figures/wo9_r859c_neuron_verification.json` records, and
`similarity_denominator_d37.py` with 9.0.1. Then fetch and compile the two third-party models:

```bash
bash scripts/fetch_modeldb.sh
(cd Neuron_Test/R859C             && nrnivmodl)
(cd experiments/wo6/phase2/mutant && nrnivmodl)
```

Some NEURON 9 builds refuse to compile the `VERBATIM return 0; ENDVERBATIM` blocks in the two
R859C mechanisms. `experiments/wo97/build_work.py` and
`figures/scripts/wo9_task11_neuron_verification.py` strip those no-op blocks from working copies
before compiling; the header of the latter also documents an `LD_PRELOAD` workaround for the
system `libstdc++` on some distributions.

| Result | Script |
|---|---|
| Figure 3 data | `experiments/wo6/phase2/wo6b_taskD1_fig6a.py` |
| Figure 5 data (conductance search) | `experiments/wo6/phase2/wo6b_taskD2_search.py`, merged by `wo6b_analyse.py` |
| Figure 4 traces | `figures/scripts/wo7_task21d_kv7_ladder_run.py` |
| Figure 8 data (stability, multi-compartment) | `experiments/wo6/phase2/wo95_taskE_stability_fresh.py` (one fresh process per evaluation, so results do not depend on evaluation order) |
| Transfer from the differentiable model | `experiments/wo6/phase2/wo6b_taskF_transfer.py` |
| R859C mechanism checks | `figures/scripts/wo9_task11_neuron_verification.py`, `wo9_bridge_sensitivity_neuron.py` |
| Best-value firing patterns | `python3 experiments/wo97/build_work.py`, then `step4_full_run.py` and `step5_analysis.py` |
| The original R859C search, genetic algorithms and simulation-based inference (2020) | `pipeline/r859c/`, SLURM launchers included; its README says which script produced which result |
| Table 1, NEURON rates (1.1, 16.4, 16.9) | read from the job logs: `pipeline/r859c/logs_excerpt/` |
| Efficacy denominator 37 (Equation 4) | `python3 experiments/wo97/build_work.py`, then `python3 figures/scripts/similarity_denominator_d37.py` → `figures/similarity_denominator_d37.json` |

`wo9_task11_slow_inactivation.py` needs SciPy for part C, which always runs, and runs its two slow
pure-Python simulations, parts B and D, only with `--with-sim`.

The archived R859C voltage traces in `Neuron_Test/R859C/` (the two
`APthreshold ... - log every 10th data point.txt` files) are dated 2020-07-07 and were computed
with the ModelDB 87585 mechanisms. What wrote them is not identified. The write calls in the
two `APthreshold *.hoc` files are commented out, and they would write a different header; the
files begin with a `label:` line and then the row count, 3000, and no script found in the
project writes that layout.

## 4. Data with no script that regenerates them

Some files and numbers were produced by steps that were never saved as a script, or whose raw
output does not survive. They are listed here rather than given a script written after the
fact.

**`figures/wo97_best_value_patterns.csv`.** Written on 2026-09-06 by a one-off command, not a
saved script, that read `experiments/wo97/wo97_patterns_described.json` and
`experiments/wo97/wo97_measurement.json` and wrote one row per firing pattern. The copy here
omits that command's column `distance_to_archived_wild_type` and its comment header. Every
value in it can be checked against the two JSON files.

**`figures/wo97_best_value_configurations.csv`.** The 10,780 best-value R859C configurations,
extracted from the archived candidate database `sumary_explor_result100K.mutationFix.pbz2`
(see [DATA_EXTERNAL.md](DATA_EXTERNAL.md)). The extraction step is not recorded: no script that
writes this file was found. `experiments/wo97/step4_full_run.py` and `step5_analysis.py` read it.

**`figures/wo9_model_bridge_sensitivity.json`.** Combines the NEURON side, written by
`figures/scripts/wo9_bridge_sensitivity_neuron.py`, with the differentiable side,
`experiments/wo9/sens_diff.json`. The combining step was a one-off command, and the NEURON
side's output file (`sens_nrn.json`) was not kept, so only the rounded percentages in this
JSON survive for the NEURON side. Rerunning `wo9_bridge_sensitivity_neuron.py` regenerates
that input.

**Table 1, one processor core (0.089 and 20.8 sets per second).** From
`experiments/out_taskD_cpu_big_py.log` (lines 2 and 7: 0.0885 and 20.7534), written by
`experiments/taskD_cpu_big.py`. A second run of the same script, a minute earlier, is
`experiments/taskD_cpu_big.out` (0.0843 and 21.1543); no record says why the table uses the
first. `experiments/taskD_cpu.out` is the output of `wo2_taskD_benchmark.py`, which stops at 16
sets per batch and gives 0.0871 for one set.

**Table 1, memory traffic.** The RTX 2070 values (393.6 GB/s measured copy bandwidth; 412 to 420
GB/s demanded before fusion; 51 to 64 GB/s after) are in `experiments/wo3/out_taskC_profile.log`.
**The RTX 4070 Ti values (426.9 GB/s and 116 GB/s) have no log in this repository**: the logs of
that measurement were not kept on the machine the repository was built from. The ratio 5.37 is
not a measurement: it is the ratio of the two cards' rated single-precision throughputs, 40.09
and 7.465 trillion floating-point operations per second, entered as rated values rather than
measured.

**The original R859C search code.** Two points are recorded in `pipeline/r859c/README.md`. The
archived `exploration.v3.summary.6.py`, which ran the 1,437,450 spike-count and 560,230
time-warping tests, was saved after that job started, so it is not byte-identical to the code
that ran; its logic reproduces both test counts exactly. The genetic-algorithm scripts are the
versions of each directory's last job; the two genetic-algorithm runs the paper cites (GA3 job
56370369 and GA4 job 56496220) used earlier versions that were not kept.

**Figure 11, the fitted conductances (140, 39.678, 0.155).** Read from the file name of the
epoch-150 image in `mimic_cell_activity/fit_snapshots_2020-06-01/`, the only record of that
fit; its README gives the details. No run log of the fit exists.
