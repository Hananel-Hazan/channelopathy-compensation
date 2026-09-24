# The R859C search pipeline (NEURON, 2020)

These are the scripts that searched the R859C NEURON model (ModelDB accession 87585) for
compensating interventions: the exhaustive search behind the paper's R859C results, the
genetic-algorithm runs, and the simulation-based-inference run that gives Table 1's
single-core NEURON rate. They ran in 2020 under SLURM on the Tufts University
high-performance computing cluster. The directory names are those of the original runs,
and each directory holds the `neuron.hoc` that its scripts loaded.

The files are the archived copies, cleaned in two ways only: absolute paths are replaced by
paths resolved from the script's own directory, and comments were tidied (spelling, a
mislabelled comment, commented-out lines that held absolute paths). In the launchers, the
cluster home paths became `$SLURM_SUBMIT_DIR` and the conda location, and a site-specific
reservation line was removed. For every Python file, the syntax tree with docstrings removed
is identical to the archived copy's, except for the path lines.

## Which script produced which result

The job logs named here are in the Zenodo deposit; `logs_excerpt/` quotes the lines behind
Table 1.

### `Explore8/`: the exhaustive search

| Stage | Script and launcher | Job | What it produced |
|---|---|---|---|
| 1, sampling | `exploration.v3.run.exp.py` (`run-ex8.exp.sh`) and `exploration.v3.run.py` (`run-ex8.sh`), each a 601-task array | not logged | per-sample summary files `db/*_stats.pkl` |
| 2 to 4, database | `exploration.v3.summary.py` (`run-ex8_summary.sh`) | 56900048, 56900992 | the candidate database `sumary_explor_result.100K.pbz2` |
| 6, interventions | `exploration.v3.summary.6.py` (`run-ex8_summary.6.sh`) | 56900993 | the 1,437,450 spike-count and 560,230 time-warping tests, `sumary_explor_result100K.mutationFix.pbz2` |

How each is tied to its run, and what is not certain:

- **Sampling.** The two run scripts differ only in the sampling box (lines 575 to 580):
  `run.exp.py` draws the leak from a factor of 50 below to 50 above its baseline and the two
  gated conductances from a factor of 5,000 below to 5,000 above; `run.py` uses a factor of
  10 for all three. The database contains values that only the wider box can produce, and
  most of its entries lie inside the narrower box, so both scripts contributed samples.
  Which array job ran which script is not recorded: no sampling log survives.
- **Database.** Both database logs print `Open old DB -->`, a line only
  `exploration.v3.summary.py` prints (line 768).
- **Interventions.** The log of job 56900993 prints `Program part 6 - Done`, and part 6
  exists only in `exploration.v3.summary.6.py`. **The archived file is not byte-identical to
  the code that ran**: it was last saved on 2020-11-25 at 11:21, about seven hours after the
  job started, and a report file the job wrote lacks a line ending the archived code
  writes. Applying the archived file's selection and construction logic to the archived
  database gives exactly the job's 1,437,450 and 560,230 test counts.

### `GA1/` to `GA6/`: the genetic-algorithm runs

Each directory's `neuron-ga.py` is the version its **last** job ran: in GA1 to GA5 the
script was saved 22 to 64 seconds before that job started, and GA6's version is confirmed
by a job's error trace quoting its line 506 (line 485 here, after the cleaning). **Two runs the paper cites used earlier
versions that were not kept**: GA3 job 56370369 (its logged fitness values, about
1.25 million, cannot come from the archived GA3 script, whose score is at most 35) and GA4
job 56496220 (it started before the archived GA4 script was saved). The `neuron.hoc` here
is the one the GA runs loaded; unlike `Neuron_Test/R859C/neuron.hoc`, it sets
`secondorder = 2`.

### `SBI-HH_d-1/`: simulation-based inference

`neuron-sbi-hh.py` is the only version in the archive. Its job log (job 56604885) says
`Running 250000 simulations`, the number set at line 383 of the archived copy (line 362
here), and ends with the `Done` its last
line prints. The job's start time is not logged, so the file is not byte-verified against
the run. Its progress bar gives Table 1's single-core NEURON rate (`logs_excerpt/`). Its
`neuron.hoc` adds two Hodgkin-Huxley-type cell templates to the protocol file.

## Not included

- `Explore2/exploration.py`, and `run.py` and `run2.py` from the protocol directory: none
  produced a number in the paper.
- The two candidate databases and the full job logs, which are too large for the
  repository: they are in the Zenodo deposit.

## Running them

Each directory expects the two ModelDB mechanisms compiled in place:

```bash
bash scripts/fetch_modeldb.sh                       # places the .mod files in Neuron_Test/R859C/
cd pipeline/r859c/Explore8                          # or GA1 ... GA6, SBI-HH_d-1
cp ../../../Neuron_Test/R859C/ichanWT2005.mod ../../../Neuron_Test/R859C/ichanR859C1.mod .
nrnivmodl
sbatch run-ex8_summary.6.sh                         # or run the python command in the launcher
```

Some NEURON 9 builds refuse the two mechanisms' `VERBATIM return 0; ENDVERBATIM` blocks;
`experiments/wo97/build_work.py` shows how to strip them from working copies. The Python
environment of the 2020 runs was not recorded; the launchers activate conda environments
named `deap` (Explore8 and GA: NumPy, SciPy, dtaidistance, tqdm, Matplotlib, DEAP, SCOOP and
NEURON) and `sbi` (PyTorch and sbi). The scripts use `np.float`, which NumPy 1.24 removed.
