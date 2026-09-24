# Where Table 1's NEURON rates come from

Excerpts of two job logs of the original runs, with the physical line number and byte
offset of each quoted line. The full logs are in the Zenodo deposit.

| Table 1 row | Rate (sets/s) | Derivation |
|---|---|---|
| NEURON, 72-core node, spike-count stage | 16.4 | 1,437,450 tests in 24 h 17 min 3 s (87,423 s) = 16.44 |
| NEURON, 72-core node, time-warping stage | 16.9 | 560,230 tests in 9 h 12 min 50 s (33,170 s) = 16.89 |
| NEURON, one processor core | 1.1 | the log reads 1.00 s per set at one point; over the whole run, 250,000 sets in 62 h 47 min 30 s (226,050 s) = 1.106 per second |

Both 72-core rates come from `res_56900993_rh7pcomp01.err`. The two stages ran in the same
job, which ended 33 h 29 min 53 s after it started; each rate divides a stage's tests by that
stage's own duration.

These excerpts are data and are licensed under CC BY 4.0.
