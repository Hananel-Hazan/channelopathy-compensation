# Fit snapshots of 2020-06-01

These 20 images are the saved output of the original gradient-descent fit of the
differentiable Hodgkin-Huxley model. They are the only record of that run: no log
of it survives. Each file name gives the epoch and the three conductances (sodium,
potassium, leak, in mS/cm²) at that epoch.

The fitted vector drawn in the paper's Figure 11, (140, 39.678, 0.155), is read
from the file name `150 - Na(140.000) K(39.678) l(0.155).png`.

## Two runs share this folder

The file times, recorded on the original disk, separate them:

| Run | Epochs | Written (2020-06-01) | Potassium conductance |
|---|---|---|---|
| short run | 0 to 30 | 18:11:50 to 18:13:03 | stays near 20.00 |
| main run | 0 to 150 | 18:13:26 to 18:44:57 | rises from 20.160 to 39.678 |

The two epoch-0 images are byte-identical.

## How they were made

The fitting script that wrote them was an earlier version of
`mimic_cell_activity/HH.py`, and it is not preserved. Its plotting routine had
two defects, which `figures/scripts/wo8_task12_fig11.py` describes and corrects
when it redraws Figure 11: the time axis is labelled 0 to 10 ms whatever the
number of steps, and the phase plane has its axes transposed. The images are
shipped unchanged, defects included.

These images are data and are licensed under CC BY 4.0 (see `LICENSE-DATA` and
`NOTICE`).
