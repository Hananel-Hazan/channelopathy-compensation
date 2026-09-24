"""Extended learning-rate pre-sweep.

Three optimizers (adam_ams, adadelta, rprop) selected the largest scale of the
first pre-sweep (wo2_taskB.py stage 1), i.e. the edge of the grid. The grid is
extended upward here so that their best learning rate lies inside it.
sgd/asgd/adam/adagrad/rmsprop are included as controls.

Writes taskB_ext.npz.
"""
import numpy as np, torch
torch.set_num_threads(1)
import wo2_lib as L

OPT = ["sgd", "asgd", "adam", "adam_ams", "adadelta", "adagrad", "rmsprop", "rprop"]
SCALES = [0.003, 0.03, 0.3, 3.0, 30.0, 100.0, 300.0, 1000.0]
LEAK_RATIO = 0.02
cfgs = [dict(frozen="Na", start=L.VARIANT, optimizer=o,
             lrs={"Na": 0.0, "K": s, "l": s * LEAK_RATIO}, label=f"{o}@{s:g}")
        for o in OPT for s in SCALES]
r = L.fit_grid(cfgs, loss_name="mse", iters=120, I=8.0, clamp_min=1e-6,
               reset_each_iter=False)
f = r["loss"][-1]
print("extended pre-sweep, final mean-squared-error loss after 120 iterations")
print(f"{'optimizer':11s}" + "".join(f"{s:>11g}" for s in SCALES) + "     best")
for o in OPT:
    vals, row = [], []
    for s in SCALES:
        b = cfgs.index(dict(frozen="Na", start=L.VARIANT, optimizer=o,
                            lrs={"Na": 0.0, "K": s, "l": s*LEAK_RATIO},
                            label=f"{o}@{s:g}"))
        v = f[b]; vals.append(v if np.isfinite(v) else np.inf)
        row.append("   diverged" if not np.isfinite(v) else f"{v:11.4g}")
    k = int(np.argmin(vals))
    print(f"{o:11s}" + "".join(row) + f" {SCALES[k]:8g}", flush=True)
np.savez_compressed("taskB_ext.npz", loss=r["loss"], K=r["K"], l=r["l"],
                    labels=np.array([c["label"] for c in cfgs]),
                    scales=np.array(SCALES), optimizers=np.array(OPT))
