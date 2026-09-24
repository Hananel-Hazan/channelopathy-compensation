"""Does the potassium freeze under the correlation loss depend on optimizer adaptivity?

With the correlation loss, the derivative with respect to potassium is about four
orders of magnitude smaller than with mean squared error, so potassium can stay
nearly unchanged while the other parameters converge. That happens only if the
optimizer step is proportional to the gradient. An adaptive optimizer divides
the gradient by its own running magnitude, so a 10,000-times-smaller but
sign-consistent gradient still produces a step of roughly the learning rate.

This script runs the identical setup (sodium fixed at the variant value,
potassium and leak free) under non-adaptive and adaptive optimizers, with both
the correlation and the mean-squared-error loss, and reports how far potassium
and leak actually move.

Writes taskC.npz.
"""
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

I = 8.0
ITERS = 300
CLAMP = 1e-6

# Learning rates: those set in mimic_cell_activity/HH.py for the non-adaptive
# family (leak 0.01, K 0.5; HH.py itself uses ASGD) and, for the Adam family,
# the values that reproduce the recorded run analysed in wo2_taskA.py
# (leak 0.001, K 0.15). The label strings are stored in taskC.npz.
CONFIGS = [
    ("sgd",       {"K": 0.5,  "l": 0.01},  "non-adaptive, repository lrs"),
    ("asgd",      {"K": 0.5,  "l": 0.01},  "non-adaptive, repository lrs (HH.py:124)"),
    ("sgd",       {"K": 0.15, "l": 0.001}, "non-adaptive, Adam-family lrs"),
    ("asgd",      {"K": 0.15, "l": 0.001}, "non-adaptive, Adam-family lrs"),
    ("adam",      {"K": 0.15, "l": 0.001}, "adaptive"),
    ("adam_ams",  {"K": 0.15, "l": 0.001}, "adaptive"),
    ("rmsprop",   {"K": 0.15, "l": 0.001}, "adaptive"),
    ("adagrad",   {"K": 0.15, "l": 0.001}, "adaptive"),
]


def run(loss_name):
    cfgs = [dict(frozen="Na", start=L.VARIANT, optimizer=o,
                 lrs={"Na": 0.0, **lr}, label=f"{o} {note}")
            for o, lr, note in CONFIGS]
    return cfgs, L.fit_grid(cfgs, loss_name=loss_name, iters=ITERS, I=I,
                            clamp_min=CLAMP, reset_each_iter=False)


def report(loss_name, cfgs, r):
    print(f"\n=== loss = {loss_name}, {ITERS} iterations, injected current {I} ===",
          flush=True)
    print(f"{'optimizer':11s} {'class':28s} {'lr_K':>7} {'lr_leak':>8} "
          f"{'K start':>8} {'K end':>9} {'K moved %':>10} "
          f"{'leak end':>9} {'leak moved %':>13} {'|dK/dg_K| ratio':>16}")
    for b, (o, lr, note) in enumerate(CONFIGS):
        K, l = r["K"][:, b], r["l"][:, b]
        dKpc = (K[-1] - 20.0) / 20.0 * 100
        dlpc = (l[-1] - 0.30) / 0.30 * 100
        gK = np.nanmedian(np.abs(r["gK"][:, b]))
        stepK = np.nanmedian(np.abs(np.diff(K)))
        ratio = stepK / gK if gK else np.nan
        print(f"{o:11s} {note:28s} {lr['K']:7g} {lr['l']:8g} "
              f"{20.0:8.2f} {K[-1]:9.4f} {dKpc:10.2f} "
              f"{l[-1]:9.5f} {dlpc:13.2f} {ratio:16.4g}", flush=True)


def main():
    print("Does the potassium freeze depend on optimizer adaptivity?")
    out = {}
    for loss_name in ("correlation", "mse"):
        cfgs, r = run(loss_name)
        report(loss_name, cfgs, r)
        out[f"K_{loss_name}"] = r["K"]
        out[f"l_{loss_name}"] = r["l"]
        out[f"loss_{loss_name}"] = r["loss"]
        out[f"gK_{loss_name}"] = r["gK"]
        out[f"gl_{loss_name}"] = r["gl"]
    out["labels"] = np.array([f"{o}|{lr['K']}|{note}" for o, lr, note in CONFIGS])
    np.savez_compressed("taskC.npz", **out)


if __name__ == "__main__":
    main()
