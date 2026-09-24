"""Identify the optimizer and learning rates of a recorded fitting run.

The recorded run is a fit made with the original mimic_cell_activity/HH.py
script (sodium fixed, potassium and leak free, starting from the variant point).
Its conductances at every tenth epoch are listed in RECORDED:

    epoch    0   K 20.160  leak 0.299
    epoch   10   K 21.742  leak 0.289
    ...
    epoch  150   K 39.678  leak 0.155

In that run the values recorded at epoch e were saved after that epoch's
optimizer step, so they are the state after e+1 updates.

The script replays the fit with Adam and AMSGrad over a grid of potassium
learning rates, with the optimizer state either reset every epoch (as the
original driver does) or kept, and prints the RMS distance of each replay to the
recorded points and how constant the potassium step size is.

Usage:  python3 wo2_taskA.py <loss_name> <injected_current>
Writes  taskA_<loss>_I<current>.npz
"""
import sys
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

RECORDED = np.array([
    (0, 20.160, 0.299), (10, 21.742, 0.289), (20, 23.294, 0.279),
    (30, 24.814, 0.269), (40, 26.304, 0.259), (50, 27.765, 0.249),
    (60, 29.197, 0.239), (70, 30.600, 0.229), (80, 31.976, 0.219),
    (90, 33.324, 0.209), (100, 34.646, 0.199), (110, 35.941, 0.189),
    (120, 37.211, 0.179), (130, 38.455, 0.169), (140, 39.675, 0.159),
    (150, 39.678, 0.155),
])

LR_K = [0.10, 0.12, 0.15, 0.16, 0.20]
LR_L = 0.001
ITERS = 151


def build_configs(reset):
    cfgs = []
    for opt in ("adam", "adam_ams"):
        for lrk in LR_K:
            cfgs.append(dict(frozen="Na", start=L.VARIANT, optimizer=opt,
                             lrs={"Na": 0.0, "K": lrk, "l": LR_L},
                             label=f"{opt} lrK={lrk} lrl={LR_L} reset={reset}"))
    return cfgs


def residual(K, l):
    """Root-mean-square distance to the recorded points, in each parameter."""
    idx = RECORDED[:, 0].astype(int)
    if len(K) <= idx.max():
        return np.nan, np.nan
    dK = K[idx] - RECORDED[:, 1]
    dl = l[idx] - RECORDED[:, 2]
    return float(np.sqrt(np.nanmean(dK ** 2))), float(np.sqrt(np.nanmean(dl ** 2)))


def main():
    loss_name = sys.argv[1]
    I = float(sys.argv[2])
    out = {}
    for reset in (True, False):
        cfgs = build_configs(reset)
        r = L.fit_grid(cfgs, loss_name=loss_name, iters=ITERS, I=I,
                       reset_each_iter=reset)
        tag = "reset" if reset else "persist"
        out[f"K_{tag}"] = r["K"]
        out[f"l_{tag}"] = r["l"]
        out[f"loss_{tag}"] = r["loss"]
        out[f"gK_{tag}"] = r["gK"]
        out[f"gl_{tag}"] = r["gl"]
        out[f"labels_{tag}"] = np.array([c["label"] for c in cfgs])

        print(f"=== loss={loss_name} I={I} optimizer-state={tag} ===", flush=True)
        print(f"{'configuration':38s} {'K@150':>9} {'leak@150':>9} "
              f"{'rmsE K':>9} {'rmsE leak':>10} {'step K const?':>14}", flush=True)
        for b, c in enumerate(cfgs):
            K, l = r["K"][:, b], r["l"][:, b]
            rk, rl = residual(K, l)
            dK = np.diff(K[:141])
            const = (np.nanstd(dK) / abs(np.nanmean(dK))) if np.nanmean(dK) else np.nan
            print(f"{c['label']:38s} {K[150]:9.3f} {l[150]:9.4f} "
                  f"{rk:9.3f} {rl:10.4f} {const:14.4f}", flush=True)
        print(flush=True)

    np.savez_compressed(f"taskA_{loss_name}_I{int(I)}.npz", recorded=RECORDED, **out)


if __name__ == "__main__":
    main()
