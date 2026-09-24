"""Locate the edge of the compensation envelope (wide sodium sweep).

The E2 sweep of wo2_taskE.py (sodium 100 to 200, i.e. 83% to 167% of the
wild-type value) found no failure: compensation closed 99.36% to 99.99% of the
gap at every point. This sweep widens the range by more than an order of
magnitude in each direction to find where compensation stops working.

Same setup as E2: sodium held fixed, potassium and leak free from (20, 0.30),
target the wild-type trace, Adam, uniform non-negativity clamp, 400 iterations.
Success is measured on the best loss reached, not the final loss, because the
forward-Euler integrator can blow up after convergence.

Writes taskE2_wide.npz.
"""
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

I, ITERS, CLAMP = 8.0, 400, 1e-6
LR = {"Na": 0.0, "K": 0.15, "l": 0.001}
GNA = np.array([5, 10, 20, 40, 60, 80, 100, 140, 200, 260, 320, 400, 500,
                650, 800, 1000, 1400, 2000], dtype=float)


def main():
    cfgs = [dict(frozen="Na", start=(g, 20.0, 0.30), optimizer="adam",
                 lrs=LR, label=f"gNa={g:g}") for g in GNA]
    r = L.fit_grid(cfgs, loss_name="mse", iters=ITERS, I=I, clamp_min=CLAMP,
                   reset_each_iter=False)
    loss, K, l = r["loss"], r["K"], r["l"]

    with torch.no_grad():
        tgt = L.simulate(*L.WT, I=I)
        wt_self = float(L.loss_mse(tgt, tgt)[0])
    print(f"wild type against itself: {wt_self:.6g}\n")
    print(f"{'g_Na':>7} {'% of WT':>9} {'start':>9} {'best':>10} {'gap closed':>11} "
          f"{'K':>9} {'leak':>9}  note")
    closed = []
    for b, g in enumerate(GNA):
        L0 = loss[0, b]
        if not np.isfinite(loss[:, b]).any():
            print(f"{g:7g} {g/120*100:8.1f}% {L0:9.4g} {'--':>10} {'--':>11} "
                  f"{'--':>9} {'--':>9}  never produced a finite loss")
            closed.append(np.nan)
            continue
        i = int(np.nanargmin(loss[:, b]))
        L1 = loss[i, b]
        c = (1 - L1 / L0) * 100
        closed.append(c)
        ndiv = int(np.sum(~np.isfinite(loss[:, b])))
        fin = np.where(np.isfinite(loss[:, b]))[0]
        note = f"diverged after iter {int(fin[-1])}" if ndiv else ""
        print(f"{g:7g} {g/120*100:8.1f}% {L0:9.4g} {L1:10.4g} {c:10.3f}% "
              f"{K[i,b]:9.3f} {l[i,b]:9.5f}  {note}", flush=True)

    closed = np.array(closed)
    good = GNA[np.nan_to_num(closed, nan=-1) > 95.0]
    print(f"\nsodium values where at least 95% of the gap was closed: "
          f"{good.min():g} to {good.max():g}  "
          f"({good.min()/120*100:.0f}% to {good.max()/120*100:.0f}% of wild type)")
    np.savez_compressed("taskE2_wide.npz", gna=GNA, loss=loss, K=K, l=l,
                        closed=closed)


if __name__ == "__main__":
    main()
