"""Optimizer invariance sweep.

Measures whether the compensating conductances found by gradient descent depend
on the optimizer used to find them.

Setup: sodium held fixed at the variant value 140, potassium and leak free,
starting from the variant point (K=20, leak=0.3), target = the wild-type trace
(Na=120, K=36, leak=0.03). Integration step 10/1500 ms, 1500 steps, injected
current 8, mean-squared-error loss.

Comparison protocol:
  * every optimizer gets its own learning rate, chosen by a pre-sweep
    (stage 1 below) rather than a single shared value;
  * a non-negativity clamp is applied uniformly to every run, so that an
    optimizer is not scored as failing merely because it stepped a conductance
    negative. Whether it *would* have gone negative without the clamp is recorded
    separately;
  * the iteration budget is generous and the plateau point is reported rather
    than the budget.

The learning rates for the two free parameters keep the ratio used in
mimic_cell_activity/HH.py (leak 0.01, potassium 0.5, a ratio of 0.02). The
pre-sweep varies the overall scale.

Writes taskB_stage1.npz, taskB_stage2.npz.
"""
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

OPTIMIZERS = ["sgd", "asgd", "adam", "adam_ams", "adadelta", "adagrad",
              "rmsprop", "rprop"]
SCALES = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LEAK_RATIO = 0.02          # leak/K learning-rate ratio of HH.py (0.01 / 0.5)
CLAMP = 1e-6               # uniform non-negativity floor
STAGE1_ITERS = 120
STAGE2_ITERS = 1500
I = 8.0


def cfg(opt, scale):
    return dict(frozen="Na", start=L.VARIANT, optimizer=opt,
                lrs={"Na": 0.0, "K": scale, "l": scale * LEAK_RATIO},
                label=f"{opt}@{scale:g}")


def main():
    # ---------------- stage 1: learning-rate pre-sweep
    cfgs1 = [cfg(o, s) for o in OPTIMIZERS for s in SCALES]
    r1 = L.fit_grid(cfgs1, loss_name="mse", iters=STAGE1_ITERS, I=I,
                    clamp_min=CLAMP, reset_each_iter=False)
    final1 = r1["loss"][-1]
    print("=== stage 1: learning-rate pre-sweep "
          f"({STAGE1_ITERS} iterations, clamp on) ===", flush=True)
    print(f"{'optimizer':11s}" + "".join(f"{s:>11g}" for s in SCALES) + "   chosen")
    best = {}
    for o in OPTIMIZERS:
        row, vals = [], []
        for s in SCALES:
            b = cfgs1.index(cfg(o, s))
            v = final1[b]
            vals.append(v if np.isfinite(v) else np.inf)
            row.append("   diverged" if not np.isfinite(v) else f"{v:11.4g}")
        k = int(np.argmin(vals))
        best[o] = SCALES[k]
        print(f"{o:11s}" + "".join(row) + f"   {SCALES[k]:g}", flush=True)
    np.savez_compressed("taskB_stage1.npz", loss=r1["loss"],
                        labels=np.array([c["label"] for c in cfgs1]),
                        scales=np.array(SCALES),
                        optimizers=np.array(OPTIMIZERS))

    # ---------------- stage 2: long run at the chosen learning rate
    cfgs2 = [cfg(o, best[o]) for o in OPTIMIZERS]
    # run the same set once more without the clamp, to record who would go negative
    r2 = L.fit_grid(cfgs2, loss_name="mse", iters=STAGE2_ITERS, I=I,
                    clamp_min=CLAMP, reset_each_iter=False)
    r2n = L.fit_grid([dict(c) for c in cfgs2], loss_name="mse",
                     iters=min(300, STAGE2_ITERS), I=I,
                     clamp_min=None, reset_each_iter=False)

    print(f"\n=== stage 2: {STAGE2_ITERS} iterations at the chosen learning rate,"
          " clamp on ===", flush=True)
    print(f"{'optimizer':11s} {'lr_K':>8} {'lr_leak':>9} {'plateau':>8} "
          f"{'final K':>9} {'final leak':>11} {'final loss':>12} "
          f"{'neg?':>6} {'step shape':>12}")
    rows = []
    for b, o in enumerate(OPTIMIZERS):
        K, l, loss = r2["K"][:, b], r2["l"][:, b], r2["loss"][:, b]
        pi = L.plateau_iter(loss)
        dK = np.diff(K[:max(pi, 20)])
        dK = dK[np.isfinite(dK)]
        if len(dK) and np.mean(np.abs(dK)) > 0:
            cv = np.std(np.abs(dK)) / np.mean(np.abs(dK))
        else:
            cv = np.nan
        shape = "adaptive" if cv < 0.25 else "grad-prop"
        rows.append((o, best[o], best[o] * LEAK_RATIO, pi, K[-1], l[-1],
                     loss[-1], bool(r2n["would_go_negative"][b]), cv, shape))
        print(f"{o:11s} {best[o]:8g} {best[o]*LEAK_RATIO:9g} {pi:8d} "
              f"{K[-1]:9.4f} {l[-1]:11.5f} {loss[-1]:12.5g} "
              f"{str(bool(r2n['would_go_negative'][b])):>6} "
              f"{shape:>12} (cv={cv:.2f})", flush=True)

    ok = [r for r in rows if np.isfinite(r[6])]
    Ks = np.array([r[4] for r in ok])
    ls = np.array([r[5] for r in ok])
    Ls = np.array([r[6] for r in ok])
    print("\n--- spread across optimizers that produced a finite result ---",
          flush=True)
    print(f"n = {len(ok)} of {len(OPTIMIZERS)}")
    print(f"final K    : min {Ks.min():.4f}  max {Ks.max():.4f}  "
          f"mean {Ks.mean():.4f}  sd {Ks.std():.4f}  "
          f"relative spread {(Ks.max()-Ks.min())/Ks.mean()*100:.2f}%")
    print(f"final leak : min {ls.min():.5f}  max {ls.max():.5f}  "
          f"mean {ls.mean():.5f}  sd {ls.std():.5f}  "
          f"relative spread {(ls.max()-ls.min())/ls.mean()*100:.2f}%")
    print(f"final loss : min {Ls.min():.5g}  max {Ls.max():.5g}")
    # reference: loss of the variant start and of the wild-type point itself
    print(f"\nreference losses at fixed points (same target, same protocol):",
          flush=True)
    with torch.no_grad():
        tgt = L.simulate(*L.WT, I=I)
        for nm, p in [("variant start (140,20,0.30)", (140.0, 20.0, 0.30)),
                      ("wild type    (120,36,0.03)", L.WT)]:
            v = L.simulate(*p, I=I)
            print(f"  {nm}: mse = {float(L.loss_mse(v, tgt)[0]):.5g}")

    np.savez_compressed("taskB_stage2.npz", K=r2["K"], l=r2["l"], loss=r2["loss"],
                        gK=r2["gK"], gl=r2["gl"],
                        optimizers=np.array(OPTIMIZERS),
                        lr_K=np.array([best[o] for o in OPTIMIZERS]),
                        would_go_negative=r2n["would_go_negative"])


if __name__ == "__main__":
    main()
