"""Optimizer invariance run (1500 iterations, eight optimizers).

Learning rates come from the union of the two pre-sweeps (taskB_stage1.npz,
scales 1e-4 to 10; taskB_ext.npz, scales 3e-3 to 1000), taking for each
optimizer the scale that gave the lowest mean-squared-error loss after 120
iterations. Every chosen scale is an interior point of the combined grid, so no
optimizer is left at an edge of the grid where a better learning rate might lie
untested.

Setup: sodium held fixed at the variant value 140, potassium and leak free,
starting from (K=20, leak=0.30), target the wild-type trace (120, 36, 0.03),
integration step 10/1500 ms, 1500 steps, injected current 8, mean squared error.
The two free learning rates keep the leak-to-potassium ratio of 0.02 used in
mimic_cell_activity/HH.py. Optimizer state is kept across iterations.

A uniform non-negativity clamp at 1e-6 is applied to every run, so that no
optimizer is scored as failing merely for stepping a conductance negative. A
separate shorter run without the clamp records which ones would have.

Writes taskB2.npz.
"""
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

BEST = {"sgd": 0.003, "asgd": 0.003, "adam": 0.3, "adam_ams": 10.0,
        "adadelta": 30.0, "adagrad": 3.0, "rmsprop": 0.1, "rprop": 30.0}
PRESWEEP = {"sgd": 2.263, "asgd": 1.787, "adam": 1.469, "adam_ams": 0.4262,
            "adadelta": 0.2768, "adagrad": 1.394, "rmsprop": 1.799,
            "rprop": 0.2702}
OPT = list(BEST)
RATIO, CLAMP, ITERS, I = 0.02, 1e-6, 1500, 8.0


def cfgs():
    return [dict(frozen="Na", start=L.VARIANT, optimizer=o,
                 lrs={"Na": 0.0, "K": BEST[o], "l": BEST[o] * RATIO}, label=o)
            for o in OPT]


def main():
    r = L.fit_grid(cfgs(), loss_name="mse", iters=ITERS, I=I, clamp_min=CLAMP,
                   reset_each_iter=False)
    rn = L.fit_grid(cfgs(), loss_name="mse", iters=300, I=I, clamp_min=None,
                    reset_each_iter=False)

    with torch.no_grad():
        tgt = L.simulate(*L.WT, I=I)
        L_variant = float(L.loss_mse(L.simulate(*L.VARIANT, I=I), tgt)[0])
        L_wt = float(L.loss_mse(L.simulate(*L.WT, I=I), tgt)[0])

    print(f"reference: untreated variant loss {L_variant:.6g}; "
          f"wild type against itself {L_wt:.6g}\n", flush=True)
    print(f"{'optimizer':10s} {'lr_K':>8} {'lr_leak':>9} {'plateau':>8} "
          f"{'final K':>9} {'final leak':>11} {'final loss':>12} "
          f"{'gap closed':>11} {'neg?':>6} {'step shape':>11}")
    rows = []
    for b, o in enumerate(OPT):
        K, l, loss = r["K"][:, b], r["l"][:, b], r["loss"][:, b]
        fin = np.where(np.isfinite(loss))[0]
        last = int(fin[-1]) if len(fin) else -1
        pi = L.plateau_iter(loss) if len(fin) > 20 else -1
        dK = np.abs(np.diff(K[:max(pi, 20)]))
        dK = dK[np.isfinite(dK)]
        cv = (np.std(dK) / np.mean(dK)) if len(dK) and np.mean(dK) > 0 else np.nan
        shape = "adaptive" if cv < 0.25 else "grad-prop"
        closed = (1 - loss[last] / L_variant) * 100 if last >= 0 else np.nan
        rows.append((o, K[last], l[last], loss[last], closed))
        print(f"{o:10s} {BEST[o]:8g} {BEST[o]*RATIO:9g} {pi:8d} {K[last]:9.4f} "
              f"{l[last]:11.5f} {loss[last]:12.5g} {closed:10.4f}% "
              f"{str(bool(rn['would_go_negative'][b])):>6} {shape:>11} "
              f"(cv={cv:.2f})", flush=True)

    ok = [x for x in rows if np.isfinite(x[3])]
    Ks = np.array([x[1] for x in ok])
    ls = np.array([x[2] for x in ok])
    Ls = np.array([x[3] for x in ok])
    Cs = np.array([x[4] for x in ok])
    print(f"\n--- spread across the {len(ok)} of {len(OPT)} optimizers that "
          "produced a finite result ---")
    print(f"final potassium : {Ks.min():.4f} .. {Ks.max():.4f}   "
          f"mean {Ks.mean():.4f}   sd {Ks.std():.4f}   "
          f"full spread {(Ks.max()-Ks.min())/Ks.mean()*100:.2f}% of the mean")
    print(f"final leak      : {ls.min():.5f} .. {ls.max():.5f}   "
          f"mean {ls.mean():.5f}   sd {ls.std():.5f}   "
          f"full spread {(ls.max()-ls.min())/ls.mean()*100:.2f}% of the mean")
    print(f"final loss      : {Ls.min():.5g} .. {Ls.max():.5g}")
    print(f"gap closed      : {Cs.min():.4f}% .. {Cs.max():.4f}%")
    print("\nwild-type potassium is 36 and wild-type leak is 0.03, given for "
          "reference only:\nthe compensating point need not equal the "
          "wild-type point, because sodium is held\nat the variant value.")

    np.savez_compressed("taskB2.npz", K=r["K"], l=r["l"], loss=r["loss"],
                        gK=r["gK"], gl=r["gl"], optimizers=np.array(OPT),
                        lr_K=np.array([BEST[o] for o in OPT]),
                        would_go_negative=rn["would_go_negative"],
                        L_variant=L_variant, L_wt=L_wt)


if __name__ == "__main__":
    main()
