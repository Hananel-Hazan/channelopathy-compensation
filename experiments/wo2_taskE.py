"""Generality of compensation with one conductance held fixed.

The conductance of the mutated channel is held fixed (that channel is assumed
not to be pharmacologically adjustable) and the remaining conductances are
optimized to drive the cell's voltage trace back toward a reference target.

Four questions:
  E1  Does it work whichever conductance is frozen?
  E2  How large a variant offset can be compensated?
  E3  Does it hold under mean squared error, mean absolute error and smooth L1?
  E4  Does it work when the target is an arbitrary researcher-specified trace
      rather than the wild-type trace?

Every run: integration step 10/1500 ms, 1500 steps, injected current 8, Adam,
uniform non-negativity clamp. Learning rates are scaled to each parameter's own
magnitude (see LR below), as mimic_cell_activity/HH.py also does with its
per-parameter learning rates.

Usage:  python3 wo2_taskE.py [all|e1|e2|e3|e4]
Writes  taskE_<which>.npz
"""
import sys
import numpy as np
import torch

torch.set_num_threads(1)
import wo2_lib as L

I = 8.0
ITERS = 400
CLAMP = 1e-6
OPT = "adam"

# learning rate scaled to each conductance's own magnitude (Na, K: wild-type value / 240)
LR = {"Na": 0.5, "K": 0.15, "l": 0.001}


def lrs_for(frozen):
    d = {k: v for k, v in LR.items()}
    d[frozen] = 0.0
    return d


def wt_loss_reference(loss_name="mse"):
    """Loss of the untreated variant and of the wild type against the target."""
    tgt = L.simulate(*L.WT, I=I)
    fn = L.LOSSES[loss_name]
    out = {}
    for nm, p in [("variant", L.VARIANT), ("wildtype", L.WT)]:
        out[nm] = float(fn(L.simulate(*p, I=I), tgt)[0])
    return out


def pct_closed(final_loss, start_loss):
    """How much of the gap between the untreated variant and a perfect match was
    closed, in percent. 100 percent means the target was matched exactly."""
    if not np.isfinite(final_loss):
        return np.nan
    return (1.0 - final_loss / start_loss) * 100.0


def e1_freeze_each():
    print("\n" + "=" * 78)
    print("E1  Vary which conductance is frozen at a variant value")
    print("=" * 78, flush=True)
    # Every case starts from the same point (Na 140, K 20, leak 0.30). The frozen
    # conductance plays the role of the variant channel; the other two are fitted.
    cases = [
        ("Na", (140.0, 20.0, 0.30), "sodium variant  Na 120->140"),
        ("K",  (140.0, 20.0, 0.30), "potassium variant K 36->20"),
        ("l",  (140.0, 20.0, 0.30), "leak variant     leak 0.03->0.30"),
    ]
    cfgs = [dict(frozen=f, start=s, optimizer=OPT, lrs=lrs_for(f), label=note)
            for f, s, note in cases]
    r = L.fit_grid(cfgs, loss_name="mse", iters=ITERS, I=I, clamp_min=CLAMP,
                   reset_each_iter=False)
    ref = wt_loss_reference("mse")
    print(f"{'frozen':7s} {'description':50s} {'start loss':>11} {'end loss':>10} "
          f"{'gap closed':>11}  final (Na, K, leak)")
    for b, (f, s, note) in enumerate(cases):
        L0, L1 = r["loss"][0, b], r["loss"][-1, b]
        print(f"{f:7s} {note:50s} {L0:11.4g} {L1:10.4g} "
              f"{pct_closed(L1, L0):10.2f}%  "
              f"({r['Na'][-1,b]:.3f}, {r['K'][-1,b]:.3f}, {r['l'][-1,b]:.5f})",
              flush=True)
    print(f"\n  reference: untreated variant loss {ref['variant']:.4g}, "
          f"wild type against itself {ref['wildtype']:.4g}")
    return r, cases


def e2_offset_envelope():
    print("\n" + "=" * 78)
    print("E2  How large a sodium offset can potassium and leak compensate?")
    print("=" * 78, flush=True)
    gnas = np.arange(100.0, 205.0, 5.0)
    cfgs = [dict(frozen="Na", start=(g, 20.0, 0.30), optimizer=OPT,
                 lrs=lrs_for("Na"), label=f"gNa={g:g}") for g in gnas]
    r = L.fit_grid(cfgs, loss_name="mse", iters=ITERS, I=I, clamp_min=CLAMP,
                   reset_each_iter=False)
    print(f"{'g_Na frozen':>12} {'% of WT':>9} {'start loss':>11} {'end loss':>10} "
          f"{'gap closed':>11} {'final K':>9} {'final leak':>11}")
    for b, g in enumerate(gnas):
        L0, L1 = r["loss"][0, b], r["loss"][-1, b]
        print(f"{g:12g} {g/120*100:8.1f}% {L0:11.4g} {L1:10.4g} "
              f"{pct_closed(L1, L0):10.2f}% {r['K'][-1,b]:9.4f} "
              f"{r['l'][-1,b]:11.5f}", flush=True)
    return r, gnas


def e3_losses():
    print("\n" + "=" * 78)
    print("E3  Does compensation hold under different loss functions?")
    print("=" * 78, flush=True)
    out = {}
    print(f"{'loss':12s} {'start':>12} {'end':>12} {'gap closed':>11} "
          f"{'final K':>9} {'final leak':>11}")
    for loss_name in ("mse", "mae", "smooth_l1"):
        cfgs = [dict(frozen="Na", start=L.VARIANT, optimizer=OPT,
                     lrs=lrs_for("Na"), label=loss_name)]
        r = L.fit_grid(cfgs, loss_name=loss_name, iters=ITERS, I=I,
                       clamp_min=CLAMP, reset_each_iter=False)
        L0, L1 = r["loss"][0, 0], r["loss"][-1, 0]
        print(f"{loss_name:12s} {L0:12.5g} {L1:12.5g} {pct_closed(L1, L0):10.2f}% "
              f"{r['K'][-1,0]:9.4f} {r['l'][-1,0]:11.5f}", flush=True)
        out[loss_name] = r
    return out


def e4_arbitrary_target():
    print("\n" + "=" * 78)
    print("E4  Target is an arbitrary researcher-specified conductance set,")
    print("    not the wild type")
    print("=" * 78, flush=True)
    targets = [(90.0, 50.0, 0.20), (200.0, 25.0, 0.05), (120.0, 60.0, 0.40)]
    print(f"{'target (Na,K,leak)':26s} {'start loss':>11} {'end loss':>10} "
          f"{'gap closed':>11} {'final K':>9} {'final leak':>11}")
    outs = []
    for t in targets:
        with torch.no_grad():
            tgt = L.simulate(*t, I=I)
        cfgs = [dict(frozen="Na", start=L.VARIANT, optimizer=OPT,
                     lrs=lrs_for("Na"), label=str(t))]
        r = L.fit_grid(cfgs, target_trace=tgt, loss_name="mse", iters=ITERS,
                       I=I, clamp_min=CLAMP, reset_each_iter=False)
        L0, L1 = r["loss"][0, 0], r["loss"][-1, 0]
        print(f"{str(t):26s} {L0:11.4g} {L1:10.4g} {pct_closed(L1, L0):10.2f}% "
              f"{r['K'][-1,0]:9.4f} {r['l'][-1,0]:11.5f}", flush=True)
        outs.append(r)
    return outs, targets


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    store = {}
    if which in ("all", "e1"):
        r, cases = e1_freeze_each()
        store["e1_K"], store["e1_l"], store["e1_Na"] = r["K"], r["l"], r["Na"]
        store["e1_loss"] = r["loss"]
    if which in ("all", "e2"):
        r, gnas = e2_offset_envelope()
        store["e2_K"], store["e2_l"], store["e2_loss"] = r["K"], r["l"], r["loss"]
        store["e2_gna"] = gnas
    if which in ("all", "e3"):
        o = e3_losses()
        for k, r in o.items():
            store[f"e3_{k}_loss"] = r["loss"]
            store[f"e3_{k}_K"] = r["K"]
            store[f"e3_{k}_l"] = r["l"]
    if which in ("all", "e4"):
        outs, targets = e4_arbitrary_target()
        for i, r in enumerate(outs):
            store[f"e4_{i}_loss"] = r["loss"]
            store[f"e4_{i}_K"] = r["K"]
            store[f"e4_{i}_l"] = r["l"]
        store["e4_targets"] = np.array(targets)
    np.savez_compressed(f"taskE_{which}.npz", **store)


if __name__ == "__main__":
    main()
