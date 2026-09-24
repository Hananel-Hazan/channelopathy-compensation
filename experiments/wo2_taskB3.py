"""Re-run adam_ams at smaller learning rates.

In the invariance run (wo2_taskB2.py), adam_ams used the learning rate of 10
chosen by the 120-iteration pre-sweep; over 1500 iterations that setting is
unstable. Plain Adam, the same algorithm without the max-of-second-moment
modification, converged at a learning rate of 0.3. This script re-runs adam_ams
at Adam's learning rate and two larger ones (0.3, 1, 3) to separate the effect
of the learning-rate choice from that of the algorithm.

Writes taskB3.npz.
"""
import numpy as np, torch
torch.set_num_threads(1)
import wo2_lib as L

SCALES = [0.3, 1.0, 3.0]
RATIO, CLAMP, ITERS, I = 0.02, 1e-6, 1500, 8.0
cfgs = [dict(frozen="Na", start=L.VARIANT, optimizer="adam_ams",
             lrs={"Na": 0.0, "K": s, "l": s*RATIO}, label=f"adam_ams@{s:g}")
        for s in SCALES]
r = L.fit_grid(cfgs, loss_name="mse", iters=ITERS, I=I, clamp_min=CLAMP,
               reset_each_iter=False)
with torch.no_grad():
    tgt = L.simulate(*L.WT, I=I)
    L0 = float(L.loss_mse(L.simulate(*L.VARIANT, I=I), tgt)[0])
print(f"untreated variant loss {L0:.6g}\n")
print(f"{'configuration':18s} {'final K':>9} {'final leak':>11} {'best loss':>11} {'gap closed':>11}")
for b, s in enumerate(SCALES):
    loss = r["loss"][:, b]
    fin = np.where(np.isfinite(loss))[0]
    i = int(np.nanargmin(loss)) if len(fin) else -1
    print(f"adam_ams@{s:<9g} {r['K'][i,b]:9.4f} {r['l'][i,b]:11.5f} "
          f"{loss[i]:11.5g} {(1-loss[i]/L0)*100:10.4f}%", flush=True)
np.savez_compressed("taskB3.npz", K=r["K"], l=r["l"], loss=r["loss"],
                    scales=np.array(SCALES))
