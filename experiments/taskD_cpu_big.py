"""CPU batch scaling of the differentiable model, one core. Same definition of a
candidate parameter set as wo2_taskD_benchmark.py; a shorter step count is timed
and scaled to the full 30,000 steps (the per-step cost is independent of the
step count)."""
import time, numpy as np, torch, wo2_lib as L
torch.set_num_threads(1)
STEPS, FULL, CUR = 2000, 30000, 35
print(f"{'sets':>7} {'elems':>9} {'s/300ms-run':>12} {'sets/s (1 core)':>16}")
for n in [1, 4, 16, 64, 256, 1024]:
    B = n*CUR
    gNa=np.random.uniform(60,260,B); gK=np.random.uniform(26,49,B); gl=np.random.uniform(0.1,0.5,B)
    hh=L.build(gNa,gK,gl,0.01,8.0,B=B)
    with torch.no_grad():
        for _ in range(50): hh.forward()
    hh=L.build(gNa,gK,gl,0.01,8.0,B=B)
    t0=time.time()
    with torch.no_grad():
        for _ in range(STEPS): hh.forward()
    dt=(time.time()-t0)*FULL/STEPS
    print(f"{n:>7} {B:>9} {dt:>12.1f} {n/dt:>16.4f}", flush=True)
