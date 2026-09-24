"""Throughput benchmark of the differentiable HH model.

The unit is **candidate parameter sets evaluated per second**, where one
"candidate parameter set" means exactly what the NEURON pipeline means by it:
one triple (g_leak, g_K, g_Na) simulated across all 35 injected-current levels
(20 pA to 360 pA in 10 pA steps) for 300 ms each, at an integration step of
0.01 ms -- i.e. 35 simulations of 30,000 steps.

That is the protocol in Neuron_Test/R859C/neuron.hoc
(ilow=20, istep=10, nsteps=35; dt=0.01, tstop=300).

The differentiable model here is the HH class of mimic_cell_activity/HH.py,
loaded through wo2_lib. Both models are single-compartment point neurons, so the
comparison is not confounded by compartment count.

Usage: python3 wo2_taskD_benchmark.py [cpu|cuda] [n_steps_override]
"""
import sys
import time
import numpy as np
import torch

import wo2_lib as L

NEURON_STEPS = 30000        # 300 ms at dt = 0.01 ms
NEURON_CURRENTS = 35        # 20 pA .. 360 pA in 10 pA steps
DT_NEURON = 0.01


def bench(device, n_sets, n_steps=NEURON_STEPS, warmup_steps=200):
    """Time one full candidate-parameter-set evaluation batch."""
    B = n_sets * NEURON_CURRENTS
    gNa = np.random.uniform(60, 260, B)
    gK = np.random.uniform(26, 49, B)
    gl = np.random.uniform(0.1, 0.5, B)

    hh = L.build(gNa, gK, gl, DT_NEURON, 8.0, B=B, device=device)
    # warm up (CUDA context, kernel autotune)
    with torch.no_grad():
        for _ in range(warmup_steps):
            hh.forward()
    if device == "cuda":
        torch.cuda.synchronize()

    hh = L.build(gNa, gK, gl, DT_NEURON, 8.0, B=B, device=device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            hh.forward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, n_sets / dt


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else NEURON_STEPS

    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device available")
        return

    print(f"device: {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}, {p.total_memory/2**30:.1f} GiB, "
              f"{p.multi_processor_count} multiprocessors")
    else:
        torch.set_num_threads(1)
        print(f"  torch threads: {torch.get_num_threads()} (one core)")
    print(f"one candidate parameter set = {NEURON_CURRENTS} currents "
          f"x {steps} steps at dt={DT_NEURON} ms (= 300 ms each)")
    print()
    print(f"{'param sets':>11} {'batch elems':>12} {'seconds':>10} "
          f"{'sets/second':>12} {'sim-ms/second':>14}")

    sizes = [1, 4, 16, 64, 256] if device == "cuda" else [1, 4, 16]
    for n in sizes:
        try:
            dt, rate = bench(device, n, steps)
        except RuntimeError as e:
            print(f"{n:>11} {n*NEURON_CURRENTS:>12}  failed: {type(e).__name__}: "
                  f"{str(e)[:60]}")
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        sim_ms = n * NEURON_CURRENTS * steps * DT_NEURON / dt
        print(f"{n:>11} {n*NEURON_CURRENTS:>12} {dt:>10.2f} {rate:>12.4f} "
              f"{sim_ms:>14.0f}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
