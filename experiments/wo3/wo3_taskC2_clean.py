"""GPU throughput with setup outside the timed region.

The sweep in wo3_taskC2.py calls a run function per measurement, and that
function re-initialises the state inside the timed region; the initialisation
draws the random conductances with numpy on the host -- about 9 million
double-precision values at the largest batch. For the unfused implementations
that cost is negligible against a several-hundred-second run, but for the fused
ones it is not, so the fused throughput measured that way is understated. This
script builds the state once, warms up, and times only the integration.

Unit: 35 current levels x 30,000 integration steps per candidate parameter set,
as in wo2_taskD_benchmark.py.
"""
import time
import torch
import torch._dynamo
import wo3_gpu_lib as G
import wo3_taskC2 as C

# every batch size is compiled separately (dynamic=False), so raise the limits
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 512

DEV = "cuda"
K = 25
FULL = G.STEPS_FULL
SETS = [64, 1024, 4096, 16384, 65536, 262144, 1048576]


def time_baseline(B, warm, meas):
    hh = G.baseline_hh(B, DEV)
    with torch.no_grad():
        for _ in range(warm):
            hh.forward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(meas):
            hh.forward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / meas


def time_graph(B, warm, meas):
    g, statics, keep_alive = C.make_cudagraph(B, K)
    for _ in range(max(1, warm // K)):
        g.replay()
    torch.cuda.synchronize()
    reps = max(1, meas // K)
    t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    assert keep_alive is not None
    return dt / (reps * K)


def time_compiled(B, warm, meas):
    f = torch.compile(
        lambda v, m, h, n, a, b, c: G.chunk(v, m, h, n, a, b, c,
                                            G.DT, G.I_CONST, K), dynamic=False)
    v, m, h, n, a, b, c = G.init_state(B, DEV)
    with torch.no_grad():
        for _ in range(max(1, warm // K)):
            v, m, h, n = f(v, m, h, n, a, b, c)
    torch.cuda.synchronize()
    reps = max(1, meas // K)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            v, m, h, n = f(v, m, h, n, a, b, c)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / (reps * K)


VARIANTS = [
    ("baseline: original HH.forward, eager Python loop",
     time_baseline, 100, 600),
    (f"CUDA graph capture, {K} integration steps per replay", time_graph, 200, 2000),
    (f"torch.compile (Inductor), {K} integration steps per call",
     time_compiled, 500, 20000),
]

if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  {p.name}  {p.total_memory/2**30:.2f} GiB")
    print("setup and warm-up are OUTSIDE the timed region\n")
    for label, fn, warm, meas in VARIANTS:
        print(f"--- {label} ---", flush=True)
        print(f"{'sets':>9} {'us per step':>13} {'s per set-unit':>15} "
              f"{'sets/s':>11} {'peak GiB':>10}", flush=True)
        for nsets in SETS:
            B = nsets * G.CUR_LEVELS
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                per_step = fn(B, warm, meas)
                full = per_step * FULL
                print(f"{nsets:>9} {per_step*1e6:13.2f} {full:15.3f} "
                      f"{nsets/full:11.1f} "
                      f"{torch.cuda.max_memory_allocated()/2**30:10.2f}", flush=True)
            except Exception as e:
                print(f"{nsets:>9}  FAILED: {str(e)[:70]}", flush=True)
            torch.cuda.empty_cache()
        print(flush=True)
    print("DONE", flush=True)
