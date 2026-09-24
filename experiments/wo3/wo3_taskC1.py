"""Characterize the overhead of the unfused integration loop on the GPU.

1. kernel launches per integration step, and per candidate parameter set
2. time in kernels vs. wall clock (the difference is launch + Python dispatch)
3. GPU utilization at saturation

All measurements use the repository's own HH.forward as the baseline, at the
standard unit of one candidate parameter set (35 current levels, 300 ms,
0.01 ms step).  The output of this script is out_taskC1.log.
"""
import subprocess, threading, time
import torch
from torch.profiler import profile, ProfilerActivity
import wo3_gpu_lib as G

DEV = "cuda"
torch.backends.cudnn.benchmark = False


def kernel_census(n_sets, n_steps=20):
    """Profile n_steps calls of the repository's HH.forward and count CUDA kernels."""
    B = n_sets * G.CUR_LEVELS
    hh = G.baseline_hh(B, DEV)
    with torch.no_grad():
        for _ in range(50):
            hh.forward()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        with torch.no_grad():
            for _ in range(n_steps):
                hh.forward()
        torch.cuda.synchronize()

    evs = prof.key_averages()
    n_kernels = 0
    cuda_time_us = 0.0
    per_kernel = {}
    for e in evs:
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0:
            n_kernels += e.count
            cuda_time_us += e.self_device_time_total
            per_kernel[e.key] = (e.count, e.self_device_time_total)
    # host-side: total CPU time of the top-level aten calls
    cpu_time_us = sum(e.self_cpu_time_total for e in evs)
    return n_kernels / n_steps, cuda_time_us / n_steps, cpu_time_us / n_steps, per_kernel


def wall_vs_kernel(n_sets, n_steps=2000):
    """Wall-clock time for n_steps vs. the summed CUDA kernel time over the same."""
    B = n_sets * G.CUR_LEVELS
    hh = G.baseline_hh(B, DEV)
    with torch.no_grad():
        for _ in range(100):
            hh.forward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            hh.forward()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(200):
                hh.forward()
        torch.cuda.synchronize()
    kern_us = sum(e.self_device_time_total for e in prof.key_averages()
                  if e.device_type == torch.autograd.DeviceType.CUDA)
    kern_per_step_s = kern_us / 200 / 1e6
    return wall / n_steps, kern_per_step_s


class UtilSampler(threading.Thread):
    """Sample GPU utilization with nvidia-smi while a benchmark runs."""

    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag = period, [], False

    def run(self):
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                g, m = [int(x) for x in out.split(",")]
                self.samples.append((g, m))
            except Exception:
                pass
            time.sleep(self.period)


def util_at(n_sets, seconds=8.0):
    B = n_sets * G.CUR_LEVELS
    hh = G.baseline_hh(B, DEV)
    with torch.no_grad():
        for _ in range(100):
            hh.forward()
    torch.cuda.synchronize()
    s = UtilSampler()
    s.start()
    t0 = time.perf_counter()
    steps = 0
    with torch.no_grad():
        while time.perf_counter() - t0 < seconds:
            for _ in range(200):
                hh.forward()
            steps += 200
    torch.cuda.synchronize()
    s.stop_flag = True
    s.join(timeout=2)
    g = [x[0] for x in s.samples]
    return (sum(g) / len(g) if g else float("nan"),
            max(g) if g else float("nan"), len(g))


if __name__ == "__main__":
    print("=" * 78)
    print("C1.1  kernel launches per integration step (repository HH.forward)")
    print("=" * 78)
    for n in (1024, 16384):
        k, cuda_us, cpu_us, per = kernel_census(n)
        print(f"\nbatch {n:>6} candidate parameter sets ({n*G.CUR_LEVELS:>8} elements)")
        print(f"  CUDA kernels launched per integration step : {k:.1f}")
        print(f"  kernels for one candidate set (30,000 steps): {k*G.STEPS_FULL:,.0f}")
        print(f"  summed CUDA kernel time per step           : {cuda_us:9.1f} us")
        print(f"  summed host (CPU) time per step            : {cpu_us:9.1f} us")
        if n == 16384:
            print("  per-kernel breakdown (count per 20 steps, total us per 20 steps):")
            for kk, (c, t) in sorted(per.items(), key=lambda x: -x[1][1]):
                print(f"    {kk:<40s} {c:>5d}  {t:10.1f}")

    print()
    print("=" * 78)
    print("C1.2  wall clock vs. kernel time  (the gap is launch + Python dispatch)")
    print("=" * 78)
    print(f"{'sets':>8} {'wall/step us':>13} {'kernel/step us':>15} "
          f"{'kernel share':>13} {'overhead/step us':>17}")
    for n in (64, 1024, 4096, 16384, 65536, 262144):
        try:
            w, k = wall_vs_kernel(n, n_steps=1000 if n < 65536 else 300)
            print(f"{n:>8} {w*1e6:13.1f} {k*1e6:15.1f} "
                  f"{100*k/w:12.1f}% {(w-k)*1e6:17.1f}")
        except RuntimeError as e:
            print(f"{n:>8}  FAILED {str(e)[:60]}")
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("C1.3  GPU utilization (nvidia-smi, 10 Hz sampling)")
    print("=" * 78)
    print(f"{'sets':>8} {'mean util %':>12} {'max util %':>12} {'samples':>9}")
    for n in (1024, 16384, 65536, 262144):
        try:
            mu, mx, ns = util_at(n)
            print(f"{n:>8} {mu:12.1f} {mx:12.0f} {ns:9d}")
        except RuntimeError as e:
            print(f"{n:>8}  FAILED {str(e)[:60]}")
        torch.cuda.empty_cache()
