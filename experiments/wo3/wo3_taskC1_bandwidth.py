"""Is the saturated regime of the unfused loop memory-bandwidth bound?

Two numbers:
  1. the card's achievable memory bandwidth, measured with a plain large copy
  2. the bandwidth the unfused integration loop actually demands, computed from
     the kernel census in wo3_taskC1.py (76 kernels per step) times the array
     traffic each kernel does

If (2) is close to (1) then the saturated plateau is bandwidth bound, not
launch bound, and fusing the step is the remedy because it removes the traffic
rather than because it removes launches.

Its output, followed by that of wo3_taskC2_profile.py, is out_taskC_profile.log.
"""
import time
import torch
import wo3_gpu_lib as G

DEV = "cuda"

# arity census taken from the profiler output of wo3_taskC1.py at batch 16,384
# (counts are per 20 integration steps; each entry is (count, arrays touched))
CENSUS = [
    (240, 3),   # vectorized binary multiply
    (180, 3),   # vectorized binary add
    (180, 2),   # add-on-self
    (100, 3),   # non-vectorized binary multiply
    (80, 3),    # non-vectorized binary add
    (200, 2),   # scalar-times-tensor multiply
    (100, 2),   # add-on-other
    (80, 2),    # device-to-device copy
    (120, 2),   # exp
    (40, 3),    # binary divide
    (60, 2),    # tensor-times-scalar multiply
    (60, 2),    # negate
    (20, 2),    # power
    (20, 2),    # power
    (40, 2),    # reciprocal
]
STEPS_IN_CENSUS = 20


def measured_bandwidth(n_bytes=512 * 2 ** 20, reps=50):
    n = n_bytes // 4
    a = torch.empty(n, dtype=torch.float, device=DEV)
    b = torch.empty(n, dtype=torch.float, device=DEV)
    a.fill_(1.0)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # a copy reads n_bytes and writes n_bytes
    return 2 * n_bytes * reps / dt / 1e9


def demanded(n_sets, seconds_per_step):
    n_elem = n_sets * G.CUR_LEVELS
    bytes_per_array = n_elem * 4
    arrays_per_step = sum(c * a for c, a in CENSUS) / STEPS_IN_CENSUS
    bytes_per_step = arrays_per_step * bytes_per_array
    return arrays_per_step, bytes_per_step, bytes_per_step / seconds_per_step / 1e9


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}, {p.total_memory/2**30:.2f} GiB, {p.multi_processor_count} SMs")
    bw = measured_bandwidth()
    print(f"\nmeasured device-to-device copy bandwidth: {bw:.1f} GB/s")
    print("  (this is the practical ceiling for a purely memory-bound kernel)")

    print("\nbandwidth the unfused integration loop demands, using the wall-clock")
    print("and kernel times measured in wo3_taskC1.py:")
    print(f"{'sets':>9} {'arrays/step':>12} {'GB per step':>12} "
          f"{'kernel us/step':>15} {'GB/s demanded':>15} {'% of measured':>14}")
    # (sets, kernel seconds per step) taken from wo3_taskC1.py section C1.2
    for n, k_us in ((16384, 744.4), (65536, 4094.3), (262144, 16078.1)):
        arrays, byts, gbs = demanded(n, k_us / 1e6)
        print(f"{n:>9} {arrays:12.1f} {byts/1e9:12.3f} {k_us:15.1f} "
              f"{gbs:15.1f} {100*gbs/bw:13.1f}%")
