"""Why the fused integration step does not scale with a GPU's rated arithmetic throughput.

The RTX 4070 Ti provides about 5.4 times the RTX 2070's rated single-precision
throughput, but the measured speedup of the fused implementation is only about
1.8 times. This script measures the two quantities that could explain the gap,
on whichever card it runs on:

  1. fused-multiply-add throughput  -- the operation the "TFLOPS" rating describes.
  2. exponential throughput         -- evaluated on the special-function units, which
                                       are provisioned per multiprocessor and do NOT
                                       scale with the fused-multiply-add rate.

The integration step evaluates six exponentials (wo5_lib.alphaN, betaN, alphaM, betaM,
alphaH, betaH) against roughly forty ordinary arithmetic operations, so if the
exponential rate is the binding constraint the observed speedup should track
measurement 2 rather than measurement 1.

Both chains are register-resident: the working set is one array, read once and written
once per call, with `k` operations applied in between, so memory traffic per operation
falls as 1/k and the measurement isolates arithmetic.
"""
import time
import torch

DEV = "cuda"
N = 1 << 21          # 2.10 million elements
K = 64               # operations per element per call (small enough to compile quickly)
REPS = 200


def _fma(x, a, b, k):
    for _ in range(k):
        x = x * a + b
    return x


def _exp(x, k):
    for _ in range(k):
        x = torch.exp(-(x * x) * 1e-6)      # bounded in (0, 1], so it cannot overflow
    return x


def timeit(fn, args, reps=REPS, warm=5):
    for _ in range(warm):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.multi_processor_count} multiprocessors  "
          f"L2 {p.L2_cache_size / 2**20:.0f} MiB  torch {torch.__version__}")

    x = torch.rand(N, device=DEV, dtype=torch.float32) * 0.5 + 0.25
    a = torch.tensor(1.0000001, device=DEV)
    b = torch.tensor(1e-7, device=DEV)

    fma = torch.compile(_fma, dynamic=False)
    exp = torch.compile(_exp, dynamic=False)

    with torch.no_grad():
        t_fma = timeit(fma, (x, a, b, K))
        t_exp = timeit(exp, (x, K))

    # one fused multiply-add is conventionally counted as two floating-point operations
    print(f"  fused multiply-add : {N * K * 2 / t_fma / 1e12:8.3f} Tflop/s   "
          f"({N * K / t_fma / 1e12:.3f} Tfma/s)")
    print(f"  exponential        : {N * K / t_exp / 1e12:8.3f} Texp/s")
    print(f"  ratio exp:fma      : {t_fma / t_exp:8.4f} exponentials per fused "
          f"multiply-add of throughput")
