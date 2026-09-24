"""What is the FUSED implementation bound by?

The unfused loop is launch-bound below ~16,000 parameter sets and
memory-bandwidth bound above (wo3_taskC1.py, wo3_taskC1_bandwidth.py). After
fusion the picture changes, and this script measures the new limit rather than
guessing it. Measures, for the torch.compile variant:

  * kernels per integration step (should fall from 76 to a small number)
  * share of wall clock inside kernels
  * device utilization
  * the memory traffic it still demands, against the card's measured copy
    bandwidth -- if that ratio is now small, the fused kernel is arithmetic
    bound, not bandwidth bound
"""
import subprocess, threading, time
import torch
import torch._dynamo
from torch.profiler import profile, ProfilerActivity
import wo3_gpu_lib as G

torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 512

DEV = "cuda"
K = 25


def build(k=K):
    return torch.compile(
        lambda v, m, h, n, a, b, c: G.chunk(v, m, h, n, a, b, c,
                                            G.DT, G.I_CONST, k),
        dynamic=False)


class Util(threading.Thread):
    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period, self.s, self.stop = period, [], False

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.s.append(int(o))
            except Exception:
                pass
            time.sleep(self.period)


def measure(n_sets, calls=200):
    B = n_sets * G.CUR_LEVELS
    f = build()
    v, m, h, n, a, b, c = G.init_state(B, DEV)
    with torch.no_grad():
        for _ in range(20):
            v, m, h, n = f(v, m, h, n, a, b, c)
    torch.cuda.synchronize()

    # wall clock
    u = Util(); u.start()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(calls):
            v, m, h, n = f(v, m, h, n, a, b, c)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    u.stop = True; u.join(timeout=2)

    # kernel census
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(20):
                v, m, h, n = f(v, m, h, n, a, b, c)
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages()
           if e.device_type == torch.autograd.DeviceType.CUDA
           and e.self_device_time_total > 0]
    n_kern = sum(e.count for e in evs) / 20            # per call of k steps
    kern_s = sum(e.self_device_time_total for e in evs) / 20 / 1e6

    wall_per_call = wall / calls
    util = sum(u.s) / len(u.s) if u.s else float("nan")
    return dict(sets=n_sets, wall_step=wall_per_call / K,
                kern_step=kern_s / K, kernels_step=n_kern / K,
                util=util, elems=B)


if __name__ == "__main__":
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"torch.compile, {K} integration steps per call\n")
    print(f"{'sets':>9} {'kernels/step':>13} {'wall us/step':>13} "
          f"{'kernel us/step':>15} {'kernel share':>13} {'util %':>8} "
          f"{'GB/s of traffic':>16}")
    for nsets in (16384, 65536, 262144):
        try:
            r = measure(nsets)
            # after fusion each call reads 7 arrays and writes 4, once per K steps
            bytes_per_call = 11 * r["elems"] * 4
            gbs = bytes_per_call / (r["wall_step"] * K) / 1e9
            print(f"{r['sets']:>9} {r['kernels_step']:13.2f} "
                  f"{r['wall_step']*1e6:13.2f} {r['kern_step']*1e6:15.2f} "
                  f"{100*r['kern_step']/r['wall_step']:12.1f}% {r['util']:8.1f} "
                  f"{gbs:16.1f}")
        except Exception as e:
            print(f"{nsets:>9}  FAILED {str(e)[:60]}")
        torch.cuda.empty_cache()
    print("\nDONE")
