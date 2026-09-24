"""Implement and measure fused variants of the HH integration loop on the GPU.

Variants
  baseline   HH.forward from mimic_cell_activity/HH.py, called in a Python loop
  functional the same arithmetic as a pure function, eager (control, to show the
             rewrite by itself changes nothing)
  cudagraph  a CUDA graph capturing K integration steps, replayed
  compile    torch.compile over a K-step chunk (Inductor fuses the pointwise ops)
  compile+cg torch.compile with mode="reduce-overhead" (fusion + CUDA graphs)

Unit: one candidate parameter set = 35 injected-current levels x 30,000 steps
(0.01 ms step, 300 ms), the same unit as wo2_taskD_benchmark.py.

Usage:  python3 wo3_taskC2.py [--part all|sweep|equiv]
"""
import argparse, json, time
import torch
import wo3_gpu_lib as G

DEV = "cuda"
FULL = G.STEPS_FULL


# ------------------------------------------------------------------ runners

def run_baseline(B, n_steps):
    hh = G.baseline_hh(B, DEV)
    with torch.no_grad():
        for _ in range(n_steps):
            hh.forward()
    return hh.v


def run_functional(B, n_steps, k=1):
    v, m, h, n, gNa, gK, gl = G.init_state(B, DEV)
    with torch.no_grad():
        for _ in range(n_steps // k):
            v, m, h, n = G.chunk(v, m, h, n, gNa, gK, gl, G.DT, G.I_CONST, k)
    return v


def make_cudagraph(B, k, seed=0):
    """Capture k integration steps as one graph.

    Returns (graph, static state tensors, everything else that must stay alive).

    The third return value matters. A captured CUDA graph records the
    *addresses* of every tensor it reads. The conductance tensors gNa, gK and gl
    are read on every replay, so if their last Python reference goes away when
    this function returns, the caching allocator frees those blocks, later
    reuses them, and the graph then replays against whatever now occupies that
    memory (in practice producing not-a-number for every trajectory). They are
    returned here so the caller keeps them alive.
    """
    v, m, h, n, gNa, gK, gl = G.init_state(B, DEV, seed=seed)
    sv, sm, sh, sn = v.clone(), m.clone(), h.clone(), n.clone()

    # warm up on a side stream, as the CUDA graph docs require
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            a, b, c, d = G.chunk(sv, sm, sh, sn, gNa, gK, gl, G.DT, G.I_CONST, k)
    torch.cuda.current_stream().wait_stream(s)

    sv.copy_(v); sm.copy_(m); sh.copy_(h); sn.copy_(n)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        a, b, c, d = G.chunk(sv, sm, sh, sn, gNa, gK, gl, G.DT, G.I_CONST, k)
        sv.copy_(a); sm.copy_(b); sh.copy_(c); sn.copy_(d)

    # reset the static buffers to the true initial state after capture
    sv.copy_(v); sm.copy_(m); sh.copy_(h); sn.copy_(n)
    return g, (sv, sm, sh, sn), (v, m, h, n, gNa, gK, gl)


def run_cudagraph(B, n_steps, k, seed=0):
    g, (sv, sm, sh, sn), keep_alive = make_cudagraph(B, k, seed)
    for _ in range(n_steps // k):
        g.replay()
    assert keep_alive is not None       # must outlive every replay
    return sv


_COMPILED = {}


def get_compiled(k, mode):
    key = (k, mode)
    if key not in _COMPILED:
        fn = (lambda v, m, h, n, a, b, c: G.chunk(v, m, h, n, a, b, c,
                                                  G.DT, G.I_CONST, k))
        _COMPILED[key] = torch.compile(fn, mode=mode, dynamic=False)
    return _COMPILED[key]


def run_compiled(B, n_steps, k, mode=None, seed=0):
    f = get_compiled(k, mode)
    v, m, h, n, gNa, gK, gl = G.init_state(B, DEV, seed=seed)
    with torch.no_grad():
        for _ in range(n_steps // k):
            v, m, h, n = f(v, m, h, n, gNa, gK, gl)
    return v


# ------------------------------------------------------------------ timing

def timed(fn, warm_steps, meas_steps, B):
    """Warm up, then time meas_steps and extrapolate to 30,000 steps."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(B, warm_steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(B, meas_steps)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    full = dt * FULL / meas_steps
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    return full, mem


def sweep(name, fn, sets_list, warm, meas):
    print(f"\n--- {name} ---", flush=True)
    print(f"{'sets':>9} {'s per set-unit':>15} {'sets/s':>11} {'peak GiB':>10}", flush=True)
    out = {}
    for n in sets_list:
        B = n * G.CUR_LEVELS
        try:
            full, mem = timed(fn, warm, meas, B)
            rate = n / full
            out[n] = rate
            print(f"{n:>9} {full:15.2f} {rate:11.1f} {mem:10.2f}", flush=True)
        except (RuntimeError, torch.OutOfMemoryError) as e:
            print(f"{n:>9}  FAILED: {str(e)[:70]}", flush=True)
            out[n] = None
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return out


# ------------------------------------------------------------------ equivalence

def equivalence(n_sets=64, n_steps=FULL, stride=1000, k=100):
    """Max absolute deviation of the membrane-voltage trajectory between the
    original HH.forward and each fused variant, over the full 300 ms."""
    B = n_sets * G.CUR_LEVELS
    print(f"\nnumerical equivalence, batch {n_sets} sets ({B} elements), "
          f"{n_steps} steps, sampled every {stride}", flush=True)

    hh = G.baseline_hh(B, DEV)
    ref = []
    with torch.no_grad():
        for i in range(n_steps):
            hh.forward()
            if (i + 1) % stride == 0:
                ref.append(hh.v.clone())
    ref = torch.stack(ref)
    ref_scale = float(ref.abs().max())

    results = {}

    # functional, eager
    v, m, h, n, gNa, gK, gl = G.init_state(B, DEV)
    got = []
    with torch.no_grad():
        for i in range(n_steps):
            v, m, h, n = G.step(v, m, h, n, gNa, gK, gl, G.DT, G.I_CONST)
            if (i + 1) % stride == 0:
                got.append(v.clone())
    results["functional (eager)"] = float((torch.stack(got) - ref).abs().max())

    # cuda graph
    g, (sv, sm, sh, sn), _ = make_cudagraph(B, k)
    got = []
    for i in range(n_steps // k):
        g.replay()
        if ((i + 1) * k) % stride == 0:
            got.append(sv.clone())
    results[f"cudagraph (k={k})"] = float((torch.stack(got) - ref).abs().max())

    # torch.compile
    for mode in (None, "reduce-overhead"):
        f = get_compiled(k, mode)
        v, m, h, n, gNa, gK, gl = G.init_state(B, DEV)
        got = []
        with torch.no_grad():
            for i in range(n_steps // k):
                v, m, h, n = f(v, m, h, n, gNa, gK, gl)
                if ((i + 1) * k) % stride == 0:
                    got.append(v.clone())
        results[f"torch.compile (k={k}, mode={mode})"] = \
            float((torch.stack(got) - ref).abs().max())

    print(f"reference trajectory peak |v| = {ref_scale:.4f} mV")
    for kk, vv in results.items():
        print(f"  {kk:<44s} max |deviation| = {vv:.3e} mV "
              f"({100*vv/ref_scale:.2e} % of peak)")
    return results, ref_scale


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()

    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"total GPU memory {tot:.2f} GiB", flush=True)

    SETS = [64, 1024, 4096, 16384, 65536, 262144]

    if a.part in ("all", "sweep"):
        print("\n" + "=" * 78)
        print("C2  throughput, one set = 35 levels x 30,000 steps")
        print("=" * 78)

        sweep("baseline: repository HH.forward, eager Python loop",
              lambda B, s: run_baseline(B, s), SETS, 100, 600)

        sweep("control: functional rewrite, eager Python loop",
              lambda B, s: run_functional(B, s, 1), SETS, 100, 600)

        for k in (25, 100):
            sweep(f"cudagraph, {k} steps per replay",
                  lambda B, s, k=k: run_cudagraph(B, s, k), SETS, 200, 1000)

        for k in (25, 100):
            sweep(f"torch.compile (default), {k} steps per call",
                  lambda B, s, k=k: run_compiled(B, s, k, None), SETS, 200, 1000)

        for k in (25, 100):
            sweep(f"torch.compile (reduce-overhead), {k} steps per call",
                  lambda B, s, k=k: run_compiled(B, s, k, "reduce-overhead"),
                  SETS, 200, 1000)

    if a.part in ("all", "equiv"):
        print("\n" + "=" * 78)
        print("C2  numerical equivalence")
        print("=" * 78)
        equivalence()
