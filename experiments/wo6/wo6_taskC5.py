"""Does gradient descent work once the removable singularities are removed?

The rate expressions alpha_n and alpha_m of the differentiable model
(mimic_cell_activity/HH.py, experiments/wo5/wo5_lib.py) take the form
x / (1 - exp(-x/10)), which is 0/0 at V = -50 mV and V = -35 mV respectively.
They return not-a-number there, and their derivatives are wrong by four orders
of magnitude in a neighbourhood of those voltages.  Every objective tested in
wo6_taskA.py differentiates through them.

This experiment replaces them, for the fit only, by numerically safe forms that
agree with them everywhere except within 1e-3 mV of the singular voltages,
where the limiting expansion is used instead:

    x / (1 - exp(-x/10))  ->  10 + x/2 + ...     as x -> 0

so alpha_n -> 0.1 (1 + (V+50)/20) and alpha_m -> 1.0 (1 + (V+35)/20).

Both branches are guarded: the quotient branch never sees an argument near
zero, because torch.where evaluates both branches and a not-a-number in the
unused branch would still poison the gradient through the multiplication by
zero.

The safe forms live only in this script; wo5_lib keeps the plain expressions,
which are the ones every reported sweep uses.  This first version has a defect:
the substitution also reaches the scoring integrator.  wo6_taskC5b.py repeats
the experiment with the substitution confined to the fit.

Needs a CUDA graphics card.  Writes wo6_taskC5.json in the current directory.
"""
import argparse, json, os, platform, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo5"))
sys.path.insert(0, os.path.join(REPO, "experiments", "wo6"))
import wo5_lib as W
import wo6_lib as O

DEV = "cuda"
EPS = 1e-3
LRS = {"Na": 0.5, "K": 0.15, "l": 0.001}


def _ratio_safe(u, scale, lim):
    """scale * u / (1 - exp(-u/10)), evaluated by its limit near u = 0."""
    near = u.abs() < EPS
    u_far = torch.where(near, torch.full_like(u, EPS), u)     # keep the quotient away from 0
    far = scale * u_far / (1 - torch.exp(-u_far / 10))
    return torch.where(near, lim * (1 + u / 20), far)


def alphaN_safe(v, shift_n=0.0):
    return _ratio_safe((v - shift_n) + 50, 0.01, 0.1)


def alphaM_safe(v, shift=0.0):
    return _ratio_safe((v - shift) + 35, 0.1, 1.0)


def step_safe(v, m, h, n, gNa_p, gK_p, gl_p, dt, I, shift=0.0, shift_beta=False, shift_n=0.0):
    """wo5_lib.step with only the two singular rate functions replaced."""
    gNa = gNa_p * h * m ** 3
    gK = gK_p * n ** 4
    INa = gNa * (v - W.E_NA); IK = gK * (v - W.E_K); Il = gl_p * (v - W.E_L)
    am, bm = alphaM_safe(v, shift), W.betaM(v, shift, shift_beta)
    m2 = m + dt * (am * (1 - m) - bm * m)
    n2 = n + dt * (alphaN_safe(v, shift_n) * (1 - n) - W.betaN(v, shift_n) * n)
    h2 = h + dt * (W.alphaH(v) * (1 - h) - W.betaH(v) * h)
    v2 = v + dt * ((1 / W.CM) * (I - (INa + IK + Il)))
    return v2, m2, h2, n2


ap = argparse.ArgumentParser()
ap.add_argument("--restarts", type=int, default=6)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--out", default="wo6_taskC5.json")
args = ap.parse_args()

print(f"machine {platform.node()} | checking the safe rate functions first")
for nm, fn, vs, lim in (("alpha_n", alphaN_safe, -50.0, 0.1), ("alpha_m", alphaM_safe, -35.0, 1.0)):
    for off in (0.0, 1e-7, 1e-4, 1.0):
        v = torch.tensor([vs + off], dtype=torch.float32, device=DEV, requires_grad=True)
        y = fn(v); g = torch.autograd.grad(y.sum(), v)[0]
        print(f"  {nm}(V={vs+off:+.7f}) = {y.item():>11.7g}   d/dV = {g.item():>11.6g}"
              f"   (limit {lim})")
print()

# monkey-patch only the module-level step used by the trajectory integrator in wo6_lib
O.W.step = step_safe
O._blocks.clear()

FITC = np.array([0.0, 3.0, 6.0, 7.0, 8.5, 11.0, 14.0, 17.0])
C = len(FITC)
I_ALL = torch.tensor(np.arange(W.N_CUR) * W.I_STEP + W.I_LOW, dtype=torch.float32, device=DEV)
CASES = [
    ("conductance, g_Na fixed 140",   (140., 36., .03), [False, True,  True ], 0.0, 0.0, 96.15),
    ("conductance, g_K fixed 20",     (120., 20., .03), [True,  False, True ], 0.0, 0.0, 93.97),
    ("conductance, g_leak fixed 0.30",(120., 36., .30), [True,  True,  False], 0.0, 0.0, 95.02),
    ("kinetic, sodium +6.1 mV",       (120., 36., .03), [True,  True,  True ], 6.1, 0.0, 97.60),
    ("kinetic, potassium +4.7 mV",    (120., 36., .03), [True,  True,  True ], 0.0, 4.7, 99.35),
    ("kinetic, potassium +9.4 mV",    (120., 36., .03), [True,  True,  True ], 0.0, 9.4, 99.05),
    ("kinetic, both shifts",          (120., 36., .03), [True,  True,  True ], 6.1, 4.7, 95.28),
]


def hard_spikes(P, sm, sn, chunk=4096):
    """SCORING uses the repository's own unmodified integrator."""
    P = np.asarray(P, float).reshape(-1, 3)
    out = np.empty((P.shape[0], W.N_CUR), int)
    for s0 in range(0, P.shape[0], chunk):
        e = min(s0 + chunk, P.shape[0]); nb = e - s0
        blk = np.repeat(P[s0:e], W.N_CUR, axis=0)
        g = [torch.tensor(blk[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
        st = W.simulate_stats_fast(g[0], g[1], g[2], I_ALL.repeat(nb), shift=float(sm),
                                   shift_n=float(sn), device=DEV, steps_per_call=20)
        out[s0:e] = st["spikes"].cpu().numpy().reshape(nb, W.N_CUR)
    return out


SC_WT = hard_spikes([W.WT], 0.0, 0.0)[0]
D_UNTR = [int(np.abs(hard_spikes([c[1]], c[3], c[4])[0] - SC_WT).sum()) for c in CASES]
with torch.no_grad():
    g = [torch.full((C,), float(x), dtype=torch.float32, device=DEV) for x in W.WT]
    TGT = O.traj(g[0], g[1], g[2], torch.tensor(FITC, dtype=torch.float32, device=DEV),
                 grad=False).unsqueeze(1)

rng = np.random.default_rng(20260804)
starts, masks, sm, sn, owner = [], [], [], [], []
for ci, (nm, untr, mask, s_m, s_n, _) in enumerate(CASES):
    for r in range(args.restarts):
        p = np.array(untr, float)
        if r > 0:
            for j in range(3):
                if mask[j]: p[j] = p[j] * 10 ** rng.uniform(-0.6, 0.6)
        starts.append(p); masks.append(mask); sm.append(s_m); sn.append(s_n); owner.append(ci)
starts = np.array(starts); masks = np.array(masks); owner = np.array(owner)
P = starts.shape[0]
I = torch.tensor(np.tile(FITC, P), dtype=torch.float32, device=DEV)
sh_m = torch.tensor(np.repeat(sm, C), dtype=torch.float32, device=DEV)
sh_n = torch.tensor(np.repeat(sn, C), dtype=torch.float32, device=DEV)
par = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV, requires_grad=True) for j in range(3)]
fixed = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
last = [torch.tensor(starts[:, j], dtype=torch.float32, device=DEV) for j in range(3)]
keep = [torch.tensor(~masks[:, j], device=DEV) for j in range(3)]
opt = torch.optim.Adam([{"params": [par[j]], "lr": LRS[W.NAMES[j]]} for j in range(3)])

best = np.full(P, np.inf); best_p = starts.copy(); n_nonfinite = 0; n_guard = 0
t0 = time.time()
for it in range(args.iters):
    gg = [q.unsqueeze(1).expand(P, C).reshape(-1) for q in par]
    v = O.traj(gg[0], gg[1], gg[2], I, shift=sh_m, shift_n=sh_n).view(-1, P, C)
    loss = O.obj_voltage_mse(v, TGT)
    lv = loss.detach().double().cpu().numpy()
    n_nonfinite += int((~np.isfinite(lv)).sum())
    opt.zero_grad(set_to_none=True)
    torch.nan_to_num(loss, nan=1e12, posinf=1e12, neginf=1e12).sum().backward()
    opt.step()
    with torch.no_grad():
        for j in range(3):
            bad = ~torch.isfinite(par[j])
            if bool(bad.any()):
                n_guard += int(bad.sum()); par[j].copy_(torch.where(bad, last[j], par[j]))
            par[j].clamp_(min=1e-6)
            par[j].copy_(torch.where(keep[j], fixed[j], par[j]))
            last[j].copy_(par[j])
    pv = np.stack([q.detach().double().cpu().numpy() for q in par], axis=1)
    imp = np.isfinite(lv) & (lv < best)
    best[imp] = lv[imp]; best_p[imp] = pv[imp]
secs = time.time() - t0

print(f"non-finite losses {n_nonfinite} of {args.iters*P}; guard fired {n_guard} times "
      f"[{secs:.0f} s]\n")
print(f"{'case':<32} {'safe rates':>12} {'as reported':>12} {'direct search':>14}")
AS_REPORTED = [96.15, -90.52, -133.83, 0.22, 97.40, -7.62, -258.27]   # wo6_taskA.py, voltage MSE
OUT = {"host": platform.node(), "nonfinite": n_nonfinite, "guard": n_guard, "results": []}
for ci, (nm, untr, mask, s_m, s_n, bench) in enumerate(CASES):
    sel = np.where(owner == ci)[0]
    d = np.abs(hard_spikes(best_p[sel], s_m, s_n) - SC_WT[None, :]).sum(axis=1)
    sims = W.similarity(d, D_UNTR[ci])
    b = int(np.argmin(best[sel]))
    OUT["results"].append(dict(case=nm, sim=float(sims[b]), benchmark=bench,
                               as_reported=AS_REPORTED[ci],
                               params=list(map(float, best_p[sel][b]))))
    print(f"{nm:<32} {sims[b]:>11.2f}% {AS_REPORTED[ci]:>11.2f}% {bench:>13.2f}%")
hits = sum(1 for r in OUT["results"] if r["sim"] >= r["benchmark"] - 5)
harm = sum(1 for r in OUT["results"] if r["sim"] < 0)
print(f"\n-> within 5 points of direct search in {hits} of 7 (was 2 of 7); "
      f"worse than no treatment in {harm} of 7 (was 4 of 7)")
with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1)
