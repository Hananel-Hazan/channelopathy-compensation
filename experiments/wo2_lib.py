"""Shared harness for the gradient-fitting experiments (wo2_task*.py).

The Hodgkin-Huxley model is taken unchanged from mimic_cell_activity/HH.py:
the file is read and only the part above its driver code (which starts at
`running_time = 1500`) is executed. The HH class accepts array-valued
conductances, so a batch of B parameter sets integrates in one forward pass.

This module adds loss functions, optimizer construction and the training loop.

One structural detail of the original driver script matters and is reproduced
as an option. The driver rebuilds the whole HH object at the top of every epoch,
and because the optimizer is created in `HH.__init__`, a new optimizer is created
every epoch too. Any optimizer state (Adam's moment estimates, Adagrad's
accumulator, Rprop's step sizes) is therefore discarded every epoch.
`reset_each_iter=True` reproduces that; `reset_each_iter=False` keeps state.
"""
import os

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HH_SRC = os.path.join(REPO, "mimic_cell_activity", "HH.py")

_ns = {}
_text = open(HH_SRC).read()
exec(compile(_text[:_text.index("running_time = 1500")], HH_SRC, "exec"), _ns)
HH = _ns["HH"]
pearsonr_1d = _ns["pearsonr"]          # Pearson correlation defined in HH.py

DT_FIG = 10.0 / 1500.0                 # ms; a 10 ms window in 1500 steps, the time axis of the original driver
DT_FILE = 0.05                         # ms; the default dt of the HH class
N_STEPS = 1500

WT = (120.0, 36.0, 0.03)               # wild-type reference conductances (gNa, gK, gl), mS/cm^2
VARIANT = (140.0, 20.0, 0.30)          # variant starting point for the fits
NAMES = ("Na", "K", "l")


def _vec(x, B):
    return np.full(B, float(x)) if np.isscalar(x) else np.asarray(x, dtype=float)


def build(gNa, gK, gl, dt, I, B=1, v0=-60.0, device="cpu"):
    hh = HH(v=np.full(B, v0), g_Na=_vec(gNa, B), g_K=_vec(gK, B), g_l=_vec(gl, B),
            dt=dt, device=device)
    hh.const_I = torch.tensor(float(I), dtype=torch.float, device=device)
    return hh


def roll(hh, n_steps=N_STEPS, grad=True):
    B = hh.v.shape[0] if hh.v.dim() else 1
    out = torch.zeros(n_steps, B, device=hh.v.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i in range(n_steps):
            hh.forward()
            out[i] = hh.v
    return out


def simulate(gNa, gK, gl, dt=DT_FIG, I=8.0, n_steps=N_STEPS, B=1, device="cpu"):
    return roll(build(gNa, gK, gl, dt, I, B=B, device=device), n_steps, grad=False)


# ---------------------------------------------------------------- losses
# Each returns a vector of length B, one loss per batch element. The batch
# elements never interact in the forward pass, so summing the vector and calling
# backward gives every element an independent gradient.

def loss_mse(v, t):
    return ((v - t) ** 2).mean(dim=0)


def loss_mae(v, t):
    return (v - t).abs().mean(dim=0)


def loss_smooth_l1(v, t, beta=1.0):
    d = (v - t).abs()
    return torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean(dim=0)


def loss_correlation(v, t):
    """1 - Pearson r, with the same sign handling as the original driver (1 + r for r <= 0)."""
    vm = v - v.mean(dim=0, keepdim=True)
    tm = t - t.mean(dim=0, keepdim=True)
    r = (vm * tm).sum(dim=0) / (vm.norm(dim=0) * tm.norm(dim=0))
    return torch.where(r > 0, 1 - r, 1 + r)


LOSSES = {"mse": loss_mse, "correlation": loss_correlation,
          "mae": loss_mae, "smooth_l1": loss_smooth_l1}


def check_correlation_matches_repo(n=500, seed=0):
    """Confirm the batched correlation loss equals the one computed with HH.py's pearsonr on 1-D input."""
    torch.manual_seed(seed)
    a, b = torch.randn(n), torch.randn(n)
    mine = float(loss_correlation(a.unsqueeze(1), b.unsqueeze(1))[0])
    r = pearsonr_1d(a, b)
    theirs = float(1 - r if r > 0 else 1 * (1 + r))
    return mine, theirs


# ---------------------------------------------------------------- optimizers

def make_optimizer(name, groups):
    n = name.lower()
    table = {
        "sgd":        lambda g: torch.optim.SGD(g),
        "sgd_mom":    lambda g: torch.optim.SGD(g, momentum=0.9),
        "asgd":       lambda g: torch.optim.ASGD(g),
        "adam":       lambda g: torch.optim.Adam(g, amsgrad=False),
        "adam_ams":   lambda g: torch.optim.Adam(g, amsgrad=True),
        "adamw":      lambda g: torch.optim.AdamW(g),
        "adadelta":   lambda g: torch.optim.Adadelta(g, rho=0.9, eps=1e-06, weight_decay=0),
        "adagrad":    lambda g: torch.optim.Adagrad(g, lr_decay=0, weight_decay=0,
                                                    initial_accumulator_value=0, eps=1e-1),
        "rmsprop":    lambda g: torch.optim.RMSprop(g),
        "rprop":      lambda g: torch.optim.Rprop(g),
    }
    if n not in table:
        raise ValueError(name)
    return table[n](groups)


ADAPTIVE = {"adam", "adam_ams", "adamw", "adadelta", "adagrad", "rmsprop", "rprop"}


# ---------------------------------------------------------------- training

def fit(frozen="Na", start=VARIANT, target_params=WT, target_trace=None,
        loss_name="mse", optimizer="adam", lrs=None, iters=150,
        dt=DT_FIG, I=8.0, n_steps=N_STEPS, B=1, clamp_min=None,
        reset_each_iter=True, device="cpu"):
    """Hold one conductance fixed, optimize the other two toward a target.

    Returns a dict of numpy arrays: iteration index, loss, the three
    conductances, and the gradients, each of shape (n_recorded, B).
    """
    if target_trace is None:
        with torch.no_grad():
            target = roll(build(*target_params, dt, I, B=B, device=device),
                          n_steps, grad=False)
    else:
        target = target_trace

    # persistent leaf parameters, so optimizer state can survive across iterations
    p = {n: torch.tensor(_vec(v, B), dtype=torch.float, device=device,
                         requires_grad=(n != frozen))
         for n, v in zip(NAMES, start)}

    groups = [{"params": p[n], "lr": float(lrs[n])} for n in NAMES if n != frozen]
    opt = make_optimizer(optimizer, groups)
    lossfn = LOSSES[loss_name]

    rec = {k: [] for k in ("iter", "loss", "Na", "K", "l", "gNa", "gK", "gl")}
    would_go_negative = False
    diverged_at = None

    for it in range(iters):
        if reset_each_iter:
            opt = make_optimizer(optimizer, groups)
        hh = build(0.0, 0.0, 0.0, dt, I, B=B, device=device)
        # inject the persistent parameters; the initial m/h/n depend only on v, not on the conductances
        hh.Var_g_Na, hh.Var_g_K, hh.Var_g_l = p["Na"], p["K"], p["l"]

        v = roll(hh, n_steps, grad=True)
        losses = lossfn(v, target)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()

        g = {n: (p[n].grad.detach().cpu().numpy().copy()
                 if p[n].grad is not None else np.full(B, np.nan)) for n in NAMES}
        opt.step()

        with torch.no_grad():
            for n in NAMES:
                if n == frozen:
                    continue
                if bool((p[n] < 0).any()):
                    would_go_negative = True
                if clamp_min is not None:
                    p[n].clamp_(min=clamp_min)

        rec["iter"].append(it)
        rec["loss"].append(losses.detach().cpu().numpy().copy())
        for n in NAMES:
            rec[n].append(p[n].detach().cpu().numpy().copy())
            rec["g" + n].append(g[n])

        if not np.isfinite(np.concatenate([rec["K"][-1], rec["l"][-1],
                                           rec["Na"][-1]])).all():
            diverged_at = it
            break

    out = {k: np.array(v) for k, v in rec.items()}
    out["would_go_negative"] = would_go_negative
    out["diverged_at"] = diverged_at
    out["frozen"] = frozen
    out["optimizer"] = optimizer
    out["loss_name"] = loss_name
    out["lrs"] = dict(lrs)
    return out


def plateau_iter(loss, rel_tol=1e-4, window=20):
    """First iteration after which the loss improves by less than rel_tol
    (relative to its own value) over `window` iterations."""
    L = np.asarray(loss, dtype=float)
    if L.ndim > 1:
        L = L[:, 0]
    n = len(L)
    for i in range(n - window):
        a, b = L[i], L[i + window]
        if not np.isfinite(a) or not np.isfinite(b) or a == 0:
            continue
        if abs(a - b) / abs(a) < rel_tol:
            return i
    return n - 1


# ------------------------------------------------- many configurations at once
# Every configuration gets its own length-1 parameter tensors and its own
# optimizer instance. The tensors are concatenated for the forward pass, so all
# configurations integrate together in one batched simulation, but each is
# stepped by its own optimizer with its own learning rates. The batch elements
# never interact, so the gradients are independent.

def fit_grid(configs, target_params=WT, target_trace=None, loss_name="mse",
             iters=150, dt=DT_FIG, I=8.0, n_steps=N_STEPS, clamp_min=None,
             reset_each_iter=True, device="cpu", record_grads=True):
    """configs: list of dicts with keys
         'frozen'    : 'Na' | 'K' | 'l'
         'start'     : (gNa, gK, gl)
         'optimizer' : name accepted by make_optimizer
         'lrs'       : {'Na':..,'K':..,'l':..}
         'label'     : free text
    Returns dict of arrays shaped (iters, n_configs).
    """
    B = len(configs)
    if target_trace is None:
        with torch.no_grad():
            target = roll(build(*target_params, dt, I, B=B, device=device),
                          n_steps, grad=False)
    else:
        target = target_trace

    P = []          # P[b][name] -> length-1 leaf tensor
    OPT = []
    for c in configs:
        pb = {}
        for n, v in zip(NAMES, c["start"]):
            pb[n] = torch.tensor([float(v)], dtype=torch.float, device=device,
                                 requires_grad=(n != c["frozen"]))
        groups = [{"params": pb[n], "lr": float(c["lrs"][n])}
                  for n in NAMES if n != c["frozen"]]
        P.append(pb)
        OPT.append(make_optimizer(c["optimizer"], groups))

    lossfn = LOSSES[loss_name]
    rec = {k: [] for k in ("loss", "Na", "K", "l", "gNa", "gK", "gl")}
    neg = np.zeros(B, dtype=bool)
    dead = np.zeros(B, dtype=bool)

    for it in range(iters):
        if reset_each_iter:
            OPT = [make_optimizer(c["optimizer"],
                                  [{"params": P[b][n], "lr": float(c["lrs"][n])}
                                   for n in NAMES if n != c["frozen"]])
                   for b, c in enumerate(configs)]

        cat = {n: torch.cat([P[b][n] for b in range(B)]) for n in NAMES}
        hh = build(0.0, 0.0, 0.0, dt, I, B=B, device=device)
        hh.Var_g_Na, hh.Var_g_K, hh.Var_g_l = cat["Na"], cat["K"], cat["l"]

        v = roll(hh, n_steps, grad=True)
        losses = lossfn(v, target)
        raw_loss = losses.detach().cpu().numpy().copy()   # keep NaN/Inf as-is
        # only the value fed to backward is sanitised, so that one diverged
        # batch element cannot poison the gradients of the others
        losses = torch.where(torch.isfinite(losses), losses,
                             torch.zeros_like(losses))
        for b in range(B):
            OPT[b].zero_grad(set_to_none=True)
        losses.sum().backward()

        gsnap = {n: np.array([float(P[b][n].grad) if P[b][n].grad is not None
                              else np.nan for b in range(B)]) for n in NAMES} \
            if record_grads else {n: np.full(B, np.nan) for n in NAMES}

        for b in range(B):
            if not dead[b]:
                OPT[b].step()

        with torch.no_grad():
            for b, c in enumerate(configs):
                for n in NAMES:
                    if n == c["frozen"]:
                        continue
                    if bool((P[b][n] < 0).any()):
                        neg[b] = True
                    if clamp_min is not None:
                        P[b][n].clamp_(min=clamp_min)
                    if not torch.isfinite(P[b][n]).all():
                        dead[b] = True
                        P[b][n].data = torch.full_like(P[b][n], float("nan"))

        # raw_loss is already NaN/Inf whenever the parameters or the trace are,
        # so no masking is applied -- a diverged configuration reports NaN, not 0
        rec["loss"].append(raw_loss)
        for n in NAMES:
            rec[n].append(np.array([float(P[b][n].detach()) for b in range(B)]))
            rec["g" + n].append(gsnap[n])

    out = {k: np.array(v) for k, v in rec.items()}
    out["configs"] = configs
    out["would_go_negative"] = neg
    out["dead"] = dead
    out["loss_name"] = loss_name
    out["I"] = I
    return out
