"""What the ModelDB 87585 R859C sodium-channel mechanism implements.

Reads the two archived NEURON mechanisms and the two archived voltage traces,
and answers one question with numbers: over the 300 ms AP-threshold sweep, how
much does the slow-inactivation state variable `s` move, and how much of that
movement is attributable to the WT-vs-R859C difference in `stau`?

Two independent lines of evidence:

  (A) trace-driven -- integrate the mechanism's own exponential-Euler update for
      `s` along each ARCHIVED voltage trace, once with the R859C `stau` law and
      once with the WT `stau` law, and report the divergence.  This needs no
      re-simulation and cannot drift from the record.

  (B) closed-loop -- a from-scratch single-compartment reimplementation of the
      .mod + .hoc, validated against the archived spike counts, then re-run with
      ONLY the `stau` law swapped, to see whether any spike count changes.

  (C) the voltage at which each mechanism's `stau` law reproduces the slow-recovery time
      constants of Barela et al. (2006), and `stau` on a voltage grid.

  (D) attribution -- R859C re-run with the wild-type activation curve, and with the
      wild-type `stau` law, one at a time.

Reads the archived traces in Neuron_Test/R859C/ and writes
figures/wo9_r859c_mechanism_audit.json.  Parts (B) and (D) are slow pure-Python
simulations and run only with --with-sim.  figures/scripts/wo9_task11_neuron_verification.py
repeats the closed-loop experiments in NEURON itself.

Usage:  python3 wo9_task11_slow_inactivation.py [--with-sim]
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
FILES = {"WT": os.path.join(ARCH, "APthreshold WT-2005 - log every 10th data point.txt"),
         "MT": os.path.join(ARCH, "APthreshold R859C - log every 10th data point.txt")}
OUT = os.path.join(REPO, "figures")


# --------------------------------------------------------------------------
# mechanism constants, transcribed from ichanWT2005.mod and ichanR859C1.mod
# --------------------------------------------------------------------------
MECH = {
    #            minf midpoint,  minf slope k,  stau prefactor(s), stau Vpeak, stau width
    "WT":  dict(m_half=-27.4, m_k=4.7, s_pre=140.4, s_vp=-71.3, s_w=30.9),
    "MT":  dict(m_half=-21.3, m_k=3.5, s_pre=190.2, s_vp=-90.4, s_w=38.9),
}
# identical in both files:
#   hinf = 1/(1+exp((v+41.9)/6.7)); htau = 23.12*exp(-0.5*((v+77.58)/43.92)^2)
#   sinf = 1/(1+exp((v+46.0)/6.6))
#   nf   : alpha = -0.07*vtrap(v+65-47,-6); beta = 0.264/exp((v+65-22)/40)
#   mtau = 0.15


def minf(v, p):
    return 1.0 / (1.0 + np.exp(-(v - p["m_half"]) * p["m_k"] * 0.03937))


def hinf(v):
    return 1.0 / (1.0 + np.exp((v + 41.9) / 6.7))


def htau(v):
    return 23.12 * np.exp(-0.5 * ((v + 77.58) / 43.92) ** 2)


def sinf(v):
    return 1.0 / (1.0 + np.exp((v + 46.0) / 6.6))


def stau(v, p):
    """ms.  The .mod writes stau = 1000*(pre*exp(...)), and declares stau (ms),
    so `pre` is in SECONDS and stau comes out in milliseconds."""
    return 1000.0 * (p["s_pre"] * np.exp(-0.5 * ((v - p["s_vp"]) / p["s_w"]) ** 2))


def vtrap(x, y):
    small = np.abs(x / y) < 1e-6
    return np.where(small, y * (1 - x / y / 2.0), x / (np.expm1(x / y)))


def nf_rates(v):
    alpha = -0.07 * vtrap((v + 65 - 47), -6.0)
    beta = 0.264 / np.exp((v + 65 - 22) / 40.0)
    s = alpha + beta
    return 1.0 / s, alpha / s          # nftau, nfinf


def load(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        f.readline()
        rows = [[float(p) for p in line.rstrip("\n").split("\t") if p != ""]
                for line in f if line.strip()]
    data = np.array(rows)
    currents = np.array([int(h.split()[0]) for h in header[1:] if h.strip()])
    return data[:, 0], currents, data[:, 1:].T


def find_spikes_pipeline(x):
    """Spike count per trace: binarize at -10 mV and count level changes / 2.

    Transcribed from the original R859C search pipeline's spike detector (the same rule
    as in wo7_task13_excitability.py)."""
    v = np.atleast_2d(x).copy()
    thr = -10
    v[v < thr] = -10
    v[v > thr] = 10
    d = np.diff(v, axis=1)
    return np.array((d != 0).sum(axis=1) / 2, dtype=int)


# ==========================================================================
# (A) trace-driven integration of s
# ==========================================================================
def integrate_s(vtrace, dt, p, q10=1.0):
    """Mechanism's own update: s <- s + (1-exp(-dt*q10/stau))*(sinf-s)."""
    s = sinf(vtrace[0])
    out = np.empty_like(vtrace)
    out[0] = s
    for i in range(1, vtrace.size):
        v = vtrace[i - 1]
        sexp = 1.0 - np.exp(-dt * q10 / stau(v, p))
        s = s + sexp * (sinf(v) - s)
        out[i] = s
    return out


def part_A():
    rep = {}
    for arm, path in FILES.items():
        t, I, V = load(path)
        dt = float(t[1] - t[0])
        own = MECH[arm]
        other = MECH["WT" if arm == "MT" else "MT"]
        max_drift, max_cross, s0 = 0.0, 0.0, sinf(V[0, 0])
        for k in range(V.shape[0]):
            s_own = integrate_s(V[k], dt, own)
            s_oth = integrate_s(V[k], dt, other)
            max_drift = max(max_drift, float(np.max(np.abs(s_own - s_own[0]))))
            max_cross = max(max_cross, float(np.max(np.abs(s_own - s_oth))))
        vmin, vmax = float(V.min()), float(V.max())
        rep[arm] = dict(
            trace=os.path.basename(path), dt_ms=dt, n_levels=int(V.shape[0]),
            sweep_ms=float(t[-1]), v_min_mV=vmin, v_max_mV=vmax,
            s_initial=float(s0),
            stau_at_rest_ms=float(stau(-60.0, own)),
            stau_min_over_visited_ms=float(np.min(stau(np.linspace(vmin, vmax, 4001), own))),
            stau_max_over_visited_ms=float(np.max(stau(np.linspace(vmin, vmax, 4001), own))),
            max_abs_s_drift_from_initial=max_drift,
            max_abs_s_difference_WT_vs_MT_stau=max_cross,
        )
    return rep


# ==========================================================================
# (B) closed-loop reimplementation of the .hoc cell
# ==========================================================================
AREA_CM2 = np.pi * 25e-4 * 25e-4          # pi * diam(cm) * L(cm), nseg=1
GNATBAR, GKFBAR, GL = 0.2, 0.06, 0.0005   # mho/cm2, from the .hoc
ENAT, EKF, EL = 50.0, -80.0, -60.0
CM = 1.0                                   # uF/cm2


def simulate(I_pA, p, s_law=None, dt=0.01, tstop=300.0, v_init=-60.0,
             log_every=10, q10=1.0):
    """Single-compartment reimplementation of the .hoc cell.  Gates use the
    mechanism's own exponential-Euler update; the membrane uses the same
    exponential step on the instantaneous conductance sum.  `s_law` overrides
    which mechanism's stau is used, leaving every other parameter at `p`."""
    import math
    if s_law is None:
        s_law = p
    exp = math.exp
    mh, mk = p["m_half"], p["m_k"]
    spre, svp, sw = s_law["s_pre"], s_law["s_vp"], s_law["s_w"]
    n = int(round(tstop / dt))
    v = v_init
    m = 1.0 / (1.0 + exp(-(v - mh) * mk * 0.03937))
    h = 1.0 / (1.0 + exp((v + 41.9) / 6.7))
    s = 1.0 / (1.0 + exp((v + 46.0) / 6.6))
    a = -0.07 * ((v + 65 - 47) / (exp((v + 65 - 47) / -6.0) - 1.0))
    b = 0.264 / exp((v + 65 - 22) / 40.0)
    nfv = a / (a + b)
    i_inj = I_pA * 1e-9 / AREA_CM2               # pA -> mA (1e-9), per cm2
    out_v = []
    for k in range(n):
        t = k * dt
        if k % log_every == 0:
            out_v.append(v)
        # --- gates, at the current v -------------------------------------
        m_inf = 1.0 / (1.0 + exp(-(v - mh) * mk * 0.03937))
        h_inf = 1.0 / (1.0 + exp((v + 41.9) / 6.7))
        h_tau = 23.12 * exp(-0.5 * ((v + 77.58) / 43.92) ** 2)
        s_inf = 1.0 / (1.0 + exp((v + 46.0) / 6.6))
        s_tau = 1000.0 * (spre * exp(-0.5 * ((v - svp) / sw) ** 2))
        a = -0.07 * ((v + 65 - 47) / (exp((v + 65 - 47) / -6.0) - 1.0))
        b = 0.264 / exp((v + 65 - 22) / 40.0)
        nf_tau, nf_inf = 1.0 / (a + b), a / (a + b)
        m += (1.0 - exp(-dt * q10 / 0.15)) * (m_inf - m)
        h += (1.0 - exp(-dt * q10 / h_tau)) * (h_inf - h)
        s += (1.0 - exp(-dt * q10 / s_tau)) * (s_inf - s)
        nfv += (1.0 - exp(-dt * q10 / nf_tau)) * (nf_inf - nfv)
        # --- membrane -----------------------------------------------------
        gna = GNATBAR * m * m * m * h * s
        gk = GKFBAR * nfv * nfv * nfv * nfv
        gtot = gna + gk + GL
        drive = gna * ENAT + gk * EKF + GL * EL
        if 50.0 <= t < 250.0:
            drive += i_inj
        vinf = drive / gtot
        v = vinf + (v - vinf) * exp(-dt * 1000.0 * gtot / CM)
    return np.array(out_v)


def part_B():
    """Validate the reimplementation against the archived spike counts, then swap
    only the slow-inactivation law."""
    res = {}
    for arm in ("WT", "MT"):
        t_a, I, V_a = load(FILES[arm])
        arch_counts = find_spikes_pipeline(V_a)
        own = np.array([find_spikes_pipeline(simulate(int(i), MECH[arm]))[0] for i in I])
        swapped_law = MECH["WT" if arm == "MT" else "MT"]
        swap = np.array([find_spikes_pipeline(
            simulate(int(i), MECH[arm], s_law=swapped_law))[0] for i in I])
        res[arm] = dict(
            currents=I.tolist(),
            archived_counts=arch_counts.tolist(),
            archived_total=int(arch_counts.sum()),
            reimpl_counts=own.tolist(),
            reimpl_total=int(own.sum()),
            reimpl_matches_archive=bool(np.array_equal(arch_counts, own)),
            reimpl_levels_matching=int((arch_counts == own).sum()),
            stau_swapped_counts=swap.tolist(),
            stau_swapped_total=int(swap.sum()),
            stau_swap_changes_any_level=bool(not np.array_equal(own, swap)),
            stau_swap_levels_changed=int((own != swap).sum()),
        )
    return res




# ==========================================================================
# (C) does the mechanism's stau law encode Barela's recovery constants?
# ==========================================================================
def part_C():
    from scipy.optimize import brentq
    out = {}
    for arm, target in (("WT", 40.5), ("MT", 141.7)):
        p = MECH[arm]
        v = brentq(lambda x: stau(x, p) / 1000.0 - target, -200.0, p["s_vp"] - 1e-9)
        out[arm] = dict(barela_tau_slow_recovery_s=target,
                        voltage_where_mechanism_matches_mV=float(v))
    grid = {}
    for v in (-140, -130, -120, -110, -100, -90, -80, -70, -60):
        grid[str(v)] = dict(stau_WT_s=float(stau(v, MECH["WT"]) / 1000.0),
                            stau_MT_s=float(stau(v, MECH["MT"]) / 1000.0),
                            ratio_MT_over_WT=float(stau(v, MECH["MT"]) / stau(v, MECH["WT"])))
    out["stau_vs_voltage_seconds"] = grid
    out["note"] = ("Both Gaussians reproduce Barela et al. 2006 Table 3 recovery "
                   "tau_slow at v = -120 mV (ratio 3.51 there, matching the paper's "
                   "'approximately 3.5-fold'). At the protocol's resting -60 mV the "
                   "ratio is 1.07 and both constants exceed 130 s.")
    return out


# ==========================================================================
# (D) attribution -- revert ONE difference at a time
# ==========================================================================
def part_D():
    t, I, V = load(FILES["WT"])
    wt_re = np.array([int(find_spikes_pipeline(simulate(int(i), MECH["WT"]))[0]) for i in I])
    mt_re = np.array([int(find_spikes_pipeline(simulate(int(i), MECH["MT"]))[0]) for i in I])
    mt_actWT = dict(MECH["MT"]); mt_actWT["m_half"] = MECH["WT"]["m_half"]
    mt_actWT["m_k"] = MECH["WT"]["m_k"]
    c_act = np.array([int(find_spikes_pipeline(simulate(int(i), mt_actWT))[0]) for i in I])
    c_stau = np.array([int(find_spikes_pipeline(
        simulate(int(i), MECH["MT"], s_law=MECH["WT"]))[0]) for i in I])
    return dict(
        currents=I.tolist(),
        WT_reimpl=wt_re.tolist(), WT_total=int(wt_re.sum()),
        MT_reimpl=mt_re.tolist(), MT_total=int(mt_re.sum()),
        MT_with_WT_activation=c_act.tolist(), MT_with_WT_activation_total=int(c_act.sum()),
        MT_with_WT_activation_equals_WT_at_all_levels=bool(np.array_equal(c_act, wt_re)),
        MT_with_WT_stau=c_stau.tolist(), MT_with_WT_stau_total=int(c_stau.sum()),
        MT_with_WT_stau_equals_MT_at_all_levels=bool(np.array_equal(c_stau, mt_re)),
    )


if __name__ == "__main__":
    report = {"partA_trace_driven": part_A()}
    report["partC_barela_crosscheck"] = part_C()
    if "--with-sim" in sys.argv:
        report["partB_closed_loop"] = part_B()
        report["partD_attribution"] = part_D()
    # static facts about the mechanisms
    report["mechanism_constants"] = {
        "minf_midpoint_mV": {"WT": MECH["WT"]["m_half"], "MT": MECH["MT"]["m_half"],
                             "shift_mV": MECH["MT"]["m_half"] - MECH["WT"]["m_half"]},
        "minf_boltzmann_slope_factor_mV": {
            "WT": 1.0 / (MECH["WT"]["m_k"] * 0.03937),
            "MT": 1.0 / (MECH["MT"]["m_k"] * 0.03937),
            "pct_change": 100.0 * (MECH["WT"]["m_k"] / MECH["MT"]["m_k"] - 1.0)},
        "gnatbar_mho_cm2": {"WT": GNATBAR, "MT": GNATBAR,
                            "note": "identical in both .hoc files"},
        "sinf_identical": True,
        "stau_peak_seconds": {"WT": MECH["WT"]["s_pre"], "MT": MECH["MT"]["s_pre"]},
        "stau_peak_ratio_MT_over_WT": MECH["MT"]["s_pre"] / MECH["WT"]["s_pre"],
    }
    path = os.path.join(OUT, "wo9_r859c_mechanism_audit.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("\nwrote", path)
