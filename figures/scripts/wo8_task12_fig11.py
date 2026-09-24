"""Regenerate the Hodgkin-Huxley convergence figure (Figure 11) with corrected axes.

The figure compares the reference (wild-type) Hodgkin-Huxley model with the
fitted model at epoch 150, conductances (140, 39.678, 0.155).  The leak
component of that vector is not determined by the fitting objective; the panel
is simply drawn at that value, and the in-image footnote says so.

Two plotting errors in the plotting routine of the original fitting script,
which drew the published panel, are corrected here:

Transposed phase-plane axes.
    That routine called plt.plot(n_target, v_target) -- n on the abscissa,
    voltage on the ordinate -- under xlabel('voltage') and ylabel('n channels').
    Here V is on the abscissa and n on the ordinate.

Wrong time axis.
    That routine built the time axis as np.linspace(0, 10, 1500), i.e. always
    0-10 ms, while the integrator advances by dt per step.  At the class
    default dt = 0.05, 1500 steps span 75 ms.  Here the time vector is
    np.arange(n_steps) * dt.
    A second figure (hh_time_axis_defect) shows the same run on both axes.

The integration step and injected current used are printed and stated in the
figure footnote.  The HH class is loaded unchanged from
mimic_cell_activity/HH.py via wo7_hh.load_hh_class.

Writes, in figures/: hh_convergence_phase_plane.{pdf,png,csv,json} and
hh_time_axis_defect.{pdf,png}.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save
from wo7_hh import simulate, load_hh_class

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")

N_STEPS = 1500
DT_FIG = 10.0 / N_STEPS      # 0.0066667 ms -- the step that reproduces the panel
DT_FILE = 0.05               # HH class default dt
I_FIG = 8.0                  # the injected current of the recorded fitting run
I_FILE = 7.0                 # HH class default injected current

WT = dict(g_Na=120.0, g_K=36.0, g_l=0.03)          # reference / wild type
CONV = dict(g_Na=140.0, g_K=39.678, g_l=0.155)      # epoch-150 converged model


def three_panel(t, tgt, mdl, dt, current, stem, subtitle):
    (v_t, m_t, h_t, n_t) = tgt
    (v_m, m_m, h_m, n_m) = mdl

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.2))
    ax1, ax2, ax3 = axes

    # --- panel 1: voltage trajectory ----------------------------------------
    ax1.plot(t, v_t, "-", color=C.WT, label="reference (wild type)")
    ax1.plot(t, v_m, "--", color=C.COMPENSATED, label="fitted model (epoch 150)")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("membrane potential (mV)")
    ax1.set_xlim(t[0], t[-1])
    ax1.legend(loc="upper right", ncol=1)
    ax1.set_title("a   voltage trajectory", loc="left", color=C.INK)

    # --- panel 2: gating variables ------------------------------------------
    for arr_t, arr_m, col, name in ((m_t, m_m, C.M, "m"),
                                    (h_t, h_m, C.H, "h"),
                                    (n_t, n_m, C.N, "n")):
        ax2.plot(t, arr_t, "-", color=col, label=f"{name}  reference")
        ax2.plot(t, arr_m, "--", color=col, label=f"{name}  fitted")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("gating variable (dimensionless, 0–1)")
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.4)
    ax2.set_title("b   gating dynamics  (colour = variable, "
                  "solid = reference, dashed = fitted)",
                  loc="left", color=C.INK)

    # --- panel 3: phase plane, n on the ORDINATE and V on the ABSCISSA -------
    ax3.plot(v_t, n_t, "-", color=C.WT, label="reference (wild type)")
    ax3.plot(v_m, n_m, "--", color=C.COMPENSATED, label="fitted model (epoch 150)")
    ax3.set_xlabel("membrane potential V (mV)")
    ax3.set_ylabel("potassium activation n (dimensionless)")
    ax3.legend(loc="lower right")
    ax3.set_title("c   phase plane  (n against V)",
                  loc="left", color=C.INK)

    fig.suptitle("Hodgkin-Huxley fit at convergence — corrected axes",
                 x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.002, subtitle, ha="left", va="top",
             fontsize=7.5, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 0.975))
    return save(fig, stem)


def main():
    HH, span = load_hh_class()
    print(f"HH class taken verbatim from "
          f"mimic_cell_activity/HH.py lines {span[0]}-{span[1]}")
    print(f"class default dt = {HH.__init__.__defaults__[3]}  "
          f"(HH class default);  default injected current = 7 (HH class)\n")

    # ---- the regenerated figure: the step and current that make the panel ---
    t, v_t, m_t, h_t, n_t = simulate(**WT, dt=DT_FIG, n_steps=N_STEPS, current=I_FIG)
    _, v_m, m_m, h_m, n_m = simulate(**CONV, dt=DT_FIG, n_steps=N_STEPS, current=I_FIG)

    def n_spikes(v):
        above = v > 0.0
        return int((above[1:] & ~above[:-1]).sum())

    print(f"regenerated at dt = {DT_FIG:.7f} ms, I = {I_FIG:g}: "
          f"span {t[-1] + DT_FIG:.2f} ms, "
          f"reference peak {v_t.max():.2f} mV ({n_spikes(v_t)} action potential(s)), "
          f"fitted peak {v_m.max():.2f} mV ({n_spikes(v_m)})")
    print(f"  phase-plane ranges: V {v_t.min():.1f}..{v_t.max():.1f} mV, "
          f"n {n_t.min():.3f}..{n_t.max():.3f}")
    print(f"  -> the published panel's axis labelled 'voltage' spans "
          f"{n_t.min():.2f}-{n_t.max():.2f}; those are n values, which is the "
          f"transposition, visible in the figure itself")

    sub = (f"Integration step dt = {DT_FIG:.7f} ms (= 10 ms / {N_STEPS} steps), "
           f"injected current I = {I_FIG:g}, {N_STEPS} steps, span "
           f"{t[-1] + DT_FIG:.2f} ms.\n"
           f"Reference (120, 36, 0.03) mS/cm²; fitted model "
           f"(140, 39.678, 0.155) mS/cm² is the epoch-150 vector of the "
           f"original fit. Its leak component "
           f"is not determined by this objective.\n"
           f"Time vector is np.arange(n_steps) * dt, not the hard-coded "
           f"np.linspace(0, 10, 1500) of the original fitting script. Phase plane is n against V, "
           f"not its transposed\nplt.plot(n_target, v_target). "
           f"The HH class defaults are dt = 0.05 and I = 7; neither reproduces "
           f"the published panel.")
    paths = three_panel(t, (v_t, m_t, h_t, n_t), (v_m, m_m, h_m, n_m),
                        DT_FIG, I_FIG,
                        os.path.join(OUT, "hh_convergence_phase_plane"), sub)
    print("wrote " + ", ".join(paths))

    # ---- CSV behind the figure ---------------------------------------------
    csv_path = os.path.join(OUT, "hh_convergence_phase_plane.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_ms",
                    "V_reference_mV", "m_reference", "h_reference", "n_reference",
                    "V_fitted_mV", "m_fitted", "h_fitted", "n_fitted"])
        for i in range(N_STEPS):
            w.writerow([f"{t[i]:.7f}",
                        f"{v_t[i]:.6f}", f"{m_t[i]:.6f}", f"{h_t[i]:.6f}", f"{n_t[i]:.6f}",
                        f"{v_m[i]:.6f}", f"{m_m[i]:.6f}", f"{h_m[i]:.6f}", f"{n_m[i]:.6f}"])
    print(f"wrote {csv_path}")

    # ---- second figure: one run at the class defaults on both time axes ----
    t2, v2, _m2, _h2, _n2 = simulate(**WT, dt=DT_FILE, n_steps=N_STEPS,
                                     current=I_FILE)
    print(f"\nat the file's own dt = {DT_FILE} and I = {I_FILE:g}: span "
          f"{t2[-1] + DT_FILE:.1f} ms, {n_spikes(v2)} action potentials")

    apply_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    axa.plot(np.linspace(0, 10, N_STEPS), v2, "-", color=C.WT)
    axa.set_xlabel("time (ms) — as the original fitting script labels it")
    axa.set_ylabel("membrane potential (mV)")
    axa.set_title("a   the hard-coded axis", loc="left", color=C.INK)
    axa.set_xlim(0, 10)
    axb.plot(t2, v2, "-", color=C.WT)
    axb.set_xlabel("time (ms) — np.arange(n_steps) × dt")
    axb.set_ylabel("membrane potential (mV)")
    axb.set_title("b   the true axis", loc="left", color=C.INK)
    axb.set_xlim(0, t2[-1] + DT_FILE)
    fig.suptitle("The same 1500 integration steps at dt = 0.05 ms, "
                 "drawn two ways", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.002,
             f"Identical data in both panels. The original fitting script builds the "
             f"axis as np.linspace(0, 10, 1500), so {t2[-1] + DT_FILE:.0f} ms of "
             f"simulated activity is drawn on a 10 ms axis —\n"
             f"a factor of {(t2[-1] + DT_FILE) / 10:.1f}. "
             f"{n_spikes(v2)} action potentials are shown as though they "
             f"occurred in 10 ms.",
             ha="left", va="top", fontsize=7.5, color=C.MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    paths = save(fig, os.path.join(OUT, "hh_time_axis_defect"))
    print("wrote " + ", ".join(paths))

    with open(os.path.join(OUT, "hh_convergence_phase_plane.json"), "w") as f:
        json.dump({
            "regenerated_with": {"dt_ms": DT_FIG, "injected_current": I_FIG,
                                 "n_steps": N_STEPS,
                                 "span_ms": round(float(t[-1] + DT_FIG), 4)},
            "file_defaults": {"dt_ms": DT_FILE, "injected_current": I_FILE,
                              "span_ms": round(float(t2[-1] + DT_FILE), 4),
                              "action_potentials": n_spikes(v2)},
            "reference_conductances_mS_per_cm2": WT,
            "fitted_conductances_mS_per_cm2": CONV,
            "reference_peak_mV": round(float(v_t.max()), 4),
            "fitted_peak_mV": round(float(v_m.max()), 4),
            "reference_action_potentials": n_spikes(v_t),
            "n_range_reference": [round(float(n_t.min()), 4),
                                  round(float(n_t.max()), 4)],
            "hh_class_source_lines": list(span),
        }, f, indent=2)
    print(f"wrote {os.path.join(OUT, 'hh_convergence_phase_plane.json')}")


if __name__ == "__main__":
    main()
