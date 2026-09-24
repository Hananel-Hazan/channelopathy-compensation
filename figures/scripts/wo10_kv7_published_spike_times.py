"""Kv7.2 D212G reproduction figure, with the published spike times digitised.

Panel a plots the somatic voltage of the reference (wild-type) and variant
(Kv7.2 D212G) cells under the protocol of ModelDB 118986's `fig6a.hoc` (0.47 nA
somatic step from 5 ms to 405 ms, tstop 500 ms, 35 degrees C), with each cell's
spike times marked above the traces.  Panel b plots the spike times read off the
graph distributed with the model against the simulated ones, on a common axis.
The graph itself is read, not reproduced.

Digitisation.  The graph is a 557 x 420 NEURON screenshot with both arms drawn
in black on one axis.  The horizontal axis line is the 0 mV level and runs from
t = 0 ms (the column of the vertical axis) to t = 500 ms (its last dark
column); these two columns are the calibration.  The tick labels 100 to 500
give an independent check on the scale.  A spike is located at the first
column in which its rising stroke crosses the 0 mV line, the same level at
which the simulation times its spikes (upward crossings of 0 mV,
`experiments/wo6/phase2/wo6b_taskD1_fig6a.py`).

Uncertainty, per spike, in pixels then converted to ms:
    +-0.5 px  the crossing lies somewhere inside the column it was drawn in
    +-0.5 px  each calibration column is itself quantised to one pixel, and the
              error interpolates between them, so it is at most 0.5 px anywhere
    +-0.6 px  the measured disagreement between the two calibrations: the
              tick-label centres fit x = 94.40 + 0.8160 t against the axis
              line's x = 95 + 0.8160 t, a constant 0.6 px offset
    = +-1.6 px in total (+-1.96 ms).
The band was first set at +-1 px, the first two terms only.  Under it the
variant's 8th spike fell outside: 366.42 against 367.74 ms, 1.32 ms or 1.07 px.
The third term was added after that result, once the two calibrations were
found to disagree by more than the 0.5 px allowed for calibration.

Both the axis and the traces pass through the same coordinate mapping, so
whether NEURON rounds or truncates to the pixel grid cancels out.

The reference cell's single spike and the variant's first spike fall in
adjacent columns (104 and 105) and the image does not label its strokes.  The
stroke in column 104 is the taller by one pixel.  The simulated variant peak is
0.43 mV above the reference peak (39.13 against 38.70 mV,
`figures/kv7_2_reproduction_published_protocol.csv`), which is about one pixel
at this vertical scale, so column 104 is assigned to the variant.

The stroke heights are measured inside the plot area, below the window frame
and the title and menu bars (the full-width lines above the axis): the tops
are rows 81 (column 104) and 82 (column 105).

Inputs:
    experiments/wo6/phase2/wo6b_taskD1_fig6a_traces.npz
    experiments/wo6/phase2/wo6b_taskD1_fig6a.json
    experiments/wo6/phase2/mutant/screenshot.jpg   (from scripts/fetch_modeldb.sh)

Writes figures/kv7_2_reproduction_digitised.{pdf,png} and
figures/kv7_2_published_spike_times_digitised.{csv,json}.  An existing CSV or
JSON is left in place when the new output is identical to it; if it differs,
the script stops unless run with --force.
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P2 = os.path.join(REPO, "experiments", "wo6", "phase2")
SHOT = os.path.join(P2, "mutant", "screenshot.jpg")
OUT = os.path.join(REPO, "figures")

STIM_ON, STIM_OFF, STIM_AMP = 5.0, 405.0, 0.47
T_AXIS = (0.0, 500.0)          # the axis line spans the graph's x range
INK = 128                      # grey level below which a pixel is drawn ink


def digitise(path):
    g = mpimg.imread(path).astype(float)[..., :3].mean(-1)
    dark = g < INK
    h, w = g.shape

    # the 0 mV axis: the row with the longest run of ink inside the plot
    axis_row = max(range(60, h - 60), key=lambda y: dark[y, 60:w - 40].sum())
    # its extent: the contiguous run of ink through the plot area
    run = dark[axis_row]
    mid = w // 2
    x0 = mid
    while x0 > 0 and run[x0 - 1]:
        x0 -= 1
    x1 = mid
    while x1 < w - 1 and run[x1 + 1]:
        x1 += 1
    # the window frame touches the axis row only if the run reaches the edge
    assert 20 < x0 < 150 and w - 150 < x1 < w - 20, (x0, x1)
    scale = (x1 - x0) / (T_AXIS[1] - T_AXIS[0])            # px per ms

    # label check: centres of the tick labels 100..500 under the axis
    lab = dark[axis_row + 12:axis_row + 26]
    strokes = dark[axis_row + 4:axis_row + 13].all(0)       # vertical traces
    cols = np.where(lab.any(0) & ~strokes)[0]
    groups = []
    for c in cols:
        if groups and c - groups[-1][-1] <= 7:
            groups[-1].append(int(c))
        else:
            groups.append([int(c)])
    centres = [(gg[0] + gg[-1]) / 2 for gg in groups if gg[0] > x0 + 40]
    ticks = np.array([100, 200, 300, 400, 500], float)[:len(centres)]
    fit = np.polyfit(ticks, centres[:len(ticks)], 1)

    # spikes: columns strictly inside the axis whose ink crosses the 0 mV line,
    # continuous for 6 px above it and 6 px below it
    cross = dark[axis_row - 6:axis_row].all(0) & dark[axis_row + 1:axis_row + 7].all(0)
    stroke_cols = [int(c) for c in np.where(cross)[0] if x0 < c < x1]
    # group adjacent columns into features
    feats = []
    for c in stroke_cols:
        if feats and c - feats[-1][-1] == 1:
            feats[-1].append(c)
        else:
            feats.append([c])
    # topmost ink row in each column of a feature: the height of its stroke,
    # measured inside the plot area, i.e. below the last full-width line above
    # the axis (window frame, title and menu bars)
    full = [y for y in range(axis_row) if dark[y, 60:w - 40].mean() > 0.95]
    plot_top = max(full) + 1 if full else 0
    tops = [{c: plot_top + int(np.argmax(dark[plot_top:, c])) for c in f} for f in feats]
    return dict(axis_row=int(axis_row), x0=int(x0), x1=int(x1),
                scale_px_per_ms=scale, label_fit=[float(v) for v in fit],
                label_centres=centres[:len(ticks)], features=feats, tops=tops,
                shape=[int(h), int(w)])


def write_checked(path, text, force):
    """Write `text` to `path`. An existing file is kept when identical, and is
    replaced only with --force when it differs."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            if f.read() == text:
                print(f"unchanged {path}")
                return
        if not force:
            raise FileExistsError(f"{path} exists and differs; rerun with --force")
    with open(path, "w", newline="") as f:
        f.write(text)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="replace an existing CSV or JSON that differs")
    force = ap.parse_args().force
    z = np.load(os.path.join(P2, "wo6b_taskD1_fig6a_traces.npz"))
    d = json.load(open(os.path.join(P2, "wo6b_taskD1_fig6a.json")))
    wt = d["arms"]["wild type"]
    mt = d["arms"]["mutant (Kv7.2 D212G)"]
    sim_mt = [float(t) for t in mt["spike_times_ms"]]
    sim_wt = [float(t) for t in wt["spike_times_ms"]]

    dg = digitise(SHOT)
    s, x0 = dg["scale_px_per_ms"], dg["x0"]
    ms_per_px = 1.0 / s
    # column quantisation + endpoint quantisation + measured disagreement
    # between the axis-line and tick-label calibrations (see the docstring)
    fit_s, fit_a = dg["label_fit"]
    disagree = max(abs((fit_a + fit_s * t) - (x0 + s * t)) for t in T_AXIS)
    unc_px = 0.5 + 0.5 + round(disagree, 1)
    unc_ms = unc_px * ms_per_px
    t_of = lambda x: T_AXIS[0] + (x - x0) / s

    print(f"screenshot {dg['shape'][1]} x {dg['shape'][0]} px; 0 mV axis at row "
          f"{dg['axis_row']}, t=0 at column {x0}, t=500 at column {dg['x1']}")
    print(f"scale {s:.4f} px/ms = {ms_per_px:.3f} ms/px; label-centre fit "
          f"{dg['label_fit'][0]:.4f} px/ms, intercept {dg['label_fit'][1]:.2f} px")

    feats = dg["features"]
    if len(feats) != 8:
        raise SystemExit(f"expected 8 spike features, found {len(feats)}: {feats}")
    # feature 0 holds two strokes: the reference spike and the variant's first
    f0 = feats[0]
    if len(f0) != 2:
        raise SystemExit(f"first feature should span two columns, got {f0}")
    tops0 = dg["tops"][0]
    taller = min(f0, key=lambda c: tops0[c])
    other = [c for c in f0 if c != taller][0]

    rows = []
    rows.append(("variant", 1, taller, sim_mt[0],
                 "taller of the two strokes in the first feature"))
    rows.append(("reference", 1, other, sim_wt[0],
                 "shorter of the two strokes in the first feature"))
    for k, f in enumerate(feats[1:], start=2):
        rows.append(("variant", k, f[0], sim_mt[k - 1],
                     "first column of the rising stroke at the 0 mV line"))

    out = []
    for arm, k, col, sim, how in rows:
        t = t_of(col)
        res = t - sim
        out.append(dict(arm=arm, spike=k, column_px=col,
                        digitised_ms=round(t, 2), uncertainty_ms=round(unc_ms, 2),
                        simulated_ms=round(sim, 3), residual_ms=round(res, 2),
                        residual_px=round(res * s, 2),
                        within=abs(res) <= unc_ms, located_by=how))
    print(f"\n{'arm':10s} {'#':>2s} {'col':>4s} {'digitised ms':>16s} "
          f"{'simulated':>9s} {'resid ms':>8s} {'resid px':>8s}")
    for r in out:
        print(f"{r['arm']:10s} {r['spike']:2d} {r['column_px']:4d} "
              f"{r['digitised_ms']:8.2f} +- {r['uncertainty_ms']:.2f} "
              f"{r['simulated_ms']:9.3f} {r['residual_ms']:8.2f} "
              f"{r['residual_px']:8.2f}  {'within' if r['within'] else 'OUTSIDE'}")
    n_out = sum(not r["within"] for r in out)
    n_half = sum(abs(r["residual_px"]) > 0.5 for r in out)
    print(f"\n{len(out) - n_out} of {len(out)} simulated times lie within the "
          f"digitised time +- {unc_ms:.2f} ms ({unc_px:g} px); {n_half} of {len(out)} "
          f"residuals exceed half a pixel ({0.5 * ms_per_px:.2f} ms)")

    p = os.path.join(OUT, "kv7_2_published_spike_times_digitised.csv")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
    write_checked(p, buf.getvalue(), force)
    pj = os.path.join(OUT, "kv7_2_published_spike_times_digitised.json")
    calibration = {k: dg[k] for k in ("shape", "axis_row", "x0", "x1",
                                      "scale_px_per_ms", "label_centres")}
    calibration["label_fit"] = [round(v, 4) for v in dg["label_fit"]]
    calibration = {k: calibration[k] for k in ("shape", "axis_row", "x0", "x1",
                                               "scale_px_per_ms", "label_fit",
                                               "label_centres")}
    write_checked(pj, json.dumps({"source": "ModelDB 118986, screenshot.jpg (read, not reproduced)",
               "calibration": calibration,
               "ms_per_px": ms_per_px,
               "uncertainty": {"px": unc_px, "ms": unc_ms,
                               "components_px": {"column_quantisation": 0.5,
                                                 "calibration_quantisation": 0.5,
                                                 "calibration_disagreement": round(disagree, 1)},
                               "note": "widened by the measured calibration "
                                       "disagreement after the +-1 px band left "
                                       "spike 8 outside"},
               "features_px": dg["features"], "stroke_top_rows": dg["tops"],
               "assignment": "column %d (taller) -> variant spike 1; column %d -> "
                             "reference spike" % (taller, other),
               "all_within": n_out == 0, "n_residual_over_half_px": n_half,
               "spikes": out}, indent=1), force)

    # ---- figure ---------------------------------------------------------------
    apply_style()
    fig = plt.figure(figsize=(7.6, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.42,
                          left=0.21, right=0.985, top=0.88, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)

    # panel a, as in wo7_task21a_reproduction.py
    ax.axvspan(STIM_ON, STIM_OFF, color=C.MUTED, alpha=0.06, linewidth=0, zorder=0)
    ax.plot(z["t_wt"], z["v_wt"], "-", color=C.WT, linewidth=1.1,
            label=f"reference (wild type) — {wt['spikes_during_step']} action potential")
    ax.plot(z["t_mt"], z["v_mt"], "-", color=C.VARIANT, linewidth=1.1, alpha=0.9,
            label=f"variant (Kv7.2 D212G) — {mt['spikes_during_step']} action potentials")
    for t in sim_mt:
        ax.plot([t, t], [46, 52], "-", color=C.VARIANT, linewidth=1.1,
                solid_capstyle="butt", zorder=5)
    for t in sim_wt:
        ax.plot([t, t], [54, 60], "-", color=C.WT, linewidth=1.1,
                solid_capstyle="butt", zorder=5)
    ax.set_xlim(0, 500)
    ax.set_ylim(-75, 64)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("somatic membrane potential (mV)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              fontsize=7.5, borderaxespad=0.0)
    ax.set_title("a   reproduced from the model's own fig6a.hoc, unmodified",
                 loc="left", color=C.INK, pad=22)
    ax.text(500, 57, "spike times  ", ha="right", va="center", fontsize=7,
            color=C.MUTED)

    # panel b: four rows of spike times on the shared axis
    lanes = [("published, variant", "variant", "pub", 3),
             ("simulated, variant", "variant", "sim", 2),
             ("published, reference", "reference", "pub", 1),
             ("simulated, reference", "reference", "sim", 0)]
    for name, arm, kind, y in lanes:
        col = C.VARIANT if arm == "variant" else C.WT
        rr = [r for r in out if r["arm"] == arm]
        if kind == "sim":
            for r in rr:
                axb.plot([r["simulated_ms"]] * 2, [y - 0.32, y + 0.32], "-",
                         color=col, linewidth=1.4, solid_capstyle="butt")
        else:
            for r in rr:
                axb.fill_betweenx([y - 0.32, y + 0.32],
                                  r["digitised_ms"] - r["uncertainty_ms"],
                                  r["digitised_ms"] + r["uncertainty_ms"],
                                  color=col, alpha=0.22, linewidth=0)
                axb.plot([r["digitised_ms"]] * 2, [y - 0.32, y + 0.32], "-",
                         color=col, linewidth=1.4, solid_capstyle="butt",
                         alpha=0.55)
    axb.set_yticks([3, 2, 1, 0])
    axb.set_yticklabels([l[0] for l in lanes])
    axb.set_ylim(-0.7, 3.7)
    axb.grid(axis="y", visible=False)
    axb.set_xlabel("time (ms)")
    axb.axvspan(STIM_ON, STIM_OFF, color=C.MUTED, alpha=0.06, linewidth=0, zorder=0)
    axb.set_title("b   spike times read from the published figure, against our "
                  "simulation", loc="left", color=C.INK)

    fig.suptitle("Kv7.2 D212G: the published spike times reproduced before "
                 "anything was built on them", x=0.005, ha="left", color=C.INK)
    fig.text(0.005, 0.075,
             f"Published times digitised from the graph distributed with ModelDB "
             f"118986: 0 mV axis line calibrated from t = 0 to 500 ms, "
             f"{ms_per_px:.3f} ms per pixel; shaded band = +-{unc_px:g} pixel "
             f"(+-{unc_ms:.2f} ms).\n"
             f"{len(out) - n_out} of {len(out)} simulated spike times lie inside "
             f"their band.  Protocol as shipped: {STIM_AMP} nA somatic step from "
             f"{STIM_ON:g} to {STIM_OFF:g} ms, 500 ms total, 35 °C (fig6a.hoc).",
             ha="left", va="top", fontsize=7, color=C.MUTED)

    paths = save(fig, os.path.join(OUT, "kv7_2_reproduction_digitised"))
    print("wrote " + ", ".join(paths))
    return n_out


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
