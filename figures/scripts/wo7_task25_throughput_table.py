"""Build the throughput table (Table 1).

One row per configuration: the non-differentiable NEURON pathway per processor
core and per 72-core node, the differentiable pathway on one processor core
(unbatched and batched), and the differentiable pathway on two graphics cards
(RTX 2070 and RTX 4070 Ti) unfused, graph-captured and fused; plus the
memory-traffic figures before and after fusing the integration step.  The unit
is stated with the table (see UNIT).  Only measured values are given; nothing is
extrapolated to other hardware, because throughput does not track rated
arithmetic throughput across the two card generations.

The seven graphics-card rates are medians over five independent launches of
experiments/wo3/wo3_taskC2_clean.py per card, with the min-max range across
launches in the spread column.  They are read from figures/throughput_repeats.csv
(written by figures/scripts/wo9_task4_repeats.py) and checked against the values
listed in ROWS.  The five processor-core and NEURON rows are single measurements
and have an empty spread.

Writes, in figures/:
    throughput_table.csv   the data, one row per configuration, with sources
    throughput_table.md    Markdown table with unit, notes and caveats
    throughput_table.pdf   rendered table, plus .png
"""
import csv
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo7_style import apply_style, C, save

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures")
REPEATS = os.path.join(OUT, "throughput_repeats.csv")

UNIT = ("one candidate parameter set = one conductance triple simulated across "
        "all 35 injected-current levels, 300 ms each, at a 0.01 ms integration step")

# pathway, hardware, implementation, sets/s (display), sets/s (numeric), source,
# spread (min - max across five independent launches; empty = single measurement)
ROWS = [
    ("Non-differentiable (NEURON)", "one processor core", "—",
     "1.1", 1.1,
     "NEURON single-core timing log of the R859C model (1.00 s per set); "
     "the single-core NEURON figure is 250,000 sets in 226,050 s", ""),
    ("Non-differentiable (NEURON)", "72-core node", "spike-count stage",
     "16.4", 16.4,
     "R859C search log: 1,437,450 evaluations in the spike-count stage's 24 h 17 m", ""),
    ("Non-differentiable (NEURON)", "72-core node", "time-warping stage",
     "16.9", 16.9,
     "same search log: 560,230 evaluations in 9 h 12 m", ""),

    ("Differentiable", "one processor core", "unbatched (1 set at a time)",
     "0.089", 0.089, "experiments/taskD_cpu_big.py (log: experiments/out_taskD_cpu_big_py.log)", ""),
    ("Differentiable", "one processor core", "batched (1,024 sets)",
     "20.8", 20.8, "experiments/taskD_cpu_big.py (log: experiments/out_taskD_cpu_big_py.log)", ""),

    ("Differentiable", "RTX 2070 (2018)", "batched, unfused",
     "543", 542.9,
     "median of 5 launches at 262,144 sets per batch (first single run: 542.9)",
     "542.8 – 543.0"),
    ("Differentiable", "RTX 2070 (2018)", "batched, graph-captured",
     "572", 571.7,
     "median of 5 launches at 1,048,576 sets per batch (first single run: 571.7)",
     "571.7 – 571.7"),
    ("Differentiable", "RTX 2070 (2018)", "batched, fused",
     "34,279", 34279.4,
     "median of 5 launches at 262,144 sets per batch (first single run: 34,350)",
     "34,275 – 34,288"),

    ("Differentiable", "RTX 4070 Ti (2023)", "batched, unfused",
     "2,053", 2052.9,
     "median of 5 launches at 65,536 sets per batch, the peak (first single run: 2,044)",
     "2,046 – 2,057"),
    ("Differentiable", "RTX 4070 Ti (2023)", "batched, graph-captured",
     "2,806", 2805.6,
     "median of 5 launches at 16,384 sets per batch, the peak (first single run: 2,924)",
     "2,790 – 2,809"),
    ("Differentiable", "RTX 4070 Ti (2023)", "batched, fused",
     "63,649", 63649.2,
     "median of 5 launches at 1,048,576 sets per batch (first single run: 63,205)",
     "63,633 – 63,651"),
    ("Differentiable", "RTX 4070 Ti (2023)",
     "batched, fused, cache-resident batch",
     "100,443", 100443.4,
     "median of 5 launches at 16,384 sets per batch — the working set (17.5 MiB) "
     "fits the 48 MiB level-two cache (first single run: 107,013, above "
     "every one of the five repeats)",
     "98,001 – 104,926"),
]

# The seven graphics-card rows above are recomputed at run time from the
# per-launch data in figures/throughput_repeats.csv (written by
# figures/scripts/wo9_task4_repeats.py); the literals in ROWS serve only as the
# expected values, and any disagreement aborts.  In that file, the row a Table 1
# rate comes from is the one whose `table1_quoted` column is non-empty.  The key
# below maps each table row to (card, implementation, batch size); batch size is
# needed because the RTX 4070 Ti contributes two fused rows, the 1,048,576
# production batch and the 16,384 cache-resident one.

TABLE1_SOURCE_ROWS = {
    ("RTX 2070 (2018)", "batched, unfused"):                     ("RTX 2070",    "unfused",        "262144"),
    ("RTX 2070 (2018)", "batched, graph-captured"):              ("RTX 2070",    "graph-captured", "1048576"),
    ("RTX 2070 (2018)", "batched, fused"):                       ("RTX 2070",    "fused",          "262144"),
    ("RTX 4070 Ti (2023)", "batched, unfused"):                  ("RTX 4070 Ti", "unfused",        "65536"),
    ("RTX 4070 Ti (2023)", "batched, graph-captured"):           ("RTX 4070 Ti", "graph-captured", "16384"),
    ("RTX 4070 Ti (2023)", "batched, fused"):                    ("RTX 4070 Ti", "fused",          "1048576"),
    ("RTX 4070 Ti (2023)", "batched, fused, cache-resident batch"): ("RTX 4070 Ti", "fused",        "16384"),
}


def _r(v):
    """Round half up, so the result does not depend on Python's banker's rounding."""
    return int(math.floor(float(v) + 0.5))


def _display(median):
    """The rate as Table 1 prints it: nearest whole set per second, grouped."""
    return f"{_r(median):,}"


def _spread(mn, mx, median):
    """The launch range as Table 1 prints it: one decimal below 1,000, whole above."""
    if median < 1000:
        return f"{float(mn):.1f} – {float(mx):.1f}"
    return f"{_r(mn):,} – {_r(mx):,}"


def _measured():
    """Read the per-launch benchmark file and return the seven Table 1 rows."""
    if not os.path.exists(REPEATS):
        raise SystemExit(
            f"wo7_task25_throughput_table.py: cannot find {REPEATS}.\n"
            "The seven graphics-card rows of Table 1 are medians over five launches "
            "and are read from that file, not typed into this script. Run "
            "figures/scripts/wo9_task4_repeats.py to create it.")
    out, marked = {}, 0
    with open(REPEATS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row.get("table1_quoted") or "").strip():
                continue
            marked += 1
            out[(row["card"], row["implementation"], row["batch"])] = (
                float(row["median"]), float(row["min"]), float(row["max"]))
    if marked != 7:
        raise SystemExit(
            f"wo7_task25_throughput_table.py: {REPEATS} marks {marked} rows as quoted "
            "in Table 1, expected 7. Table 1 has seven graphics-card rates; the marker "
            "column and the table have gone out of step.")
    return out


def resolve_rows():
    """ROWS with the seven graphics-card rates recomputed from the launch data.

    Every recomputed value is checked against the literal it replaces, so this
    cannot silently change what Table 1 prints -- it can only refuse to run.
    """
    measured, resolved, checked = _measured(), [], 0
    for r in ROWS:
        key = (r[1], r[2])
        if key not in TABLE1_SOURCE_ROWS:
            resolved.append(r)          # the five processor-core and NEURON rows
            continue                    # are single measurements with no data file
        ident = TABLE1_SOURCE_ROWS[key]
        if ident not in measured:
            raise SystemExit(
                f"wo7_task25_throughput_table.py: {REPEATS} has no row marked "
                f"table1_quoted for {ident}, which Table 1 needs for "
                f"{r[1]} / {r[2]}.")
        median, mn, mx = measured[ident]
        got = (_display(median), median, _spread(mn, mx, median))
        want = (r[3], r[4], r[6])
        if got != want:
            raise SystemExit(
                "wo7_task25_throughput_table.py: Table 1 and the launch data "
                f"disagree for {r[1]} / {r[2]} (batch {ident[2]}).\n"
                f"  this script says : {want}\n"
                f"  {os.path.basename(REPEATS)} says : {got}\n"
                "Nothing was written. Fix whichever is wrong before rebuilding "
                "Table 1; the paper reports these numbers.")
        resolved.append(r[:3] + (got[0], got[1]) + (r[5], got[2]))
        checked += 1
    if checked != 7:
        raise SystemExit(
            f"wo7_task25_throughput_table.py: recomputed {checked} graphics-card "
            "rows, expected 7.")
    print(f"  {checked} graphics-card rates recomputed from "
          f"{os.path.basename(REPEATS)} and matched the table")
    return resolved


# stage, card, demanded GB/s, measured device GB/s, share, source
# RTX 2070 values: experiments/wo3/out_taskC_profile.log (copy bandwidth, line 3;
# demanded traffic before fusion, lines 10-11; after fusion, lines 19-21), written
# by wo3_taskC1_bandwidth.py and wo3_taskC2_profile.py from the kernel times in
# experiments/wo3/out_taskC1.log.  The RTX 4070 Ti row has no log in this
# repository; see docs/REPRODUCE.md.
TRAFFIC = [
    ("Before fusion", "RTX 2070", "412 – 420", "393.6", "105 % – 107 %",
     "measured at 65,536 and 262,144 sets per batch"),
    ("After fusion", "RTX 2070", "51 – 64", "393.6", "13 % – 16 %", "measured"),
    ("After fusion", "RTX 4070 Ti", "116", "426.9", "27 %", "measured"),
]

CAVEATS = [
    "The non-differentiable rate includes the summary statistics computed for each "
    "parameter set; the differentiable rate is simulation only. The comparison "
    "therefore flatters the differentiable pathway.",
    "Hardware is not normalised: 2018 and 2023 consumer graphics cards against 2020 "
    "server processor cores.",
    "The biophysics differs slightly: the non-differentiable mechanism carries four "
    "state variables against three in the differentiable model.",
    "Without batching the advantage reverses — one processor core running the "
    "differentiable model one parameter set at a time is 12.5 times slower "
    "than the same core running NEURON.",
    "The seven graphics-card rates are medians of five independent launches of the "
    "same benchmark, with the full range across launches in the spread column; the "
    "five processor-core rates are single measurements. At the batch sizes quoted the "
    "range is below 0.6 % of the median except for the cache-resident row, where "
    "it is 6.9 %.",
    "No figure here should be extrapolated to other hardware. Across the two card "
    "generations measured, fused throughput rose by a factor of 1.86 while the "
    "cards' rated single-precision arithmetic throughput differs by a factor of "
    "5.37, so throughput does not track the rating.",
]


def main():
    print(f"unit: {UNIT}\n")
    rows = resolve_rows()
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"  {r[0]:<{w}}  {r[1]:<22}  {r[2]:<32}  {r[3]:>8} sets/s")
    print()
    for t in TRAFFIC:
        print(f"  {t[0]:<14} {t[1]:<14} demanded {t[2]:>10} GB/s   "
              f"device {t[3]:>6} GB/s   {t[4]:>12}")

    # ---- CSV -----------------------------------------------------------------
    p = os.path.join(OUT, "throughput_table.csv")
    with open(p, "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["table", "pathway", "hardware", "implementation",
                    "candidate_parameter_sets_per_second",
                    "spread_min_max_across_5_launches", "source"])
        for r in rows:
            c.writerow(["throughput", r[0], r[1], r[2], r[4], r[6], r[5]])
        c.writerow([])
        c.writerow(["table", "stage", "hardware", "memory_traffic_demanded_GB_per_s",
                    "device_bandwidth_measured_GB_per_s", "share_of_device_bandwidth",
                    "source"])
        for t in TRAFFIC:
            c.writerow(["memory traffic", t[0], t[1], t[2], t[3], t[4], t[5]])
    print(f"\nwrote {p}")

    # ---- Markdown ------------------------------------------------------------
    p = os.path.join(OUT, "throughput_table.md")
    with open(p, "w") as f:
        f.write("# Throughput of the two pathways\n\n")
        f.write(f"**Unit.** {UNIT[0].upper()}{UNIT[1:]}.\n\n")
        f.write("| Pathway | Hardware | Implementation | Candidate parameter "
                "sets per second | Range across 5 launches |\n|---|---|---|---:|---:|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[6] or '—'} |\n")
        f.write("\n## Memory traffic, before and after fusing the integration step\n\n")
        f.write("| Stage | Hardware | Traffic demanded (GB/s) | Device bandwidth, "
                "measured (GB/s) | Share of device bandwidth |\n|---|---|---:|---:|---:|\n")
        for t in TRAFFIC:
            f.write(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} |\n")
        f.write("\nBefore fusion the loop demands slightly more bandwidth than the "
                "card can supply, so it is bound by memory traffic rather than by "
                "arithmetic or capacity: each integration step issues 76 separate "
                "kernels, every one of which reads its operands from device memory "
                "and writes its result back. Fusing the step so those intermediates "
                "are never committed to memory removes that bound and raises "
                "throughput on the RTX 2070 from 543 to 34,279 sets per second, a "
                "factor of 63.\n")
        f.write("\n## Caveats that travel with these figures\n\n")
        for i, c_ in enumerate(CAVEATS, 1):
            f.write(f"{i}. {c_}\n")
        f.write("\n*Sources for every value are in `throughput_table.csv`.*\n")
    print(f"wrote {p}")

    # ---- rendered table ------------------------------------------------------
    apply_style()
    fig = plt.figure(figsize=(9.6, 6.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()

    y = 0.965
    ax.text(0.012, y, "Throughput of the two pathways", fontsize=11.5,
            color=C.INK, va="top")
    y -= 0.045
    ax.text(0.012, y, f"Unit: {UNIT}.", fontsize=7.6, color=C.MUTED, va="top",
            wrap=True)
    y -= 0.042

    cols = (0.012, 0.245, 0.475, 0.800, 0.985)
    ax.text(cols[0], y, "Pathway", fontsize=8.2, color=C.INK, va="top")
    ax.text(cols[1], y, "Hardware", fontsize=8.2, color=C.INK, va="top")
    ax.text(cols[2], y, "Implementation", fontsize=8.2, color=C.INK, va="top")
    ax.text(cols[3], y, "sets / second", fontsize=8.2, color=C.INK, va="top",
            ha="right")
    ax.text(cols[4], y, "range, 5 launches", fontsize=8.2, color=C.INK, va="top",
            ha="right")
    y -= 0.018
    ax.plot([0.012, 0.985], [y, y], color=C.MUTED, linewidth=0.9)
    y -= 0.026

    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            ax.plot([0.012, 0.985], [y + 0.012, y + 0.012], color=C.GRID,
                    linewidth=0.7)
            y -= 0.008
        ax.text(cols[0], y, r[0] if r[0] != prev else "", fontsize=7.8,
                color=C.INK, va="top")
        ax.text(cols[1], y, r[1], fontsize=7.8, color=C.INK, va="top")
        ax.text(cols[2], y, r[2], fontsize=7.8, color=C.INK, va="top")
        ax.text(cols[3], y, r[3], fontsize=8.4, color=C.INK, va="top",
                ha="right", fontweight="bold")
        ax.text(cols[4], y, r[6] or "—", fontsize=7.4, color=C.MUTED, va="top",
                ha="right")
        prev = r[0]
        y -= 0.031

    y -= 0.012
    ax.plot([0.012, 0.985], [y, y], color=C.MUTED, linewidth=0.9)
    y -= 0.036
    ax.text(0.012, y, "Memory traffic, before and after fusing the integration step",
            fontsize=9.2, color=C.INK, va="top")
    y -= 0.032
    tc = (0.012, 0.175, 0.400, 0.680, 0.985)
    for lbl, x, ha in (("Stage", tc[0], "left"), ("Hardware", tc[1], "left"),
                       ("demanded (GB/s)", tc[2], "right"),
                       ("device, measured (GB/s)", tc[3], "right"),
                       ("share", tc[4], "right")):
        ax.text(x, y, lbl, fontsize=7.8, color=C.INK, va="top", ha=ha)
    y -= 0.016
    ax.plot([0.012, 0.985], [y, y], color=C.MUTED, linewidth=0.9)
    y -= 0.026
    for t in TRAFFIC:
        ax.text(tc[0], y, t[0], fontsize=7.6, color=C.INK, va="top")
        ax.text(tc[1], y, t[1], fontsize=7.6, color=C.INK, va="top")
        ax.text(tc[2], y, t[2], fontsize=7.6, color=C.INK, va="top", ha="right")
        ax.text(tc[3], y, t[3], fontsize=7.6, color=C.INK, va="top", ha="right")
        ax.text(tc[4], y, t[4], fontsize=7.6, color=C.INK, va="top", ha="right")
        y -= 0.029

    y -= 0.014
    ax.plot([0.012, 0.985], [y, y], color=C.GRID, linewidth=0.7)
    y -= 0.030
    for i, c_ in enumerate(CAVEATS, 1):
        txt = c_ if len(c_) <= 118 else c_
        words, line, lines = txt.split(), "", []
        for wd in words:
            if len(line) + len(wd) + 1 > 118:
                lines.append(line); line = wd
            else:
                line = (line + " " + wd).strip()
        lines.append(line)
        for j, ln in enumerate(lines):
            ax.text(0.012 if j == 0 else 0.028, y,
                    (f"{i}.  " if j == 0 else "") + ln,
                    fontsize=6.9, color=C.MUTED, va="top")
            y -= 0.021
        y -= 0.004

    paths = save(fig, os.path.join(OUT, "throughput_table"))
    print("wrote " + ", ".join(paths))


if __name__ == "__main__":
    main()
