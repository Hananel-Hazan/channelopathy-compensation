"""Across-run spread for Table 1's graphics-card throughput rows.

Table 1's RTX 2070 and RTX 4070 Ti rates come from single end-to-end timings.  This
script reads N independent launches of experiments/wo3/wo3_taskC2_clean.py per card
(the same script on both cards) and reports, for every implementation and batch size,
the median and the full min-max spread across launches.

It also runs a reproduction check: the median at each batch size Table 1 quotes must lie
within GATE_PCT of the value recorded in the original single run.  If any check fails the
script exits non-zero.  One cell is known to lie outside it, the RTX 4070 Ti fused batch of
16,384 sets (the cache-resident row): its single original run is above every one of the
five launches, and Table 1 reports the five-launch median with the full range.  For that
cell alone the script prints a note instead of failing (KNOWN_OUTSIDE_GATE).

Inputs (each log is the unmodified stdout of one launch):
    experiments/wo3/out_taskC2_clean.log                original RTX 2070 run
    experiments/wo3/out_taskC2_clean_rep{1..5}.log      RTX 2070 repeats
    experiments/wo5/bench/c2_clean_4070.log             original RTX 4070 Ti run
    experiments/wo5/bench/c2_clean_4070_rep{1..5}.log   RTX 4070 Ti repeats
Output:
    figures/throughput_repeats.csv   one row per (card, implementation, batch)
"""
import csv
import glob
import os
import re
import statistics
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "figures", "throughput_repeats.csv")
GATE_PCT = 5.0

CARDS = {
    "RTX 2070": dict(
        original=f"{REPO}/experiments/wo3/out_taskC2_clean.log",
        repeats=sorted(glob.glob(f"{REPO}/experiments/wo3/out_taskC2_clean_rep[0-9].log")),
        # (implementation, batch) that Table 1 quotes, and the value it quotes
        quoted={("unfused", 262144): 543, ("graph-captured", 1048576): 572,
                ("fused", 262144): 34350}),
    "RTX 4070 Ti": dict(
        original=f"{REPO}/experiments/wo5/bench/c2_clean_4070.log",
        repeats=sorted(glob.glob(f"{REPO}/experiments/wo5/bench/c2_clean_4070_rep[0-9].log")),
        quoted={("unfused", 65536): 2044, ("graph-captured", 16384): 2924,
                ("fused", 1048576): 63205, ("fused", 16384): 107013}),
}
# (card, (implementation, batch)) whose median is known to lie outside GATE_PCT of the
# single original run; Table 1 reports the median of the five launches for it.
KNOWN_OUTSIDE_GATE = {("RTX 4070 Ti", ("fused", 16384))}
IMPL = {"baseline": "unfused", "CUDA graph": "graph-captured", "torch.compile": "fused"}
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse(path):
    """-> {(impl, batch): sets_per_second}; raises if the log did not finish."""
    out, impl = {}, None
    text = open(path).read()
    if "DONE" not in text:
        raise RuntimeError(f"{path}: no DONE line, run did not finish")
    for line in text.splitlines():
        if line.startswith("---"):
            impl = next((v for k, v in IMPL.items() if k in line), None)
        m = ROW.match(line)
        if m and impl:
            out[(impl, int(m.group(1)))] = float(m.group(4))
    if len(out) != 21:
        raise RuntimeError(f"{path}: parsed {len(out)} cells, expected 21")
    return out


def main():
    rows, failed, noted = [], [], []
    for card, cfg in CARDS.items():
        orig = parse(cfg["original"])
        reps = [parse(p) for p in cfg["repeats"]]
        n = len(reps)
        print(f"\n{card}: {n} repeats  ({', '.join(os.path.basename(p) for p in cfg['repeats'])})")
        print(f"  {'implementation':<16}{'batch':>9}{'original':>10}{'median':>10}"
              f"{'min':>10}{'max':>10}{'spread%':>9}  quoted  gate")
        for key in sorted(orig, key=lambda k: (list(IMPL.values()).index(k[0]), k[1])):
            vals = [r[key] for r in reps]
            med = statistics.median(vals)
            lo, hi = min(vals), max(vals)
            spread = 100.0 * (hi - lo) / med
            quoted = cfg["quoted"].get(key)
            gate = ""
            if quoted is not None:
                dev = 100.0 * (med - orig[key]) / orig[key]
                ok = abs(dev) <= GATE_PCT
                if ok:
                    gate = f"PASS ({dev:+.1f}% vs original)"
                elif (card, key) in KNOWN_OUTSIDE_GATE:
                    gate = f"NOTE ({dev:+.1f}% vs original, see below)"
                    noted.append((card, key, orig[key], med, dev))
                else:
                    gate = f"FAIL ({dev:+.1f}% vs original)"
                    failed.append((card, key, orig[key], med))
            print(f"  {key[0]:<16}{key[1]:>9}{orig[key]:>10.1f}{med:>10.1f}{lo:>10.1f}"
                  f"{hi:>10.1f}{spread:>8.1f}%  {quoted or '':>7}  {gate}")
            rows.append(dict(card=card, implementation=key[0], batch=key[1],
                             table1_quoted=quoted or "", original_single_run=orig[key],
                             n_repeats=n, median=round(med, 1), min=round(lo, 1),
                             max=round(hi, 1), spread_pct_of_median=round(spread, 2),
                             repeats=" ".join(f"{v:.1f}" for v in vals)))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    for card, key, o, m, dev in noted:
        print(f"\nnote: {card}, {key[0]}, batch {key[1]:,}: the median of the five launches, "
              f"{m:,.1f}, differs by {abs(dev):.1f}% from the single original run, {o:,.1f}. "
              "Table 1 reports the median of the five launches, with their full range.")
    if failed:
        print("\nREPRODUCTION GATE FAILED -- do not update Table 1 from these runs:")
        for card, key, o, m in failed:
            print(f"  {card} {key}: original {o:.1f}, median now {m:.1f}")
        sys.exit(1)
    print("reproduction gate: every other quoted cell is within "
          f"{GATE_PCT:.0f}% of the original single run" if noted else
          "reproduction gate: all quoted cells within "
          f"{GATE_PCT:.0f}% of the original single run")


if __name__ == "__main__":
    main()
