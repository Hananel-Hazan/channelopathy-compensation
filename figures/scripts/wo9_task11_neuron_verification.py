"""R859C slow-inactivation experiments run in NEURON itself.

Companion to figures/scripts/wo9_task11_slow_inactivation.py, which runs the same
single-compartment experiments in a from-scratch re-implementation without NEURON.  This
script compiles the Barela et al. (2006) mechanisms (ModelDB 87585; ichanWT2005.mod and
ichanR859C1.mod, placed in Neuron_Test/R859C/ by scripts/fetch_modeldb.sh) together with three
swapped variants built from copies of them:

  mt_wtstau   R859C with the wild-type slow-inactivation time constant (stau)
  mt_wtact    R859C with the wild-type activation curve (minf)
  wt_mtstau   wild type with the R859C slow-inactivation time constant

It runs the 35-level current ladder (20-360 pA) on each, checks the wild-type and R859C counts
against the archived traces in Neuron_Test/R859C/, and writes
figures/wo9_r859c_neuron_verification.json.

Requirements:
  NEURON (tested with 9.0.2), e.g. installed in a separate virtual environment with

      python3 -m venv /path/to/nrnvenv
      /path/to/nrnvenv/bin/pip install neuron

  Two things must be worked around, both mechanical:

  1. The 2007-vintage .mod files do not compile under NEURON 9.  `INITIAL` and
     `PROCEDURE states()` each contain a `VERBATIM return 0; ENDVERBATIM` block,
     which now generates `return 0;` inside a void C++ function.  The blocks are
     no-ops, and this script strips them from copies written to the work directory.
  2. If the interpreter comes from miniconda, its libstdc++ is older than the one
     nrnivmodl compiles against, and loading libnrnmech.so fails with
     "GLIBCXX_3.4.32 not found".  Run with
         LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

Usage:  LD_PRELOAD=... /path/to/nrnvenv/bin/python wo9_task11_neuron_verification.py --work DIR
"""
import argparse, json, os, re, shutil, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
OUT = os.path.join(REPO, "figures")

MT_MINF = "minf = 1/(1+exp(-(v+21.3)*3.5*0.03937))"
WT_MINF = "minf = 1/(1+exp(-(v+27.4)*4.7*0.03937))"
MT_STAU = "stau = 1000*(190.2*exp(-0.5*((v+90.4)/38.9)^2))"
WT_STAU = "stau = 1000*(140.4*exp(-0.5*((v+71.3)/30.9)^2))"


def strip_verbatim(s):
    return re.sub(r"[ \t]*VERBATIM\s*\n[ \t]*return 0;\s*\n[ \t]*ENDVERBATIM[ \t]*\n", "", s)


def build(work):
    os.makedirs(work, exist_ok=True)
    wt = strip_verbatim(open(os.path.join(ARCH, "ichanWT2005.mod")).read())
    mt = strip_verbatim(open(os.path.join(ARCH, "ichanR859C1.mod")).read())
    mech = {"ichanWT2005": wt, "ichanR859C1": mt}
    a = mt.replace("ichanR859C1", "mt_wtstau"); assert MT_STAU in a
    mech["mt_wtstau"] = a.replace(MT_STAU, WT_STAU)
    b = mt.replace("ichanR859C1", "mt_wtact"); assert MT_MINF in b
    mech["mt_wtact"] = b.replace(MT_MINF, WT_MINF)
    c = wt.replace("ichanWT2005", "wt_mtstau"); assert WT_STAU in c
    mech["wt_mtstau"] = c.replace(WT_STAU, MT_STAU)
    for name, src in mech.items():
        open(os.path.join(work, name + ".mod"), "w").write(src)
    shutil.rmtree(os.path.join(work, "x86_64"), ignore_errors=True)
    subprocess.check_call([os.path.join(os.path.dirname(sys.executable), "nrnivmodl")],
                          cwd=work)
    return list(mech)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch directory for the .mod copies")
    args = ap.parse_args()
    names = build(args.work)
    os.chdir(args.work)

    import numpy as np
    from neuron import h
    h.load_file("stdrun.hoc")

    def detector(V):
        """Spike count: binarize the trace at -10 mV and count level changes / 2.

        Transcribed from the original R859C search pipeline's spike detector."""
        v = np.atleast_2d(V).copy(); v[v < -10] = -10; v[v > -10] = 10
        return int((np.diff(v, axis=1) != 0).sum() / 2)

    def ladder(mech):
        """Spike counts for one mechanism on the protocol of APthreshold R859C.hoc.

        35 current steps of 20-360 pA (200 ms from t = 50 ms), 300 ms runs, dt 0.01 ms;
        the trace is decimated by 10 to match the archived logs."""
        out = []
        for I in range(20, 361, 10):
            s = h.Section()
            s.nseg, s.L, s.diam, s.Ra, s.cm = 1, 25, 25, 210, 1
            s.insert(mech)
            for seg in s:
                setattr(seg, "gnatbar_" + mech, 0.2)
                setattr(seg, "gkfbar_" + mech, 0.06)
                setattr(seg, "gl_" + mech, 0.0005)
                setattr(seg, "el_" + mech, -60)
            s.enat, s.ekf = 50, -80
            ic = h.IClamp(s(0.5)); ic.delay, ic.dur, ic.amp = 50, 200, I / 1000.0
            vec = h.Vector().record(s(0.5)._ref_v)
            h.secondorder = 2; h.dt = 0.01
            h.finitialize(-60); h.continuerun(300)
            out.append(detector(np.array(vec)[::10]))
        return np.array(out)

    def archived(fn):
        with open(os.path.join(ARCH, fn)) as f:
            f.readline(); f.readline()
            rows = [[float(p) for p in l.rstrip("\n").split("\t") if p != ""]
                    for l in f if l.strip()]
        a = np.array(rows)[:, 1:].T
        return np.array([detector(a[k]) for k in range(a.shape[0])])

    R = {m: ladder(m) for m in names}
    arch_wt = archived("APthreshold WT-2005 - log every 10th data point.txt")
    arch_mt = archived("APthreshold R859C - log every 10th data point.txt")
    wt, mt = R["ichanWT2005"], R["ichanR859C1"]

    rep = {
        "neuron_version": h.nrnversion(),
        "currents_pA": list(range(20, 361, 10)),
        "counts": {m: R[m].tolist() for m in names},
        "totals": {m: int(R[m].sum()) for m in names},
        "archive_check": {
            "archived_WT_total": int(arch_wt.sum()), "neuron_WT_total": int(wt.sum()),
            "WT_levels_matching": int((arch_wt == wt).sum()),
            "archived_MT_total": int(arch_mt.sum()), "neuron_MT_total": int(mt.sum()),
            "MT_levels_matching": int((arch_mt == mt).sum()),
            "exact": bool(np.array_equal(arch_wt, wt) and np.array_equal(arch_mt, mt)),
        },
        "summed_abs_difference_from_wild_type": {
            m: int(np.abs(wt - R[m]).sum()) for m in names},
        "swaps": {
            "R859C_vs_R859C_with_WT_slow_inactivation": {
                "levels_differing": int((mt != R["mt_wtstau"]).sum()),
                "levels_pA": [20 + 10 * int(i) for i in np.where(mt != R["mt_wtstau"])[0]],
                "total_before": int(mt.sum()), "total_after": int(R["mt_wtstau"].sum())},
            "wild_type_vs_wild_type_with_R859C_slow_inactivation": {
                "levels_differing": int((wt != R["wt_mtstau"]).sum()),
                "total_before": int(wt.sum()), "total_after": int(R["wt_mtstau"].sum())},
            "wild_type_vs_R859C_with_WT_activation": {
                "levels_differing": int((wt != R["mt_wtact"]).sum()),
                "total_before": int(wt.sum()), "total_after": int(R["mt_wtact"].sum())},
        },
    }
    path = os.path.join(OUT, "wo9_r859c_neuron_verification.json")
    json.dump(rep, open(path, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in
                      ("archive_check", "totals",
                       "summed_abs_difference_from_wild_type", "swaps")}, indent=2))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
