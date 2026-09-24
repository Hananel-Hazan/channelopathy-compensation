"""Build the NEURON working directory used by wo97_lib.py.

Copies ichanWT2005.mod and ichanR859C1.mod from Neuron_Test/R859C/ (placed there by
scripts/fetch_modeldb.sh), strips the two `VERBATIM return 0; ENDVERBATIM` no-op
blocks that NEURON 9 refuses to compile, copies Neuron_Test/R859C/neuron.hoc
unchanged into experiments/wo97/work/, and compiles the mechanisms with nrnivmodl.
The source files are not modified.
"""
import os, re, shutil, subprocess, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARCH = os.path.join(REPO, "Neuron_Test", "R859C")
EXP8 = ARCH
WORK = os.path.join(REPO, "experiments", "wo97", "work")


def strip_verbatim(s):
    return re.sub(r"[ \t]*VERBATIM\s*\n[ \t]*return 0;\s*\n[ \t]*ENDVERBATIM[ \t]*\n", "", s)


def main():
    os.makedirs(WORK, exist_ok=True)
    for m in ("ichanWT2005.mod", "ichanR859C1.mod"):
        src = open(os.path.join(EXP8, m)).read()
        open(os.path.join(WORK, m), "w").write(strip_verbatim(src))
    shutil.copy(os.path.join(EXP8, "neuron.hoc"), os.path.join(WORK, "neuron.hoc"))
    shutil.rmtree(os.path.join(WORK, "x86_64"), ignore_errors=True)
    subprocess.check_call([os.path.join(os.path.dirname(sys.executable), "nrnivmodl")],
                          cwd=WORK)
    print("built in", WORK)


if __name__ == "__main__":
    main()
