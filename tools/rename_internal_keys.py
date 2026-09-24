"""Rename two keys in data files whose generators cannot be rerun here.

The three wo6 files were written by experiments/wo6/wo6_taskC2.py, wo6_taskC5.py
and wo6_taskC5b.py, and the null-ensemble file by
figures/scripts/wo9_task31_null_ensemble_allvariants.py. All of them need a CUDA
graphics card, and the null ensemble takes hours per variant. The generators now
write the new names; this script applies the same renames to the shipped
outputs, so the files match what their generators write. No value changes.

    experiments/wo6/wo6_taskC2.json    label "(as recited)"  -> "(as reported)"
    experiments/wo6/wo6_taskC5.json    key   "as_recited"    -> "as_reported"
    experiments/wo6/wo6_taskC5b.json   key   "as_recited"    -> "as_reported"
    figures/wo9_dimensionality_null_ensemble_allvariants.json
                                       key   "g5"            -> "pooled_over_variants",
                                       and its note reworded

Each replacement is made on the file's text, so every other byte is kept, and
each must match the stated number of times. The script refuses to run on a file
whose md5 is not the recorded original, so it cannot be applied twice.

md5 of each file, before -> after:
    wo6_taskC2.json   0ce2fc4d303cc7bd905be7a0bfa45985 -> d510840b393557195c63eac77021c94d
    wo6_taskC5.json   550459dde973e1d777f6955aa469c526 -> b8000ec08dfa50bc06976e965634388a
    wo6_taskC5b.json  25d13cb18785413a4faba5ec511b6331 -> 56f120e3f64e1578c62269c6bd109fa8
    wo9_dimensionality_null_ensemble_allvariants.json
                      354653b2445101d9db982833a6da8867 -> 8e2ebf7bdcb271c7c537ee9d46eafcd6
"""
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

NEW_NOTE = ("Range of the ensemble members' medians, pooled over the four variant "
            "types, and whether each variant's own compensating-set median lies "
            "inside that range.")

# The renamed block is the last one in the null-ensemble file, so its note is
# the last "note" value in the file; LAST_NOTE stands for that value.
LAST_NOTE = object()

# (file, md5 before, md5 after, [(old text, new text, expected count)])
PLAN = [
    ("experiments/wo6/wo6_taskC2.json",
     "0ce2fc4d303cc7bd905be7a0bfa45985", "d510840b393557195c63eac77021c94d",
     [("(as recited)", "(as reported)", 1)]),
    ("experiments/wo6/wo6_taskC5.json",
     "550459dde973e1d777f6955aa469c526", "b8000ec08dfa50bc06976e965634388a",
     [('"as_recited"', '"as_reported"', 7)]),
    ("experiments/wo6/wo6_taskC5b.json",
     "25d13cb18785413a4faba5ec511b6331", "56f120e3f64e1578c62269c6bd109fa8",
     [('"as_recited"', '"as_reported"', 7)]),
    ("figures/wo9_dimensionality_null_ensemble_allvariants.json",
     "354653b2445101d9db982833a6da8867", "8e2ebf7bdcb271c7c537ee9d46eafcd6",
     [('"g5": {', '"pooled_over_variants": {', 1), (LAST_NOTE, NEW_NOTE, 1)]),
]


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    done = []
    for rel, before, after, subs in PLAN:
        path = os.path.join(REPO, rel)
        data = open(path, "rb").read()
        if md5(data) != before:
            sys.exit(f"{rel}: md5 {md5(data)} is not the recorded original {before}; "
                     f"nothing written")
        text = data.decode("utf-8")
        for old, new, n in subs:
            if old is LAST_NOTE:
                key = '"note": "'
                i = text.rindex(key) + len(key)
                j = text.index('"', i)
                if i < text.index('"pooled_over_variants": {'):
                    sys.exit(f"{rel}: last note is not in the renamed block; nothing written")
                text = text[:i] + new + text[j:]
                continue
            if text.count(old) != n:
                sys.exit(f"{rel}: {old!r} occurs {text.count(old)} times, expected {n}; "
                         f"nothing written")
            text = text.replace(old, new)
        out = text.encode("utf-8")
        if md5(out) != after:
            sys.exit(f"{rel}: result md5 {md5(out)} is not the recorded {after}; "
                     f"nothing written")
        done.append((path, out, rel, before, after))
    for path, out, rel, before, after in done:
        with open(path, "wb") as f:
            f.write(out)
        print(f"{rel}: {before} -> {after}")


if __name__ == "__main__":
    main()
