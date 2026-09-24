#!/usr/bin/env bash
# Download the two third-party NEURON models this work builds on.
#
# Neither is redistributed in this repository. ModelDB publishes no blanket licence:
# copyright in each entry rests with whoever submitted it, so the models are fetched
# from source rather than copied here. Both are free to download.
#
#   ModelDB 87585   Barela et al. 2006, Nav1.1 R859C          -> Neuron_Test/R859C/
#   ModelDB 118986  Miceli et al. 2009, CA1 with Kv7.2 D212G  -> experiments/wo6/phase2/mutant/
#
# Usage:  bash scripts/fetch_modeldb.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

need() { command -v "$1" >/dev/null || { echo "need $1 on PATH" >&2; exit 1; }; }
need curl; need unzip

get() {  # accession -> extracted directory in $TMP
  local acc="$1"
  echo "downloading ModelDB $acc ..."
  curl -fsSL "https://modeldb.science/download/$acc" -o "$TMP/$acc.zip"
  mkdir -p "$TMP/$acc"; unzip -q "$TMP/$acc.zip" -d "$TMP/$acc"
}

place() {  # find a file by name anywhere under the extracted tree, copy it to a destination
  local acc="$1" name="$2" dest="$3"
  local src
  src="$(find "$TMP/$acc" -type f -name "$name" -print -quit)"
  if [ -z "$src" ]; then echo "  MISSING in $acc: $name" >&2; return 1; fi
  mkdir -p "$(dirname "$dest")"
  if [ -e "$dest" ]; then echo "  keeping existing $dest"; else cp "$src" "$dest"; echo "  $name"; fi
}

# ---------------------------------------------------------------- 87585
get 87585
echo "placing into Neuron_Test/R859C/ ..."
for f in README.TXT ichanWT2005.mod ichanR859C1.mod APthreshold.hoc mosinit.hoc displaytraces.hoc; do
  place 87585 "$f" "$ROOT/Neuron_Test/R859C/$f" || true
done

# ---------------------------------------------------------------- 118986
get 118986
echo "placing into experiments/wo6/phase2/mutant/ ..."
MUT="$ROOT/experiments/wo6/phase2/mutant"
mkdir -p "$MUT"
# the entry ships the whole model directory; take it wholesale
SRCDIR="$(dirname "$(find "$TMP/118986" -type f -name 'kmtwt.mod' -print -quit)")"
if [ -z "$SRCDIR" ] || [ "$SRCDIR" = "." ]; then
  echo "  could not locate kmtwt.mod inside the 118986 archive; inspect $TMP by hand" >&2; exit 1
fi
cp -n "$SRCDIR"/* "$MUT"/ 2>/dev/null || true
ls "$MUT"

cat <<'MSG'

Both models are in place. Next, compile the mechanisms:

    (cd Neuron_Test/R859C              && nrnivmodl)
    (cd experiments/wo6/phase2/mutant  && nrnivmodl)

Cite ModelDB and the two original papers; see NOTICE.
MSG
