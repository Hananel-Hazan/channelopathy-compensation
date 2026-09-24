#!/bin/bash
# Five independent launches of wo3_taskC2_clean.py (the Table 1 GPU benchmark) on
# the RTX 4070 Ti, each writing c2_clean_4070_rep<i>.log in this directory. Rep 1
# is timed first: if it takes longer than BUDGET seconds the remaining repeats are
# skipped. Set PY to choose the Python interpreter (default: python3).
cd "$(dirname "$0")"
PY=${PY:-python3}
BUDGET=7200   # seconds
echo "START $(date -Is) host=$(hostname) torch=$($PY -c "import torch;print(torch.__version__)")"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for i in 1 2 3 4 5; do
    t0=$(date +%s)
    $PY ../../wo3/wo3_taskC2_clean.py > "c2_clean_4070_rep${i}.log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    echo "rep${i} rc=${rc} elapsed=${dt}s finished $(date -Is)"
    if [ $rc -ne 0 ]; then echo "STOP: rep${i} failed (rc=${rc})"; exit 1; fi
    if [ $i -eq 1 ] && [ $dt -gt $BUDGET ]; then echo "STOP: rep1 exceeded budget"; exit 2; fi
done
echo "ALL DONE $(date -Is)"
