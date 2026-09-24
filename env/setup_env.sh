#!/usr/bin/env bash
# Create the self-contained conda environment "hh-compensation" for the GPU experiments.
# No sudo, no system-wide installs.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n hh-compensation python=3.11 -y
conda activate hh-compensation
# torch pinned to the version the GPU results were produced with
pip install --no-input torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-input numpy scipy pandas matplotlib scikit-learn
python - <<'PY'
import torch, sys, platform
print("host        :", platform.node())
print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__, "cuda", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu         :", p.name, f"{p.total_memory/2**30:.2f} GiB",
          f"{p.multi_processor_count} SMs", f"cc {p.major}.{p.minor}")
PY
