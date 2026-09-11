#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
docker run --rm --gpus device=0 --ipc=host -v "${ROOT}:/workspace" \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 bash -lc '
set -euo pipefail
python -c "import tensorrt_llm,tensorrt,torch; print(\"tensorrt_llm\",tensorrt_llm.__version__); print(\"tensorrt\",tensorrt.__version__); print(\"torch\",torch.__version__,torch.version.cuda)"
command -v trtllm-build || true
command -v trtexec || true
find /app/tensorrt_llm/examples -type f \( -name convert_checkpoint.py -o -name quantize.py \) | sort | head -50
trtllm-build --help | sed -n "1,240p"
' 2>&1 | tee "${ROOT}/logs/trtllm_probe.log"
