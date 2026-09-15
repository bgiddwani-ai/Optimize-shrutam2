#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-trtllm:0.20}
cd "$ROOT"

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" python quantize_encoder_modelopt.py \
    --onnx /workspace/artifacts/shrutam2_encoder_equiv_fp32.onnx \
    --calibration /workspace/artifacts/encoder_equiv_fp8_calibration.npz \
    --output /workspace/artifacts/shrutam2_encoder_modelopt_fp8.onnx \
    --log /workspace/logs/equivalent_encoder_modelopt_fp8.log \
  2>&1 | tee logs/equivalent_encoder_modelopt_fp8_console.log

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" bash -lc '
set -euo pipefail
ROOT=/workspace \
ONNX=/workspace/artifacts/shrutam2_encoder_equiv_bf16.onnx \
ENGINE=/workspace/artifacts/shrutam2_encoder_equiv_bf16.plan \
CACHE=/workspace/artifacts/shrutam2_encoder_equiv_bf16.timing.cache \
TRTEXEC=/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
PRECISION=bf16 OPT_BATCH=32 MAX_BATCH=64 MAX_MEL_FRAMES=2001 \
bash build_trt_encoder.sh
' 2>&1 | tee logs/equivalent_encoder_trt_bf16_build.log

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" bash -lc '
set -euo pipefail
ROOT=/workspace \
ONNX=/workspace/artifacts/shrutam2_encoder_modelopt_fp8.onnx \
ENGINE=/workspace/artifacts/shrutam2_encoder_modelopt_fp8.plan \
CACHE=/workspace/artifacts/shrutam2_encoder_modelopt_fp8.timing.cache \
TRTEXEC=/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
OPT_BATCH=32 MAX_BATCH=64 MAX_MEL_FRAMES=2001 \
bash build_trt_encoder_fp8.sh
' 2>&1 | tee logs/equivalent_encoder_trt_fp8_build.log

python3 - <<'PY'
import hashlib
import json
from pathlib import Path
rows = {}
for name in ("shrutam2_encoder_equiv_bf16.plan", "shrutam2_encoder_modelopt_fp8.plan"):
    path = Path("artifacts") / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows[name] = {"bytes": path.stat().st_size, "sha256": digest}
Path("artifacts/equivalent_encoder_engines.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
