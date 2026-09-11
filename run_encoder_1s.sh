#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-runtime:25.02}
MANIFEST=/workspace/data/indicvoices_quick/manifest.jsonl
COMMON=(
  --rm --gpus device=0 --ipc=host
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/artifacts/torchinductor_cache
  -v /home/nvidia/indic_transcribe_bodhan:/home/nvidia/indic_transcribe_bodhan:ro
  -v "${ROOT}:/workspace" -w /workspace "${IMAGE}"
  python benchmark_encoder.py --model-dir /workspace/model --manifest "${MANIFEST}"
  --fixed-seconds 1 --batch-sizes 1 2 8 32 64 128 256
)

mkdir -p "${ROOT}/results/encoder" "${ROOT}/logs"
bash "${ROOT}/stop_server.sh"

docker run "${COMMON[@]}" --runtime eager \
  --output /workspace/results/encoder/eager_bf16_1s.csv \
  2>&1 | tee "${ROOT}/logs/encoder_eager_bf16_1s.log"

docker run "${COMMON[@]}" --runtime compile \
  --output /workspace/results/encoder/compile_bf16_1s.csv \
  2>&1 | tee "${ROOT}/logs/encoder_compile_bf16_1s.log"
