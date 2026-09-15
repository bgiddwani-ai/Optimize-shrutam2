#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-runtime:25.02}
REVISION=${REVISION:-e249bba6f7319c27912847fbbebb4258ead3b848}
cd "$ROOT"
mkdir -p model data/fleurs_quick artifacts logs results

docker run --rm -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  huggingface-cli download bharatgenai/Shrutam-2 \
    --revision "$REVISION" --local-dir /workspace/model \
  2>&1 | tee logs/equivalent_model_download.log

test "$(stat -c %s model/model.pt)" -gt 1000000
test "$(stat -c %s model/encoder.pt)" -gt 1000000
if head -c 200 model/model.pt | grep -q 'git-lfs.github.com/spec'; then
  echo "model.pt is a Git LFS pointer" >&2
  exit 3
fi
sha256sum model/model.pt model/encoder.pt model/llm/model.safetensors \
  | tee logs/equivalent_model_sha256.txt

docker run --rm -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python export_vllm_checkpoint.py \
    --model-dir /workspace/model \
    --output-dir /workspace/artifacts/vllm_llm_bf16 \
  2>&1 | tee logs/equivalent_export_vllm_checkpoint.log

docker run --rm -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python prepare_fleurs_quick.py \
    --output-dir /workspace/data/fleurs_quick --samples-per-language 4 \
  2>&1 | tee logs/equivalent_fleurs_prep.log

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python export_encoder_onnx.py \
    --model-dir /workspace/model \
    --output /workspace/artifacts/shrutam2_encoder_equiv_bf16.onnx \
    --precision bf16 \
  2>&1 | tee logs/equivalent_encoder_bf16_onnx.log

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python export_encoder_onnx.py \
    --model-dir /workspace/model \
    --output /workspace/artifacts/shrutam2_encoder_equiv_fp32.onnx \
    --precision fp32 \
  2>&1 | tee logs/equivalent_encoder_fp32_onnx.log

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python prepare_encoder_calibration.py \
    --model-dir /workspace/model \
    --manifest /workspace/data/fleurs_quick/manifest.jsonl \
    --output /workspace/artifacts/encoder_equiv_fp8_calibration.npz \
  2>&1 | tee logs/equivalent_encoder_calibration.log

python3 - <<'PY'
import json
from pathlib import Path
manifest = Path("data/fleurs_quick/manifest.jsonl")
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
report = {
    "model_revision": "e249bba6f7319c27912847fbbebb4258ead3b848",
    "model_files": ["model.pt", "encoder.pt", "llm/model.safetensors"],
    "dataset": json.loads(Path("data/fleurs_quick/report.json").read_text()),
    "all_audio_exists": all((Path.cwd() / row["audio_filepath"].removeprefix("/workspace/")).is_file() for row in rows),
}
Path("artifacts/equivalent_setup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
