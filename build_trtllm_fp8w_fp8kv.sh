#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-trtllm:0.20}
cd "$ROOT"
mkdir -p artifacts logs

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" bash -lc '
set -euo pipefail
checkpoint=/workspace/artifacts/trtllm_equiv_fp8w_fp8kv_ckpt
engine=/workspace/artifacts/trtllm_equiv_fp8w_fp8kv_engine

if [[ ! -f "$checkpoint/config.json" ]]; then
  python /app/tensorrt_llm/examples/quantization/quantize.py \
    --model_dir /workspace/artifacts/vllm_llm_bf16 \
    --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
    --calib_dataset cnn_dailymail --calib_size 128 --batch_size 8 \
    --calib_max_seq_length 512 \
    --output_dir "$checkpoint"
fi

rm -rf "$engine"
trtllm-build \
  --checkpoint_dir "$checkpoint" --output_dir "$engine" \
  --max_batch_size 64 \
  --max_input_len 256 --max_seq_len 512 --max_beam_width 1 \
  --max_num_tokens 16384 \
  --max_prompt_embedding_table_size 16384 \
  --gpt_attention_plugin bfloat16 --gemm_plugin fp8 \
  --remove_input_padding enable --context_fmha enable \
  --use_paged_context_fmha enable --multiple_profiles enable --monitor_memory
' 2>&1 | tee logs/equivalent_trtllm_fp8w_fp8kv_build.log

python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts")
name = "trtllm_equiv_fp8w_fp8kv_engine"
config = json.loads((root / name / "config.json").read_text())
quant = config["pretrained_config"].get("quantization") or {}
path = root / "equivalent_trtllm_variants.json"
rows = json.loads(path.read_text()) if path.exists() else {}
rows[name] = {
    "version": config["version"],
    "dtype": config["pretrained_config"]["dtype"],
    "quant_algo": quant.get("quant_algo"),
    "kv_cache_quant_algo": quant.get("kv_cache_quant_algo"),
    "producer": config["pretrained_config"].get("producer"),
    "build_config": config["build_config"],
}
path.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows[name], indent=2))
PY

test "$(jq -r '.trtllm_equiv_fp8w_fp8kv_engine.quant_algo' artifacts/equivalent_trtllm_variants.json)" = FP8
test "$(jq -r '.trtllm_equiv_fp8w_fp8kv_engine.kv_cache_quant_algo' artifacts/equivalent_trtllm_variants.json)" = FP8
ls -lh artifacts/trtllm_equiv_fp8w_fp8kv_engine/rank0.engine
