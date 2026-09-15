#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-trtllm:0.20}
MODEL_DIR=/workspace/artifacts/vllm_llm_bf16
ARTIFACTS=/workspace/artifacts

docker run --rm --gpus device=0 --ipc=host -v "$ROOT:/workspace" -w /workspace \
  "$IMAGE" bash -lc '
set -euo pipefail
rm -rf \
  /workspace/artifacts/trtllm_equiv_bf16_engine \
  /workspace/artifacts/trtllm_equiv_fp8w_bf16kv_engine \
  /workspace/artifacts/trtllm_equiv_bf16w_fp8kv_engine \
  /workspace/artifacts/trtllm_equiv_fp8w_fp8kv_engine

if [[ ! -f /workspace/artifacts/trtllm_equiv_bf16_ckpt/config.json ]]; then
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /workspace/artifacts/vllm_llm_bf16 \
  --output_dir /workspace/artifacts/trtllm_equiv_bf16_ckpt \
  --dtype bfloat16 --tp_size 1
fi

if [[ ! -f /workspace/artifacts/trtllm_equiv_fp8w_bf16kv_ckpt/config.json ]]; then
python /app/tensorrt_llm/examples/quantization/quantize.py \
  --model_dir /workspace/artifacts/vllm_llm_bf16 \
  --dtype bfloat16 --qformat fp8 \
  --calib_dataset cnn_dailymail --calib_size 128 --batch_size 8 \
  --calib_max_seq_length 512 \
  --output_dir /workspace/artifacts/trtllm_equiv_fp8w_bf16kv_ckpt
fi

if [[ ! -f /workspace/artifacts/trtllm_equiv_bf16w_fp8kv_ckpt/config.json ]]; then
python /app/tensorrt_llm/examples/quantization/quantize.py \
  --model_dir /workspace/artifacts/vllm_llm_bf16 \
  --dtype bfloat16 --qformat full_prec --kv_cache_dtype fp8 \
  --calib_dataset cnn_dailymail --calib_size 128 --batch_size 8 \
  --calib_max_seq_length 512 \
  --output_dir /workspace/artifacts/trtllm_equiv_bf16w_fp8kv_ckpt
fi

if [[ ! -f /workspace/artifacts/trtllm_equiv_fp8w_fp8kv_ckpt/config.json ]]; then
python /app/tensorrt_llm/examples/quantization/quantize.py \
  --model_dir /workspace/artifacts/vllm_llm_bf16 \
  --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
  --calib_dataset cnn_dailymail --calib_size 128 --batch_size 8 \
  --calib_max_seq_length 512 \
  --output_dir /workspace/artifacts/trtllm_equiv_fp8w_fp8kv_ckpt
fi

build_one() {
  checkpoint=$1
  output=$2
  gemm=$3
  # This value is global across the batch: 64 sequences x up to 256
  # projected audio tokens avoids a per-sequence cap of only 64 tokens.
  trtllm-build \
    --checkpoint_dir "$checkpoint" --output_dir "$output" \
    --max_batch_size 64 \
    --max_input_len 256 --max_seq_len 512 --max_beam_width 1 \
    --max_num_tokens 16384 \
    --max_prompt_embedding_table_size 16384 \
    --gpt_attention_plugin bfloat16 --gemm_plugin "$gemm" \
    --remove_input_padding enable --context_fmha enable \
    --use_paged_context_fmha enable --multiple_profiles enable --monitor_memory
}
build_one /workspace/artifacts/trtllm_equiv_bf16_ckpt \
  /workspace/artifacts/trtllm_equiv_bf16_engine bfloat16
build_one /workspace/artifacts/trtllm_equiv_fp8w_bf16kv_ckpt \
  /workspace/artifacts/trtllm_equiv_fp8w_bf16kv_engine fp8
build_one /workspace/artifacts/trtllm_equiv_bf16w_fp8kv_ckpt \
  /workspace/artifacts/trtllm_equiv_bf16w_fp8kv_engine bfloat16
build_one /workspace/artifacts/trtllm_equiv_fp8w_fp8kv_ckpt \
  /workspace/artifacts/trtllm_equiv_fp8w_fp8kv_engine fp8
'

python3 - <<'PY'
import json
from pathlib import Path
root = Path("artifacts")
rows = {}
for name in (
    "trtllm_equiv_bf16_engine",
    "trtllm_equiv_fp8w_bf16kv_engine",
    "trtllm_equiv_bf16w_fp8kv_engine",
    "trtllm_equiv_fp8w_fp8kv_engine",
):
    config = json.loads((root / name / "config.json").read_text())
    quant = config["pretrained_config"].get("quantization") or {}
    rows[name] = {
        "version": config["version"],
        "dtype": config["pretrained_config"]["dtype"],
        "quant_algo": quant.get("quant_algo"),
        "kv_cache_quant_algo": quant.get("kv_cache_quant_algo"),
        "producer": config["pretrained_config"].get("producer"),
        "build_config": config["build_config"],
    }
Path("artifacts/equivalent_trtllm_variants.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
