#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
IMAGE=${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:0.20.0}
docker run --rm --gpus device=0 --ipc=host -v "${ROOT}:/workspace" -w /workspace \
  "${IMAGE}" bash -lc '
set -euo pipefail
python /app/tensorrt_llm/examples/quantization/quantize.py \
  --model_dir /workspace/model/llm \
  --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
  --calib_dataset cnn_dailymail --calib_size 64 --batch_size 4 \
  --calib_max_seq_length 512 \
  --output_dir /workspace/artifacts/trtllm_llm_modelopt_fp8_ckpt
trtllm-build \
  --checkpoint_dir /workspace/artifacts/trtllm_llm_modelopt_fp8_ckpt \
  --output_dir /workspace/artifacts/trtllm_llm_modelopt_fp8_engine \
  --max_batch_size 32 --max_input_len 256 --max_seq_len 512 \
  --max_beam_width 4 --max_num_tokens 8192 \
  --max_prompt_embedding_table_size 4096 \
  --gpt_attention_plugin bfloat16 --gemm_plugin fp8 \
  --remove_input_padding enable --context_fmha enable \
  --monitor_memory
'
