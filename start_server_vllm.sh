#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
GPU=${GPU:-0}
PORT=${PORT:-8092}
RUNTIME=${RUNTIME:-eager}
LOG=${LOG:-$ROOT/logs/server_vllm.log}

source "$ROOT/.venv-vllm/bin/activate"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/artifacts/torchinductor_vllm}

exec python server_vllm.py \
  --model-dir "$ROOT/model" \
  --decoder-dir "${DECODER_DIR:-$ROOT/artifacts/vllm_llm_bf16}" \
  --runtime "$RUNTIME" \
  --trt-engine "${TRT_ENGINE:-$ROOT/artifacts/shrutam2_encoder_bf16.plan}" \
  --port "$PORT" \
  --max-frontend-batch "${MAX_FRONTEND_BATCH:-64}" \
  --base-frontend-batch "${BASE_FRONTEND_BATCH:-32}" \
  --large-batch-queue-threshold "${LARGE_BATCH_QUEUE_THRESHOLD:-48}" \
  --length-bucket-lookahead "${LENGTH_BUCKET_LOOKAHEAD:-256}" \
  --base-frontend-delay-ms "${BASE_FRONTEND_DELAY_MS:-5}" \
  --max-frontend-delay-ms "${MAX_FRONTEND_DELAY_MS:-50}" \
  --max-inflight-decode "${MAX_INFLIGHT_DECODE:-512}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-96}" \
  --max-model-len "${MAX_MODEL_LEN:-1024}" \
  --max-num-seqs "${MAX_NUM_SEQS:-512}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.65}" \
  --llm-quantization "${LLM_QUANTIZATION:-none}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  ${CALCULATE_KV_SCALES:+--calculate-kv-scales} \
  2>&1 | tee "$LOG"
