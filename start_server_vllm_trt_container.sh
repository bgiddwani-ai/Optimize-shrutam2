#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
GPU=${GPU:-0}
IMAGE=${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:0.20.0}
CONTAINER_NAME=${CONTAINER_NAME:-shrutam2-vllm-trt}

# TensorRT 10 plans are tied to the TensorRT runtime used to build them.  The
# 0.20 image supplies TensorRT 10.10.0.31, matching the checked-in encoder
# plans, while the mounted vLLM virtualenv keeps the validated vLLM/torch stack.
exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "device=${GPU}" \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  -v "$ROOT:$ROOT" \
  -w "$ROOT" \
  -e PYTHONPATH="$ROOT/artifacts/trt_py" \
  -e ROOT="$ROOT" \
  -e GPU=0 \
  -e PORT="${PORT:-8092}" \
  -e RUNTIME=trt \
  -e TRT_ENGINE="${TRT_ENGINE:-$ROOT/artifacts/shrutam2_encoder_bf16.plan}" \
  -e MAX_FRONTEND_BATCH="${MAX_FRONTEND_BATCH:-32}" \
  -e BASE_FRONTEND_BATCH="${BASE_FRONTEND_BATCH:-32}" \
  -e LARGE_BATCH_QUEUE_THRESHOLD="${LARGE_BATCH_QUEUE_THRESHOLD:-48}" \
  -e LENGTH_BUCKET_LOOKAHEAD="${LENGTH_BUCKET_LOOKAHEAD:-256}" \
  -e BASE_FRONTEND_DELAY_MS="${BASE_FRONTEND_DELAY_MS:-5}" \
  -e MAX_FRONTEND_DELAY_MS="${MAX_FRONTEND_DELAY_MS:-50}" \
  -e MAX_INFLIGHT_DECODE="${MAX_INFLIGHT_DECODE:-512}" \
  -e MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}" \
  -e LLM_QUANTIZATION="${LLM_QUANTIZATION:-none}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}" \
  -e CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES:-}" \
  -e LOG="${LOG:-$ROOT/logs/server_vllm_trt.log}" \
  "$IMAGE" \
  bash -lc '
    mkdir -p "$ROOT/artifacts/trt_py"
    for package in /usr/local/lib/python3.12/dist-packages/tensorrt*; do
      ln -sfn "$package" "$ROOT/artifacts/trt_py/$(basename "$package")"
    done
    exec bash "$ROOT/start_server_vllm.sh"
  '
