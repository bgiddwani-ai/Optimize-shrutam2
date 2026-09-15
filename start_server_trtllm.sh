#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-trtllm:0.20}
ENGINE_DIR=${ENGINE_DIR:?set ENGINE_DIR to a TensorRT-LLM engine directory}
ENCODER_RUNTIME=${ENCODER_RUNTIME:-eager}
ENCODER_PRECISION=${ENCODER_PRECISION:-bf16}
TRT_ENGINE=${TRT_ENGINE:-}
PORT=${PORT:-8092}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-64}
MAX_DELAY_MS=${MAX_DELAY_MS:-12}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
NUM_BEAMS=${NUM_BEAMS:-1}
KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.55}
CONTAINER_NAME=${CONTAINER_NAME:-shrutam2-trtllm-server}

extra=()
if [[ "$ENCODER_RUNTIME" == "trt" ]]; then
  [[ -n "$TRT_ENGINE" ]] || { echo "TRT_ENGINE is required for encoder runtime trt" >&2; exit 2; }
  extra+=(--trt-engine "/workspace/artifacts/$(basename "$TRT_ENGINE")")
fi

exec docker run --rm --name "$CONTAINER_NAME" --gpus device=0 --ipc=host --network host \
  --shm-size=16g -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
  python server_trtllm.py \
    --model-dir /workspace/model \
    --engine-dir "/workspace/artifacts/$(basename "$ENGINE_DIR")" \
    --encoder-runtime "$ENCODER_RUNTIME" \
    --encoder-precision "$ENCODER_PRECISION" \
    --port "$PORT" --max-batch-size "$MAX_BATCH_SIZE" \
    --max-delay-ms "$MAX_DELAY_MS" --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-beams "$NUM_BEAMS" --kv-cache-fraction "$KV_CACHE_FRACTION" \
    "${extra[@]}"
