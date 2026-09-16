#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-runtime:25.02}
RUNTIME=${RUNTIME:-eager}
PRECISION=${PRECISION:-bf16}
PORT=${PORT:-8092}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
MAX_DELAY_MS=${MAX_DELAY_MS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
NUM_BEAMS=${NUM_BEAMS:-1}
WARMUP_BATCHES=${WARMUP_BATCHES:-"1 2 8 32"}

mkdir -p "${ROOT}/logs"
docker rm -f shrutam2-server >/dev/null 2>&1 || true
screen -S shrutam2_server -X quit >/dev/null 2>&1 || true

case "${RUNTIME}" in
  eager|compile) ;;
  *) echo "RUNTIME must be eager or compile" >&2; exit 2 ;;
esac

screen -L -Logfile "${ROOT}/logs/server_${RUNTIME}.log" -dmS shrutam2_server \
  docker run --rm --name shrutam2-server --gpus device=0 --ipc=host --network host \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/artifacts/torchinductor_cache \
  -v "${ROOT}:/workspace" -w /workspace "${IMAGE}" \
  python server.py \
    --model-dir /workspace/model \
    --runtime "${RUNTIME}" \
    --precision "${PRECISION}" \
    --port "${PORT}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --max-delay-ms "${MAX_DELAY_MS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-beams "${NUM_BEAMS}" \
    --warmup-batches ${WARMUP_BATCHES}

deadline=$((SECONDS + 3600))
container_grace=$((SECONDS + 30))
until curl -fsS "http://127.0.0.1:${PORT}/health" > "${ROOT}/logs/health_${RUNTIME}.json"; do
  if (( SECONDS >= deadline )); then
    echo "server did not become ready" >&2
    tail -200 "${ROOT}/logs/server_${RUNTIME}.log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= container_grace )) && ! docker ps --format '{{.Names}}' | grep -qx shrutam2-server; then
    echo "server container exited" >&2
    tail -200 "${ROOT}/logs/server_${RUNTIME}.log" >&2 || true
    exit 1
  fi
  sleep 5
done
cat "${ROOT}/logs/health_${RUNTIME}.json"
