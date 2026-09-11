#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
VARIANT=${VARIANT:-eager_bf16}
MANIFEST=${MANIFEST:-${ROOT}/data/indicvoices_quick/manifest.jsonl}
URL=${URL:-http://127.0.0.1:8092}

screen -S shrutam2_gpu_monitor -X quit >/dev/null 2>&1 || true
screen -L -Logfile "${ROOT}/logs/gpu_${VARIANT}.csv" -dmS shrutam2_gpu_monitor \
  nvidia-smi --id=0 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv -l 1
trap 'screen -S shrutam2_gpu_monitor -X quit >/dev/null 2>&1 || true' EXIT

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" \
  --manifest "${MANIFEST/#${ROOT}/\/workspace}" \
  --output-dir "/workspace/results/${VARIANT}/variable" \
  --concurrency 1 2 8 32 64 128 256 \
  --min-requests 32 \
  --rounds 1 2>&1 | tee "${ROOT}/logs/benchmark_${VARIANT}_variable.log"

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" \
  --manifest "${MANIFEST/#${ROOT}/\/workspace}" \
  --output-dir "/workspace/results/${VARIANT}/chunk_1s" \
  --fixed-seconds 1 \
  --concurrency 1 2 8 32 64 128 256 512 \
  --min-requests 64 \
  --rounds 1 2>&1 | tee "${ROOT}/logs/benchmark_${VARIANT}_chunk_1s.log"

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" \
  --manifest "${MANIFEST/#${ROOT}/\/workspace}" \
  --output-dir "/workspace/results/${VARIANT}/accuracy_requests" \
  --concurrency 1 \
  --min-requests 64 \
  --sequential \
  --rounds 1 2>&1 | tee "${ROOT}/logs/benchmark_${VARIANT}_accuracy.log"

docker exec shrutam2-server python /workspace/evaluate_accuracy.py \
  --requests "/workspace/results/${VARIANT}/accuracy_requests/requests.jsonl" \
  --output "/workspace/results/${VARIANT}/quick_accuracy.json" \
  --concurrency 1 2>&1 | tee "${ROOT}/logs/accuracy_${VARIANT}.log"

curl -fsS "${URL}/stats" | docker exec -i shrutam2-server sh -c \
  "cat > /workspace/results/${VARIANT}/server_stats_final.json"
