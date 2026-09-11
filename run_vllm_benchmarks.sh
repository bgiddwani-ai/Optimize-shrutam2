#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
VARIANT=${VARIANT:-vllm_continuous_bf16_adaptive_persistent_beam1}
MANIFEST=${MANIFEST:-$ROOT/data/indicvoices_quick/manifest.jsonl}
URL=${URL:-http://127.0.0.1:8092}

source "$ROOT/.venv-vllm/bin/activate"
cd "$ROOT"

screen -S shrutam2_gpu_monitor -X quit >/dev/null 2>&1 || true
screen -L -Logfile "$ROOT/logs/gpu_${VARIANT}.csv" -dmS shrutam2_gpu_monitor \
  nvidia-smi --id=0 --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv -l 1
trap 'screen -S shrutam2_gpu_monitor -X quit >/dev/null 2>&1 || true' EXIT

python benchmark_client.py \
  --url "$URL" \
  --manifest "$MANIFEST" \
  --output-dir "$ROOT/results/$VARIANT/variable" \
  --concurrency 1 2 8 32 64 128 256 \
  --min-requests 32 \
  --rounds 1 2>&1 | tee "$ROOT/logs/benchmark_${VARIANT}_variable.log"

python benchmark_client.py \
  --url "$URL" \
  --manifest "$MANIFEST" \
  --output-dir "$ROOT/results/$VARIANT/chunk_1s" \
  --fixed-seconds 1 \
  --concurrency 1 2 8 32 64 128 256 512 \
  --min-requests 64 \
  --rounds 1 2>&1 | tee "$ROOT/logs/benchmark_${VARIANT}_chunk_1s.log"

python benchmark_client.py \
  --url "$URL" \
  --manifest "$MANIFEST" \
  --output-dir "$ROOT/results/$VARIANT/accuracy_requests" \
  --concurrency 1 \
  --min-requests 64 \
  --sequential \
  --rounds 1 2>&1 | tee "$ROOT/logs/benchmark_${VARIANT}_accuracy.log"

python evaluate_accuracy.py \
  --requests "$ROOT/results/$VARIANT/accuracy_requests/requests.jsonl" \
  --output "$ROOT/results/$VARIANT/quick_accuracy.json" \
  --concurrency 1 2>&1 | tee "$ROOT/logs/accuracy_${VARIANT}.log"

curl -fsS "$URL/stats" | tee "$ROOT/results/$VARIANT/server_stats_final.json" >/dev/null
echo __VLLM_BENCHMARKS_DONE__
