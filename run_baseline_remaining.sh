#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
VARIANT=upstream_fp32conformer_bf16llm_beam4
MANIFEST=/workspace/data/indicvoices_quick/manifest.jsonl
URL=http://127.0.0.1:8092

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" --manifest "${MANIFEST}" \
  --output-dir "/workspace/results/${VARIANT}/variable_c256_retry" \
  --concurrency 256 --rounds 1 --timeout 900 \
  2>&1 | tee "${ROOT}/logs/baseline_c256_retry.log"

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" --manifest "${MANIFEST}" \
  --output-dir "/workspace/results/${VARIANT}/chunk_1s" \
  --fixed-seconds 1 --concurrency 1 2 8 32 64 128 256 512 \
  --min-requests 64 --rounds 1 --timeout 900 \
  2>&1 | tee "${ROOT}/logs/baseline_chunk.log"

docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url "${URL}" --manifest "${MANIFEST}" \
  --output-dir "/workspace/results/${VARIANT}/accuracy_requests" \
  --concurrency 1 --min-requests 64 --sequential --rounds 1 \
  2>&1 | tee "${ROOT}/logs/baseline_accuracy_requests.log"

docker exec shrutam2-server python /workspace/evaluate_accuracy.py \
  --requests "/workspace/results/${VARIANT}/accuracy_requests/requests.jsonl" \
  --output "/workspace/results/${VARIANT}/quick_accuracy.json" \
  --concurrency 1 2>&1 | tee "${ROOT}/logs/baseline_accuracy.log"

curl -fsS "${URL}/stats" | docker exec -i shrutam2-server sh -c \
  "cat > /workspace/results/${VARIANT}/server_stats_final.json"
