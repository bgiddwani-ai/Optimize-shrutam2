#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
VARIANT=${VARIANT:?set VARIANT}
RUNTIME=${RUNTIME:?set RUNTIME to eager or compile}
PRECISION=${PRECISION:?set PRECISION to upstream or bf16}
MANIFEST=${MANIFEST:-$ROOT/data/fleurs_quick/manifest.jsonl}
cd "$ROOT"

cleanup() {
  bash stop_server.sh || true
}
trap cleanup EXIT
cleanup

RUNTIME="$RUNTIME" PRECISION="$PRECISION" NUM_BEAMS=1 MAX_NEW_TOKENS=96 \
  MAX_BATCH_SIZE=32 MAX_DELAY_MS=8 ROOT="$ROOT" bash start_server.sh \
  2>&1 | tee "logs/equivalent_start_${VARIANT}.log"

VARIANT="$VARIANT" MANIFEST="$MANIFEST" ROOT="$ROOT" CONTAINER_NAME=shrutam2-server \
  bash run_http_benchmarks.sh \
  2>&1 | tee "logs/equivalent_run_${VARIANT}.log"

echo "$VARIANT complete"
