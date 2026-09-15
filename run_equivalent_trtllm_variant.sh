#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
VARIANT=${VARIANT:?set VARIANT}
ENGINE_DIR=${ENGINE_DIR:?set ENGINE_DIR}
ENCODER_RUNTIME=${ENCODER_RUNTIME:-eager}
ENCODER_PRECISION=${ENCODER_PRECISION:-bf16}
TRT_ENGINE=${TRT_ENGINE:-}
MANIFEST=${MANIFEST:-$ROOT/data/fleurs_quick/manifest.jsonl}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-64}
MAX_DELAY_MS=${MAX_DELAY_MS:-12}
cd "$ROOT"

cleanup() {
  docker rm -f shrutam2-server >/dev/null 2>&1 || true
  screen -S shrutam2_trtllm_server -X quit >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

launch="cd $ROOT; ROOT=$ROOT ENGINE_DIR=$ENGINE_DIR ENCODER_RUNTIME=$ENCODER_RUNTIME ENCODER_PRECISION=$ENCODER_PRECISION TRT_ENGINE=$TRT_ENGINE CONTAINER_NAME=shrutam2-server MAX_BATCH_SIZE=$MAX_BATCH_SIZE MAX_DELAY_MS=$MAX_DELAY_MS NUM_BEAMS=1 bash start_server_trtllm.sh"
screen -L -Logfile "$ROOT/logs/server_${VARIANT}.log" -dmS shrutam2_trtllm_server \
  bash -lc "$launch"

deadline=$((SECONDS + 1800))
grace=$((SECONDS + 60))
until curl -fsS http://127.0.0.1:8092/health > "logs/health_${VARIANT}.json"; do
  if (( SECONDS >= deadline )); then
    tail -200 "logs/server_${VARIANT}.log" >&2 || true
    exit 4
  fi
  if (( SECONDS >= grace )) && ! docker ps --format '{{.Names}}' | grep -qx shrutam2-server; then
    tail -200 "logs/server_${VARIANT}.log" >&2 || true
    exit 5
  fi
  sleep 5
done
cat "logs/health_${VARIANT}.json"

VARIANT="$VARIANT" MANIFEST="$MANIFEST" ROOT="$ROOT" CONTAINER_NAME=shrutam2-server \
  bash run_http_benchmarks.sh \
  2>&1 | tee "logs/equivalent_run_${VARIANT}.log"

echo "$VARIANT complete"
