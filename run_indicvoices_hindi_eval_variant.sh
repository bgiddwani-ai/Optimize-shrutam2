#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
VARIANT=${VARIANT:?set VARIANT}
SERVER_KIND=${SERVER_KIND:?set SERVER_KIND to hf or trtllm}
MANIFEST=${MANIFEST:-$ROOT/data/indicvoices_hindi_valid/manifest.jsonl}
ACCURACY_CONCURRENCY=${ACCURACY_CONCURRENCY:-32}
RUNTIME=${RUNTIME:-eager}
PRECISION=${PRECISION:-bf16}
ENGINE_DIR=${ENGINE_DIR:-}
ENCODER_RUNTIME=${ENCODER_RUNTIME:-eager}
ENCODER_PRECISION=${ENCODER_PRECISION:-bf16}
TRT_ENGINE=${TRT_ENGINE:-}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
MAX_DELAY_MS=${MAX_DELAY_MS:-8}
cd "$ROOT"

cleanup() {
  docker rm -f shrutam2-server >/dev/null 2>&1 || true
  screen -S shrutam2_server -X quit >/dev/null 2>&1 || true
  screen -S shrutam2_indic_eval_server -X quit >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

if [[ "$SERVER_KIND" == "hf" ]]; then
  ROOT="$ROOT" IMAGE=${IMAGE:-shrutam2-trtllm:0.20} \
    RUNTIME="$RUNTIME" PRECISION="$PRECISION" NUM_BEAMS=1 \
    MAX_NEW_TOKENS=96 MAX_BATCH_SIZE=32 MAX_DELAY_MS=8 bash start_server.sh \
    2>&1 | tee "logs/indicvoices_hindi_start_${VARIANT}.log"
elif [[ "$SERVER_KIND" == "trtllm" ]]; then
  [[ -n "$ENGINE_DIR" ]] || { echo "ENGINE_DIR is required" >&2; exit 2; }
  launch="cd $ROOT; ROOT=$ROOT ENGINE_DIR=$ENGINE_DIR ENCODER_RUNTIME=$ENCODER_RUNTIME ENCODER_PRECISION=$ENCODER_PRECISION TRT_ENGINE=$TRT_ENGINE CONTAINER_NAME=shrutam2-server MAX_BATCH_SIZE=$MAX_BATCH_SIZE MAX_DELAY_MS=$MAX_DELAY_MS MAX_NEW_TOKENS=96 NUM_BEAMS=1 bash start_server_trtllm.sh"
  screen -L -Logfile "$ROOT/logs/indicvoices_hindi_server_${VARIANT}.log" \
    -dmS shrutam2_indic_eval_server bash -lc "$launch"
  deadline=$((SECONDS + 1800))
  grace=$((SECONDS + 60))
  until curl -fsS http://127.0.0.1:8092/health > "logs/indicvoices_hindi_health_${VARIANT}.json"; do
    if (( SECONDS >= deadline )); then
      tail -200 "logs/indicvoices_hindi_server_${VARIANT}.log" >&2 || true
      exit 4
    fi
    if (( SECONDS >= grace )) && ! docker ps --format '{{.Names}}' | grep -qx shrutam2-server; then
      tail -200 "logs/indicvoices_hindi_server_${VARIANT}.log" >&2 || true
      exit 5
    fi
    sleep 5
  done
  cat "logs/indicvoices_hindi_health_${VARIANT}.json"
else
  echo "SERVER_KIND must be hf or trtllm" >&2
  exit 2
fi

requests=$(wc -l < "$MANIFEST" | tr -d ' ')
output="/workspace/results/$VARIANT/indicvoices_hindi_valid_accuracy_requests"
set +e
docker exec shrutam2-server python /workspace/benchmark_client.py \
  --url http://127.0.0.1:8092 \
  --manifest "${MANIFEST/#${ROOT}/\/workspace}" \
  --output-dir "$output" \
  --concurrency "$ACCURACY_CONCURRENCY" \
  --min-requests "$requests" --sequential --rounds 1 \
  2>&1 | tee "logs/indicvoices_hindi_benchmark_${VARIANT}.log"
benchmark_status=${PIPESTATUS[0]}
set -e
if (( benchmark_status != 0 && benchmark_status != 2 )); then
  exit "$benchmark_status"
fi

docker exec shrutam2-server python /workspace/evaluate_accuracy.py \
  --requests "$output/requests.jsonl" \
  --output "/workspace/results/$VARIANT/indicvoices_hindi_valid_accuracy.json" \
  --concurrency "$ACCURACY_CONCURRENCY" \
  2>&1 | tee "logs/indicvoices_hindi_accuracy_${VARIANT}.log"

curl -fsS http://127.0.0.1:8092/stats | \
  docker exec -i shrutam2-server tee \
    "/workspace/results/$VARIANT/indicvoices_hindi_valid_server_stats.json" \
    >/dev/null
echo "$VARIANT IndicVoices Hindi valid complete"
