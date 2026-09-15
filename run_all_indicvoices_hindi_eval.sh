#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
cd "$ROOT"
manifest_requests=$(wc -l < "$ROOT/data/indicvoices_hindi_valid/manifest.jsonl" | tr -d ' ')

variant_complete() {
  local variant=$1
  local accuracy="$ROOT/results/$variant/indicvoices_hindi_valid_accuracy.json"
  local requests="$ROOT/results/$variant/indicvoices_hindi_valid_accuracy_requests/requests.jsonl"
  [[ -s "$accuracy" && -s "$requests" ]] || return 1
  [[ $(wc -l < "$requests" | tr -d ' ') == "$manifest_requests" ]] || return 1
  python3 - "$accuracy" "$requests" "$manifest_requests" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
expected = int(sys.argv[3])
complete = (
    report.get("concurrency") == 32
    and report.get("samples") == expected
    and len(rows) == expected
    and all(row.get("ok") and str(row.get("text") or "").strip() for row in rows)
)
raise SystemExit(0 if complete else 1)
PY
}

run_hf() {
  local variant=$1 runtime=$2 precision=$3
  if variant_complete "$variant"; then
    echo "$variant already complete; skipping"
    return
  fi
  ROOT="$ROOT" VARIANT="$variant" SERVER_KIND=hf RUNTIME="$runtime" \
    PRECISION="$precision" ACCURACY_CONCURRENCY=32 \
    bash run_indicvoices_hindi_eval_variant.sh
}

run_trtllm() {
  local variant=$1 engine=$2 encoder_runtime=$3 encoder_precision=$4 trt_engine=${5:-}
  if variant_complete "$variant"; then
    echo "$variant already complete; skipping"
    return
  fi
  ROOT="$ROOT" VARIANT="$variant" SERVER_KIND=trtllm \
    ENGINE_DIR="$ROOT/artifacts/$engine" \
    ENCODER_RUNTIME="$encoder_runtime" ENCODER_PRECISION="$encoder_precision" \
    TRT_ENGINE="$trt_engine" MAX_BATCH_SIZE=64 MAX_DELAY_MS=12 \
    ACCURACY_CONCURRENCY=32 bash run_indicvoices_hindi_eval_variant.sh
}

run_hf equiv_base_fp32enc_hf_bf16llm_beam1 eager upstream
run_hf equiv_compile_bf16_hf_bf16llm_beam1 compile bf16

run_trtllm equiv_eager_bf16enc_trtllm_bf16_beam1 \
  trtllm_equiv_bf16_engine eager bf16
run_trtllm equiv_eager_bf16enc_trtllm_fp8w_bf16kv_beam1 \
  trtllm_equiv_fp8w_bf16kv_engine eager bf16
run_trtllm equiv_trt_bf16enc_trtllm_bf16_beam1 \
  trtllm_equiv_bf16_engine trt bf16 \
  "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"
run_trtllm equiv_trt_bf16enc_trtllm_fp8w_bf16kv_beam1 \
  trtllm_equiv_fp8w_bf16kv_engine trt bf16 \
  "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"
run_trtllm equiv_trt_bf16enc_trtllm_bf16w_fp8kv_beam1 \
  trtllm_equiv_bf16w_fp8kv_engine trt bf16 \
  "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"
run_trtllm equiv_trt_fp8enc_trtllm_fp8w_bf16kv_beam1 \
  trtllm_equiv_fp8w_bf16kv_engine trt bf16 \
  "$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.plan"
run_trtllm equiv_trt_fp8enc_trtllm_fp8w_fp8kv_beam1 \
  trtllm_equiv_fp8w_fp8kv_engine trt bf16 \
  "$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.plan"

echo "all IndicVoices Hindi validation variants complete"
