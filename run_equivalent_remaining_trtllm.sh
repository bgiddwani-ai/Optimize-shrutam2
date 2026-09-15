#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
cd "$ROOT"

run_variant() {
  local variant=$1
  local engine=$2
  local encoder_runtime=$3
  local encoder_precision=$4
  local trt_engine=${5:-}
  local status=0

  echo "BEGIN $variant $(date -u +%FT%TZ)"
  env \
    VARIANT="$variant" \
    ENGINE_DIR="$engine" \
    ENCODER_RUNTIME="$encoder_runtime" \
    ENCODER_PRECISION="$encoder_precision" \
    TRT_ENGINE="$trt_engine" \
    MAX_BATCH_SIZE=64 \
    MAX_DELAY_MS=12 \
    bash run_equivalent_trtllm_variant.sh || status=$?
  echo "END $variant status=$status $(date -u +%FT%TZ)"
  return 0
}

run_variant \
  equiv_eager_bf16enc_trtllm_fp8w_bf16kv_beam1 \
  "$ROOT/artifacts/trtllm_equiv_fp8w_bf16kv_engine" \
  eager bf16

run_variant \
  equiv_trt_bf16enc_trtllm_bf16_beam1 \
  "$ROOT/artifacts/trtllm_equiv_bf16_engine" \
  trt bf16 "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"

run_variant \
  equiv_trt_bf16enc_trtllm_fp8w_bf16kv_beam1 \
  "$ROOT/artifacts/trtllm_equiv_fp8w_bf16kv_engine" \
  trt bf16 "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"

run_variant \
  equiv_trt_bf16enc_trtllm_bf16w_fp8kv_beam1 \
  "$ROOT/artifacts/trtllm_equiv_bf16w_fp8kv_engine" \
  trt bf16 "$ROOT/artifacts/shrutam2_encoder_equiv_bf16.plan"

run_variant \
  equiv_trt_fp8enc_trtllm_fp8w_bf16kv_beam1 \
  "$ROOT/artifacts/trtllm_equiv_fp8w_bf16kv_engine" \
  trt bf16 "$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.plan"

run_variant \
  equiv_trt_fp8enc_trtllm_fp8w_fp8kv_beam1 \
  "$ROOT/artifacts/trtllm_equiv_fp8w_fp8kv_engine" \
  trt bf16 "$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.plan"
