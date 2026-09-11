#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
ENGINE=${ENGINE:-${ROOT}/artifacts/shrutam2_encoder_bf16.plan}
OUT=${OUT:-${ROOT}/results/encoder/trtexec_bf16_1s}
TRTEXEC=${TRTEXEC:-/usr/local/tensorrt/bin/trtexec}

mkdir -p "${OUT}"
for batch in 1 2 8 32; do
  "${TRTEXEC}" \
    --loadEngine="${ENGINE}" \
    --shapes="audio_signal:${batch}x80x101,lengths:${batch}" \
    --warmUp=500 --duration=0 --iterations=100 \
    --noDataTransfers --useCudaGraph \
    --exportTimes="${OUT}/times_b${batch}.json" \
    2>&1 | tee "${OUT}/trtexec_b${batch}.log"
done
