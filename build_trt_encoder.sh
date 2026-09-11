#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
ONNX=${ONNX:-${ROOT}/artifacts/shrutam2_encoder.onnx}
ENGINE=${ENGINE:-${ROOT}/artifacts/shrutam2_encoder_bf16.plan}
TRTEXEC=${TRTEXEC:-/usr/local/tensorrt/bin/trtexec}
PRECISION=${PRECISION:-bf16}
CACHE=${CACHE:-${ROOT}/artifacts/shrutam2_encoder_${PRECISION}.timing.cache}
MIN_BATCH=${MIN_BATCH:-1}
OPT_BATCH=${OPT_BATCH:-16}
MAX_BATCH=${MAX_BATCH:-64}
MIN_MEL_FRAMES=${MIN_MEL_FRAMES:-101}
OPT_MEL_FRAMES=${OPT_MEL_FRAMES:-501}
MAX_MEL_FRAMES=${MAX_MEL_FRAMES:-1001}

if [[ ! -f "${ONNX}" ]]; then
  echo "missing ONNX model: ${ONNX}" >&2
  exit 2
fi
if [[ ! -x "${TRTEXEC}" ]]; then
  TRTEXEC=$(command -v trtexec || true)
fi
if [[ -z "${TRTEXEC}" || ! -x "${TRTEXEC}" ]]; then
  echo "trtexec not found" >&2
  exit 3
fi

precision_flag=--bf16
if [[ "${PRECISION}" == "fp16" ]]; then
  precision_flag=--fp16
fi

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  "${precision_flag}" \
  --minShapes=audio_signal:${MIN_BATCH}x80x${MIN_MEL_FRAMES},lengths:${MIN_BATCH} \
  --optShapes=audio_signal:${OPT_BATCH}x80x${OPT_MEL_FRAMES},lengths:${OPT_BATCH} \
  --maxShapes=audio_signal:${MAX_BATCH}x80x${MAX_MEL_FRAMES},lengths:${MAX_BATCH} \
  --memPoolSize=workspace:16384 \
  --builderOptimizationLevel=5 \
  --timingCacheFile="${CACHE}" \
  --profilingVerbosity=detailed

ls -lh "${ENGINE}" "${CACHE}"
