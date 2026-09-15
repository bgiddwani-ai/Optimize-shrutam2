#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace}
ONNX=${ONNX:-$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.onnx}
ENGINE=${ENGINE:-$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.plan}
CACHE=${CACHE:-$ROOT/artifacts/shrutam2_encoder_modelopt_fp8.timing.cache}
TRTEXEC=${TRTEXEC:-/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec}
MIN_BATCH=${MIN_BATCH:-1}
OPT_BATCH=${OPT_BATCH:-32}
MAX_BATCH=${MAX_BATCH:-64}
MIN_MEL_FRAMES=${MIN_MEL_FRAMES:-101}
OPT_MEL_FRAMES=${OPT_MEL_FRAMES:-501}
MAX_MEL_FRAMES=${MAX_MEL_FRAMES:-2001}

[[ -f "$ONNX" ]] || { echo "missing quantized ONNX: $ONNX" >&2; exit 2; }
[[ -x "$TRTEXEC" ]] || TRTEXEC=$(command -v trtexec)

"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --stronglyTyped \
  --minShapes=audio_signal:${MIN_BATCH}x80x${MIN_MEL_FRAMES},lengths:${MIN_BATCH} \
  --optShapes=audio_signal:${OPT_BATCH}x80x${OPT_MEL_FRAMES},lengths:${OPT_BATCH} \
  --maxShapes=audio_signal:${MAX_BATCH}x80x${MAX_MEL_FRAMES},lengths:${MAX_BATCH} \
  --memPoolSize=workspace:16384 --builderOptimizationLevel=5 \
  --timingCacheFile="$CACHE" --profilingVerbosity=detailed

ls -lh "$ENGINE" "$CACHE"
