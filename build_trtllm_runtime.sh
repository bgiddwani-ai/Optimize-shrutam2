#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
cd "$ROOT"
docker build --pull=false -f Dockerfile.trtllm -t shrutam2-trtllm:0.20 . \
  2>&1 | tee logs/equivalent_trtllm_image_build.log
docker image inspect shrutam2-trtllm:0.20 --format '{{.Id}}' \
  | tee logs/equivalent_trtllm_image_id.log
