#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
cd "${ROOT}"
mkdir -p logs artifacts results data
docker build --pull=false -t shrutam2-runtime:25.02 . 2>&1 | tee logs/docker_build.log
docker image inspect shrutam2-runtime:25.02 --format '{{json .RepoDigests}} {{.Id}}' | tee logs/runtime_image_id.log

