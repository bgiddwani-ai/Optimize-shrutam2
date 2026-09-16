#!/usr/bin/env bash
set -euo pipefail
docker rm -f shrutam2-server shrutam2-trtllm-server >/dev/null 2>&1 || true
screen -S shrutam2_server -X quit >/dev/null 2>&1 || true
screen -S shrutam2_trtllm_server -X quit >/dev/null 2>&1 || true
screen -S shrutam2_indic_eval_server -X quit >/dev/null 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
