#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
OUTPUT=${OUTPUT:-/home/nvidia/Optimize-shrutam2-delivery.tar.gz}
stage=$(mktemp -d /home/nvidia/shrutam2-delivery.XXXXXX)
trap 'rm -rf "${stage}"' EXIT

mkdir -p "${stage}/Optimize-shrutam2/artifacts" "${stage}/Optimize-shrutam2/data/indicvoices_quick"
cp -a "${ROOT}/results" "${ROOT}/logs" "${stage}/Optimize-shrutam2/"
cp -a "${ROOT}/data/indicvoices_quick/manifest.jsonl" "${stage}/Optimize-shrutam2/data/indicvoices_quick/"
if [[ -f "${ROOT}/data/indicvoices_quick/report.json" ]]; then
  cp -a "${ROOT}/data/indicvoices_quick/report.json" "${stage}/Optimize-shrutam2/data/indicvoices_quick/"
fi
find "${ROOT}/artifacts" -maxdepth 1 -type f -name '*.json' -exec cp -a {} "${stage}/Optimize-shrutam2/artifacts/" \;
for engine_dir in trtllm_llm_bf16_engine trtllm_llm_modelopt_fp8_engine; do
  if [[ -f "${ROOT}/artifacts/${engine_dir}/config.json" ]]; then
    mkdir -p "${stage}/Optimize-shrutam2/artifacts/${engine_dir}"
    cp -a "${ROOT}/artifacts/${engine_dir}/config.json" "${stage}/Optimize-shrutam2/artifacts/${engine_dir}/"
  fi
done
tar -C "${stage}" -czf "${OUTPUT}" Optimize-shrutam2
sha256sum "${OUTPUT}"
du -h "${OUTPUT}"
