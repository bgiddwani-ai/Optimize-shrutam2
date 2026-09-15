#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/Optimize-shrutam2}
IMAGE=${IMAGE:-shrutam2-trtllm:0.20}
mkdir -p "$ROOT/logs"

docker run --rm --gpus device=0 --ipc=host \
  -v "$ROOT:/workspace" -w /workspace "$IMAGE" bash -lc '
set -euo pipefail
python -m pip check
python - <<'"'"'PY'"'"'
import inspect
import json

import fastapi
import modelopt
import soundfile
import tensorrt
import tensorrt_llm
import torch
from modelopt.onnx.quantization import quantize
from tensorrt_llm.runtime import ModelRunnerCpp

report = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "tensorrt": tensorrt.__version__,
    "tensorrt_llm": tensorrt_llm.__version__,
    "modelopt": modelopt.__version__,
    "fastapi": fastapi.__version__,
    "soundfile": soundfile.__version__,
    "modelopt_onnx_quantize_signature": str(inspect.signature(quantize)),
    "model_runner_from_dir_signature": str(inspect.signature(ModelRunnerCpp.from_dir)),
    "model_runner_generate_signature": str(inspect.signature(ModelRunnerCpp.generate)),
}
with open("/workspace/artifacts/equivalent_trtllm_runtime_probe.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
PY
trtllm-build --help > /workspace/logs/equivalent_trtllm_build_help.txt
python /app/tensorrt_llm/examples/quantization/quantize.py --help \
  > /workspace/logs/equivalent_trtllm_quantize_help.txt
grep -E -- "--(opt_batch_size|opt_num_tokens|multiple_profiles|use_paged_context_fmha|max_prompt_embedding_table_size)" \
  /workspace/logs/equivalent_trtllm_build_help.txt
grep -E -- "--(qformat|kv_cache_dtype|calib_dataset|calib_size|calib_max_seq_length)" \
  /workspace/logs/equivalent_trtllm_quantize_help.txt
'
