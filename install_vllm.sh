#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/nvidia/Optimize-shrutam2}
VENV=${VENV:-$ROOT/.venv-vllm}
FORK=${FORK:-$ROOT/baseline/Canary-Qwen-2.5b-vllm}

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
python -m pip install 'setuptools-scm>=8' numpy cmake ninja packaging
cd "$FORK"
# Install the fork's dependency set and Python changes. Its wheel omitted the
# downloaded ``vllm._C`` extension on this host, so overlay the ABI-compatible
# official 0.12.0 binary wheel after dependency resolution.
VLLM_USE_PRECOMPILED=1 python -m pip install . --no-build-isolation
python -m pip install --force-reinstall --no-deps 'vllm==0.12.0'
python -m pip install soundfile scipy pyyaml jiwer fastapi uvicorn httpx safetensors
python -c 'import torch, vllm, vllm._C; print(torch.__version__, torch.version.cuda, vllm.__version__, vllm._C.__file__)'
