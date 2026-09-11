FROM nvcr.io/nvidia/nemo:25.02

RUN python -m pip install --no-cache-dir --upgrade \
    "transformers==4.56.2" \
    "huggingface_hub==0.36.0" \
    "soundfile==0.13.1" \
    "fastapi==0.116.1" \
    "uvicorn[standard]==0.35.0" \
    "httpx==0.28.1" \
    "python-multipart==0.0.20" \
    "jiwer==4.0.0" \
    "onnx==1.17.0" \
    "onnxscript==0.2.2"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

