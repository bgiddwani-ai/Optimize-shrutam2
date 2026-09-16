# Optimize Shrutam-2

This repository is the minimal, reproducible server/client workflow that
produces the final nine-row Shrutam-2 comparison. Shrutam-2 has a
**FastConformer** encoder, SMEAR projector, and fine-tuned Llama decoder.

It retains only the required paths:

- HF eager and `torch.compile` control servers;
- TensorRT-LLM BF16, ModelOpt FP8-weights/BF16-KV, BF16-weights/FP8-KV, and
  ModelOpt FP8-weights/FP8-KV decoders;
- TensorRT BF16 and calibrated mixed-FP8 FastConformer plans;
- HTTP server/client RTFx matrices and one common cleaned IndicVoices Hindi
  WER/CER evaluation.

The historical evidence, raw logs, and final reports are preserved. Old AOTI,
vLLM, single-engine experiments, and their launch/build helpers are removed.

## Final results

Every row uses beam 1, max 96 new tokens, identical decoding settings, and the
same cleaned IndicVoices Hindi validation manifest. WER/CER are calculated on
all 4,955 retained clips (7.85 h); empty or failed hypotheses score as empty
and are never silently excluded.

| Key | Configuration | WER | CER | WER delta pp vs base | Peak variable-audio RTFx |
|---|---|---:|---:|---:|---:|
| `base_hf` | FP32 FastConformer + HF BF16 LLM | 60.3809% | 51.5645% | 0.0000 | 334.69 |
| `compiled_bf16` | Compiled BF16 FastConformer + HF BF16 LLM | 60.3936% | 51.5708% | +0.0127 | 313.04 |
| `trtllm_bf16` | PyTorch BF16 FastConformer + TensorRT-LLM BF16 | 59.8534% | 50.1186% | -0.5274 | 862.57 |
| `trtllm_fp8` | PyTorch BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 60.2287% | 50.5695% | -0.1521 | 983.91 |
| `trt_bf16_trtllm_bf16` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 | 59.2436% | 49.5674% | -1.1373 | 875.88 |
| `trt_bf16_trtllm_fp8` | TensorRT BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 60.0195% | 50.2724% | -0.3613 | 1,081.29 |
| `trt_bf16_trtllm_bf16_fp8kv` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 weights, FP8 KV | 60.2389% | 50.4925% | -0.1420 | 879.97 |
| `trt_fp8_trtllm_fp8` | TensorRT calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 59.1142% | 49.0432% | -1.2666 | 1,105.05 |
| `trt_fp8_trtllm_fp8_fp8kv` | TensorRT calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights, FP8 KV | 59.3285% | 49.4316% | -1.0524 | **1,145.61** |

`results/equivalent_consolidated_results.json` is the canonical report. It
contains the complete variable-audio matrix at `1,2,8,32,64,128,256`, the
exact-1-second matrix at `1,2,8,32,64,128,256,512`, per-row integrity checks,
engine inventory, and held-out WER/CER. RTFx is valid input audio seconds
divided by client wall-clock seconds, from persistent HTTP request send through
parsing a non-empty transcript.

## Prerequisites

Use a Linux GPU host with Docker access, an NVIDIA driver compatible with the
two pinned base images, and GNU screen for the long-running operations. Run the
workflow from a screen session; pass `ROOT=$PWD` on every command so no default
host path is assumed.

```bash
git clone <this-repository-url> Optimize-shrutam2
cd Optimize-shrutam2
screen -S shrutam2
nvidia-smi
docker version
```

## Install and build

Build the two pinned runtime images. The HF/NeMo image builds the control rows;
the TensorRT-LLM image supplies TensorRT 10.10, TensorRT-LLM 0.20, and NVIDIA
Model Optimizer 0.29 for the optimized rows.

```bash
ROOT=$PWD bash build_runtime.sh
ROOT=$PWD bash build_trtllm_runtime.sh
```

Download the pinned `bharatgenai/Shrutam-2` revision, export the decoder
checkpoint required by TensorRT-LLM, materialize the fixed FLEURS throughput
set, export BF16/FP32 encoder ONNX, and generate the real-audio encoder
calibration inputs.

```bash
ROOT=$PWD bash setup_equivalent.sh
```

Build all four decoder variants and both encoder plans:

```bash
ROOT=$PWD bash build_trtllm_variants.sh
ROOT=$PWD bash build_equivalent_encoder_engines.sh
```

Decoder ModelOpt calibration uses 128 CNN/DailyMail sequences. Encoder ModelOpt
calibration uses the 48 FLEURS waveforms after Shrutam's real log-mel frontend.
The mixed-FP8 TensorRT plan quantizes eligible convolution/MatMul operators;
unsupported depthwise convolutions remain at higher precision.

## Prepare the held-out accuracy set

The evaluator reads only IndicVoices `hindi/valid-00000-of-00001.parquet`.
Supply a Hugging Face read token through the environment; it is used only for
the download and is not written to the repository.

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
docker run --rm -e HF_TOKEN -v "$PWD:/workspace" -w /workspace \
  shrutam2-trtllm:0.20 python prepare_indicvoices_hindi_eval.py
unset HF_TOKEN
```

Preparation removes every reference whose text contains `unintelligible`
case-insensitively (including `<unintelligible>`), validates decoded audio, and
keeps only 0.5–19.0-second clips to respect the shared 256-token TensorRT-LLM
input contract. The output manifest and exclusion accounting are in
`data/indicvoices_hindi_valid/`.

## Reproduce the final table

Run the two HF controls, then the BF16 PyTorch-encoder TensorRT-LLM control:

```bash
ROOT=$PWD VARIANT=equiv_base_fp32enc_hf_bf16llm_beam1 \
  RUNTIME=eager PRECISION=upstream bash run_equivalent_hf_variant.sh

ROOT=$PWD VARIANT=equiv_compile_bf16_hf_bf16llm_beam1 \
  RUNTIME=compile PRECISION=bf16 bash run_equivalent_hf_variant.sh

ROOT=$PWD VARIANT=equiv_eager_bf16enc_trtllm_bf16_beam1 \
  ENGINE_DIR=$PWD/artifacts/trtllm_equiv_bf16_engine \
  ENCODER_RUNTIME=eager ENCODER_PRECISION=bf16 \
  bash run_equivalent_trtllm_variant.sh
```

Run the remaining six optimized combinations, then evaluate all nine servers
against the cleaned IndicVoices manifest and merge WER/CER into the final
report:

```bash
ROOT=$PWD bash run_equivalent_remaining_trtllm.sh
ROOT=$PWD bash run_all_indicvoices_hindi_eval.sh

docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  shrutam2-trtllm:0.20 bash -lc \
  'python finalize_indicvoices_hindi_accuracy.py && python finalize_equivalent.py'
```

Each performance row starts a real FastAPI server, checks real-audio warm-up,
then runs both requested concurrency matrices. The server rejects audio longer
than 19 seconds before adaptive batching. The IndicVoices run uses concurrency
32 and requires one non-empty successful transcript for every manifest row.

## Deploy and infer with the winning profile

Start the winning mixed-FP8 encoder plus ModelOpt FP8-weight/FP8-KV decoder
server. This is a blocking command; run it in its own screen window.

```bash
ROOT=$PWD \
ENGINE_DIR=$PWD/artifacts/trtllm_equiv_fp8w_fp8kv_engine \
ENCODER_RUNTIME=trt \
TRT_ENGINE=$PWD/artifacts/shrutam2_encoder_modelopt_fp8.plan \
NUM_BEAMS=1 MAX_BATCH_SIZE=64 MAX_DELAY_MS=12 \
  bash start_server_trtllm.sh
```

In a second shell, verify health and submit a real client request from the
prepared FLEURS set:

```bash
curl -fsS http://127.0.0.1:8092/health

docker exec shrutam2-trtllm-server python /workspace/benchmark_client.py \
  --url http://127.0.0.1:8092 \
  --manifest /workspace/data/fleurs_quick/manifest.jsonl \
  --output-dir /workspace/results/standalone_infer \
  --concurrency 1 --min-requests 1 --rounds 1
```

The HTTP API accepts non-empty little-endian signed-16-bit 16 kHz PCM on
`POST /transcribe`, with optional `x-language` (default `hi`). Inspect live
batching through `/stats`, then release GPU memory with:

```bash
bash stop_server.sh
```

## Retained evidence

- `results/equivalent_consolidated_results.json` — canonical performance and
  accuracy report.
- `results/indicvoices_hindi_valid_accuracy.json` — all-sample IndicVoices
  Hindi WER/CER comparison and integrity accounting.
- `artifacts/equivalent_evidence_20260916.tgz` — archive of remote evidence;
  SHA-256 `a204b400fe1f29c7d22abe5ad4d1e7bdaf371abe22ce2f77b648d93334199f20`.
- `LEARNIGS.md` — retained technical decisions, failures, and limitations.
