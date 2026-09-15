# Optimize Shrutam-2

Measured server/client optimization of `bharatgenai/Shrutam-2` on the Brev
instance `equivalent-purple-ostrich`. Shrutam-2 uses a **FastConformer encoder**,
not a plain Conformer encoder, followed by the SMEAR projector and a fine-tuned
Llama decoder.

All eight requested beam-1 configurations completed both throughput matrices.
There were zero failed requests among 14,336 timed matrix requests, and every
quick WER result remained within the accepted +1.5 percentage-point budget.
The consolidated machine-readable report is
[`results/equivalent_consolidated_results.json`](results/equivalent_consolidated_results.json).

## Result summary

The maximum variable-audio result was **1,105.05 RTFx at c128** with the
calibrated mixed-FP8 TensorRT FastConformer and Model Optimizer FP8
TensorRT-LLM decoder. Its quick WER was 19.2825%, a signed change of -0.2242
percentage points from the HF baseline. The best exact-1-second c1 result was
36.51 RTFx with the same profile.

The encoder FP8 label is deliberately qualified as *mixed-FP8*: NVIDIA Model
Optimizer inserted FP8 Q/DQ for eligible MatMul and convolution operators, but
TensorRT retained unsupported depthwise convolutions at higher precision. This
is the deployable calibrated engine that passed real-audio transcript gates,
not a claim that every encoder operator ran in FP8.

| Key | Requested configuration, beam 1 | WER | CER | WER delta pp | CER delta pp | Peak variable RTFx |
|---|---|---:|---:|---:|---:|---:|
| `base_hf` | Base: FP32 FastConformer + HF BF16 LLM | 19.5067% | 5.9750% | 0.0000 | 0.0000 | 334.69 |
| `compiled_bf16` | Compiled BF16 FastConformer + HF BF16 LLM | 20.6278% | 6.5474% | +1.1211 | +0.5725 | 313.04 |
| `trtllm_bf16` | PyTorch BF16 FastConformer + TensorRT-LLM BF16 | 20.2915% | 6.4043% | +0.7848 | +0.4293 | 862.57 |
| `trtllm_fp8` | PyTorch BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 19.7309% | 6.2791% | +0.2242 | +0.3041 | 983.91 |
| `trt_bf16_trtllm_bf16` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 | 19.9552% | 6.1896% | +0.4484 | +0.2147 | 875.88 |
| `trt_bf16_trtllm_fp8` | TensorRT BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 20.6278% | 6.9410% | +1.1211 | +0.9660 | 1,081.29 |
| `trt_bf16_trtllm_bf16_fp8kv` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 weights, FP8 KV | 19.5067% | 5.9571% | 0.0000 | -0.0179 | 879.97 |
| `trt_fp8_trtllm_fp8` | TensorRT calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 19.2825% | 6.2075% | -0.2242 | +0.2326 | **1,105.05** |

WER/CER are quick accuracy measurements over the same fixed 48-clip FLEURS
set for every row, not a full model-quality evaluation. A negative delta is an
improvement on this sample. The original cached IndicVoices set used on the
earlier H100 work was unavailable on this fresh instance, so the controlled
comparison uses four public test clips from each of 12 supported languages.

## Full end-to-end RTFx matrices

RTFx is:

```text
sum(valid input audio seconds) / client wall-clock seconds
```

The timed boundary begins when persistent clients send requests and ends after
non-empty transcripts are parsed. It includes HTTP transfer, server queueing,
log-mel preprocessing, FastConformer, SMEAR projection, decoder execution, and
response parsing. File loading, PCM preparation, and connection preconnect are
untimed. Concurrency is client concurrency; adaptive server microbatches are
recorded in each `server_stats.json`.

Variable-duration FLEURS audio:

| Key | c1 | c2 | c8 | c32 | c64 | c128 | c256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_hf` | 39.80 | 45.59 | 151.55 | 232.87 | 297.79 | 320.87 | 334.69 |
| `compiled_bf16` | 18.61 | 36.84 | 99.82 | 231.06 | 313.04 | 249.02 | 308.11 |
| `trtllm_bf16` | 81.67 | 119.41 | 384.60 | 570.37 | 723.51 | 862.57 | 649.89 |
| `trtllm_fp8` | 104.58 | 166.71 | 457.55 | 550.49 | 854.79 | 983.91 | 668.78 |
| `trt_bf16_trtllm_bf16` | 83.46 | 124.76 | 349.50 | 570.70 | 763.40 | 875.88 | 652.43 |
| `trt_bf16_trtllm_fp8` | 107.84 | 167.92 | 473.59 | 682.29 | 920.00 | 1,081.29 | 702.65 |
| `trt_bf16_trtllm_bf16_fp8kv` | 81.43 | 122.43 | 356.22 | 560.61 | 751.75 | 879.97 | 647.52 |
| `trt_fp8_trtllm_fp8` | 104.04 | 169.87 | 472.58 | 694.89 | 935.11 | **1,105.05** | 689.74 |

Exact-one-second audio, trimmed or zero-padded to 16,000 samples:

| Key | c1 | c2 | c8 | c32 | c64 | c128 | c256 | c512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_hf` | 22.15 | 37.23 | 106.16 | 174.30 | 205.36 | 159.05 | 132.09 | 80.29 |
| `compiled_bf16` | 28.51 | 29.14 | 37.96 | 46.15 | 46.68 | 77.18 | 67.66 | 65.97 |
| `trtllm_bf16` | 25.63 | 45.98 | 148.55 | 343.02 | 301.51 | 147.81 | 134.25 | 89.61 |
| `trtllm_fp8` | 27.13 | 48.95 | 158.30 | 348.21 | 309.99 | 147.39 | 143.98 | 90.18 |
| `trt_bf16_trtllm_bf16` | 31.56 | 56.07 | 168.74 | 291.38 | 296.69 | 152.38 | 141.02 | 91.11 |
| `trt_bf16_trtllm_fp8` | 33.74 | 60.24 | 178.95 | **352.30** | 309.58 | 144.51 | 137.44 | 91.24 |
| `trt_bf16_trtllm_bf16_fp8kv` | 31.59 | 55.89 | 168.13 | 323.79 | 330.68 | 150.47 | 141.92 | 89.04 |
| `trt_fp8_trtllm_fp8` | **36.51** | **63.90** | **190.41** | 270.53 | 301.33 | 147.05 | 139.66 | 90.35 |

The exact-one-second results fall after c64 because 512 persistent clients and
short outputs shift the bottleneck to request scheduling/transport rather than
decoder arithmetic. The complete matrices are retained instead of selecting
isolated peaks.

## Reproduced environment

| Item | Measured value |
|---|---|
| Brev instance | `equivalent-purple-ostrich` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, compute capability 12.0 |
| Driver / CUDA runtime | 595.91.07 / 12.9 in container |
| Model revision | `e249bba6f7319c27912847fbbebb4258ead3b848` |
| TensorRT-LLM / TensorRT | 0.20.0 / 10.10.0.31 |
| NVIDIA Model Optimizer | 0.29.0 |
| PyTorch | 2.7.0a0+79aa17489c.nv25.04 |
| Dataset | Google FLEURS test, 48 clips, 496.78 seconds, 12 languages |
| Beam / max new tokens | 1 / 96 for every row |
| Server/client | FastAPI HTTP server and persistent concurrent HTTP client |

The fine-tuned decoder export contains all 146 expected tensors and has SHA-256
`727c0cf4b68d9cfac69da68e5e796ea4170d697d48a57d937a70b1853e8f81fa`.
Engine sizes and hashes are recorded in the consolidated report; large weights,
ONNX graphs, and device-specific TensorRT plans are intentionally excluded from
Git.

## Install and build

Use an interactive Brev SSH session and run long operations in GNU screen:

```bash
brev shell equivalent-purple-ostrich
screen -S shrutam2_work
mkdir -p /home/ubuntu/Optimize-shrutam2
cd /home/ubuntu/Optimize-shrutam2
```

Copy this repository to that directory, then build the HF/NeMo runtime and the
TensorRT-LLM/ModelOpt runtime:

```bash
ROOT=$PWD bash build_runtime.sh
ROOT=$PWD bash build_trtllm_runtime.sh
```

Download the pinned model, export the fine-tuned decoder, prepare the fixed
FLEURS quick set, export FP32/BF16 encoder ONNX, and generate real-audio
calibration tensors:

```bash
ROOT=$PWD bash setup_equivalent.sh
```

Build the three TensorRT-LLM decoder variants and both TensorRT encoder plans:

```bash
ROOT=$PWD bash build_trtllm_variants.sh
ROOT=$PWD bash build_equivalent_encoder_engines.sh
```

ModelOpt calibration uses 128 CNN/DailyMail sequences for the Llama FP8
weights. Encoder calibration uses the 48 real FLEURS waveforms after the actual
Shrutam log-mel frontend. The decoder builds use max batch 64, beam 1, paged KV,
removed input padding, context FMHA, and a global prompt-embedding table of
16,384 entries.

## Run the server/client benchmarks

Run the two HF control rows:

```bash
ROOT=$PWD VARIANT=equiv_base_fp32enc_hf_bf16llm_beam1 \
  RUNTIME=eager PRECISION=upstream bash run_equivalent_hf_variant.sh

ROOT=$PWD VARIANT=equiv_compile_bf16_hf_bf16llm_beam1 \
  RUNTIME=compile PRECISION=bf16 bash run_equivalent_hf_variant.sh
```

Run the BF16 PyTorch encoder plus BF16 TensorRT-LLM control:

```bash
ROOT=$PWD \
VARIANT=equiv_eager_bf16enc_trtllm_bf16_beam1 \
ENGINE_DIR=$PWD/artifacts/trtllm_equiv_bf16_engine \
ENCODER_RUNTIME=eager ENCODER_PRECISION=bf16 \
  bash run_equivalent_trtllm_variant.sh
```

Run the remaining five TensorRT-LLM/FP8 combinations and consolidate results:

```bash
ROOT=$PWD bash run_equivalent_remaining_trtllm.sh
python3 finalize_equivalent.py
```

Each row starts a real HTTP server, waits for a real-audio warm-up gate, then
runs variable concurrency `1,2,8,32,64,128,256`, exact-one-second concurrency
`1,2,8,32,64,128,256,512`, and sequential accuracy capture. Failed or empty
responses make the validation fail.

For a standalone deployment, select an engine explicitly:

```bash
ROOT=$PWD \
ENGINE_DIR=$PWD/artifacts/trtllm_equiv_fp8w_bf16kv_engine \
ENCODER_RUNTIME=trt \
TRT_ENGINE=$PWD/artifacts/shrutam2_encoder_modelopt_fp8.plan \
NUM_BEAMS=1 MAX_BATCH_SIZE=64 MAX_DELAY_MS=12 \
  bash start_server_trtllm.sh
```

Health and stats are available at `http://127.0.0.1:8092/health` and
`http://127.0.0.1:8092/stats`. Run `bash stop_server.sh` after HF serving, or
stop the TensorRT-LLM container with `docker rm -f shrutam2-trtllm-server`.

## Artifact map

- `results/equivalent_consolidated_results.json`: canonical all-row report,
  matrices, WER/CER pairs, runtime health, validation counts, and checksums.
- `results/equiv_*/`: raw request JSONL, per-workload CSV/JSON summaries,
  server stats, health, and accuracy pairs for all eight rows.
- `logs/*equiv*`: builds, servers, GPU telemetry, benchmark clients, ModelOpt,
  TensorRT, and finalization logs.
- `artifacts/equivalent_*.json`: environment, TensorRT-LLM configs, engine
  hashes, and setup provenance.
- `artifacts/equivalent_evidence_20260915.tgz`: checksum-preserved remote
  evidence bundle; SHA-256
  `2c0c306515f801303c9a6f4aaacfae3904898ac8291f8ab4c63fff2c1cc3981f`.
- `LEARNIGS.md`: decisions, failed approaches, runtime traps, and conclusions.

## Recommendation

Use `trt_fp8_trtllm_fp8` when maximum variable-audio throughput is the goal and
the calibrated mixed-FP8 encoder qualification is acceptable. Use
`trt_bf16_trtllm_fp8` when the encoder must remain BF16; it reached 1,081.29
RTFx and stayed within the allowed WER/CER budget. Use
`trt_bf16_trtllm_bf16_fp8kv` when matching baseline WER is more important than
peak throughput. FP8 KV did not provide a consistent speed benefit on this
short-context ASR workload.
