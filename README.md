# Optimize Shrutam-2

Measured server/client optimization of `bharatgenai/Shrutam-2` on the Brev
instance `equivalent-purple-ostrich`. Shrutam-2 uses a **FastConformer encoder**,
not a plain Conformer encoder, followed by the SMEAR projector and a fine-tuned
Llama decoder.

All nine requested beam-1 configurations completed both throughput matrices.
There were zero failed requests among 16,128 timed matrix requests and among
44,595 held-out accuracy requests.
The consolidated machine-readable report is
[`results/equivalent_consolidated_results.json`](results/equivalent_consolidated_results.json).

## Result summary

The maximum variable-audio result was **1,145.61 RTFx at c128** with the
calibrated mixed-FP8 TensorRT FastConformer and Model Optimizer FP8
TensorRT-LLM decoder using FP8 KV cache. The best exact-1-second c1 result was
37.79 RTFx with the same profile.

The encoder FP8 label is deliberately qualified as *mixed-FP8*: NVIDIA Model
Optimizer inserted FP8 Q/DQ for eligible MatMul and convolution operators, but
TensorRT retained unsupported depthwise convolutions at higher precision. This
is the deployable calibrated engine that passed real-audio transcript gates,
not a claim that every encoder operator ran in FP8.

| Key | Requested configuration, beam 1 | WER | CER | WER delta pp | CER delta pp | Peak variable RTFx |
|---|---|---:|---:|---:|---:|---:|
| `base_hf` | Base: FP32 FastConformer + HF BF16 LLM | 60.3809% | 51.5645% | 0.0000 | 0.0000 | 334.69 |
| `compiled_bf16` | Compiled BF16 FastConformer + HF BF16 LLM | 60.3936% | 51.5708% | +0.0127 | +0.0063 | 313.04 |
| `trtllm_bf16` | PyTorch BF16 FastConformer + TensorRT-LLM BF16 | 59.8534% | 50.1186% | -0.5274 | -1.4459 | 862.57 |
| `trtllm_fp8` | PyTorch BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 60.2287% | 50.5695% | -0.1521 | -0.9949 | 983.91 |
| `trt_bf16_trtllm_bf16` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 | 59.2436% | 49.5674% | -1.1373 | -1.9970 | 875.88 |
| `trt_bf16_trtllm_fp8` | TensorRT BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 60.0195% | 50.2724% | -0.3613 | -1.2920 | 1,081.29 |
| `trt_bf16_trtllm_bf16_fp8kv` | TensorRT BF16 FastConformer + TensorRT-LLM BF16 weights, FP8 KV | 60.2389% | 50.4925% | -0.1420 | -1.0720 | 879.97 |
| `trt_fp8_trtllm_fp8` | TensorRT calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights, BF16 KV | 59.1142% | 49.0432% | -1.2666 | -2.5213 | **1,105.05** |
| `trt_fp8_trtllm_fp8_fp8kv` | TensorRT calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights, FP8 KV | 59.3285% | 49.4316% | -1.0524 | -2.1328 | **1,145.61** |

Primary WER/CER use all 4,955 usable clips (7.85 hours) from the official
IndicVoices Hindi validation split. All rows use the same paths and references;
failed or empty transcripts would be scored as empty hypotheses rather than
silently excluded. The source split had 5,530 rows. Cleaning removed 205 rows
whose reference contained `unintelligible` case-insensitively (including
`<unintelligible>`) and 370 rows outside 0.5-19.0 seconds. The upper bound keeps
every row within the shared 256-token TensorRT-LLM input contract. The absolute
error rates show the domain mismatch on this split; signed deltas isolate the
runtime/precision change. The earlier 48-clip FLEURS comparison is retained in
the consolidated JSON only as an overlapping calibration smoke test.

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
| `trt_fp8_trtllm_fp8_fp8kv` | 104.77 | 171.76 | 478.36 | 593.50 | 969.56 | **1,145.61** | 719.20 |

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
| `trt_fp8_trtllm_fp8_fp8kv` | **37.79** | **66.18** | **194.52** | **352.81** | 318.05 | 155.03 | 136.22 | **92.40** |

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
| Throughput / encoder calibration | Google FLEURS test, 48 clips, 496.78 seconds, 12 languages |
| Held-out accuracy | AI4Bharat IndicVoices, Hindi validation, 4,955 cleaned clips, 28,249.95 seconds |
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

Build the four TensorRT-LLM decoder variants and both TensorRT encoder plans:

```bash
ROOT=$PWD bash build_trtllm_variants.sh
ROOT=$PWD bash build_equivalent_encoder_engines.sh
```

ModelOpt calibration uses 128 CNN/DailyMail sequences for the Llama FP8
weights. Encoder calibration uses the 48 real FLEURS waveforms after the actual
Shrutam log-mel frontend. The decoder builds use max batch 64, beam 1, paged KV,
removed input padding, context FMHA, and a global prompt-embedding table of
16,384 entries.

Prepare the held-out Hindi accuracy set. Supply a Hugging Face read token via
the environment; the token is used for the download and is not written to an
artifact:

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
docker run --rm -e HF_TOKEN -v "$PWD:/workspace" -w /workspace \
  shrutam2-trtllm:0.20 python prepare_indicvoices_hindi_eval.py
unset HF_TOKEN
```

The preparation script reads only `hindi/valid-00000-of-00001.parquet`, checks
decoded audio, removes references containing `unintelligible`, enforces the
common 0.5-19.0-second input envelope, and records every exclusion in
`data/indicvoices_hindi_valid/report.json`.

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

Run the remaining six TensorRT-LLM/FP8 combinations and consolidate throughput
results:

```bash
ROOT=$PWD bash run_equivalent_remaining_trtllm.sh
python3 finalize_equivalent.py
```

Each row starts a real HTTP server, waits for a real-audio warm-up gate, then
runs variable concurrency `1,2,8,32,64,128,256`, exact-one-second concurrency
`1,2,8,32,64,128,256,512`, and sequential accuracy capture. Failed or empty
responses make the validation fail.

Run the same cleaned IndicVoices Hindi validation manifest through all nine
servers at concurrency 32, then calculate the canonical all-sample WER/CER
table and merge it into the consolidated report:

```bash
ROOT=$PWD bash run_all_indicvoices_hindi_eval.sh
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  shrutam2-trtllm:0.20 bash -lc \
  'python finalize_indicvoices_hindi_accuracy.py && python finalize_equivalent.py'
```

The TensorRT-LLM server rejects audio longer than 19.0 seconds before adaptive
batching, so a single invalid request cannot contaminate an otherwise valid
batch.

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

- `results/equivalent_consolidated_results.json`: canonical nine-row report,
  matrices, WER/CER pairs, runtime health, validation counts, and checksums.
- `results/indicvoices_hindi_valid_accuracy.json`: held-out Hindi comparison,
  sample-integrity digests, signed deltas, and failure accounting.
- `results/equiv_*/`: per-workload throughput request JSONL, CSV/JSON summaries,
  server stats, health, and FLEURS smoke pairs for all nine rows. The larger
  per-row IndicVoices transcript dumps are inside the evidence archive.
- `logs/*equiv*`: builds, servers, GPU telemetry, benchmark clients, ModelOpt,
  TensorRT, and finalization logs.
- `artifacts/equivalent_*.json`: environment, TensorRT-LLM configs, engine
  hashes, and setup provenance.
- `artifacts/equivalent_evidence_20260916.tgz`: checksum-preserved remote
  evidence bundle; SHA-256
  `a204b400fe1f29c7d22abe5ad4d1e7bdaf371abe22ce2f77b648d93334199f20`.
- `LEARNIGS.md`: decisions, failed approaches, runtime traps, and conclusions.

## Recommendation

Use `trt_fp8_trtllm_fp8_fp8kv` when maximum variable-audio throughput is the
goal and the calibrated mixed-FP8 encoder qualification is acceptable. It
reached 1,145.61 RTFx and its IndicVoices Hindi WER was 1.0524 percentage points
lower than the HF baseline on this evaluation. Use
`trt_bf16_trtllm_fp8` when the encoder must remain BF16; it reached 1,081.29
RTFx and its WER delta was -0.3613 points. FP8 KV is not a universal win on this
short-context ASR workload, but with the mixed-FP8 encoder plus FP8-weight
decoder it raised the variable-audio peak by 3.67% over the otherwise-identical
BF16-KV row (1,145.61 versus 1,105.05 RTFx).
