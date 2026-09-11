# Optimize Shrutam-2

Measured server/client optimization of `bharatgenai/Shrutam-2` on GPU 0 of
`minimum-fuchsia-mule`. The promoted maximum-throughput deployment runs the
FastConformer encoder in TensorRT 10.10 BF16, keeps SMEAR in BF16, and serves
the fine-tuned Llama decoder through vLLM BF16 continuous batching. It uses
adaptive length-aware frontend batches through 64, 512 persistent client
streams, and an extended server keep-alive. The earlier compiled four-beam path
remains the quality-oriented alternative.

All requested matrices completed with zero failed requests in the final runs.
Raw request records, health responses, server batch counters, GPU telemetry,
accuracy pairs, build logs, and checksums are under `results/`, `logs/`, and
`artifacts/`.

## Reproduced environment

| Item | Value |
|---|---|
| Brev instance | `minimum-fuchsia-mule` |
| Benchmark GPU | GPU 0, NVIDIA H100 NVL, 95,830 MiB |
| Driver | 595.71.05 |
| Shrutam-2 revision | `e249bba6f7319c27912847fbbebb4258ead3b848` |
| Baseline-code revision | `93586a8d949d364f26c600e2f5213a9cdccf69aa` |
| Canary/vLLM reference revision | `ae29af4502c9ac52eb88dbbbd6df721e4d8cac82` |
| vLLM / PyTorch | vLLM 0.12.0 / PyTorch 2.9.1+cu129 |
| HTTP runtime image | `shrutam2-runtime:25.02`, based on `nvcr.io/nvidia/nemo:25.02` |
| TensorRT-LLM image | `nvcr.io/nvidia/tensorrt-llm/release:0.20.0` |
| TensorRT engine builder | TensorRT 10.10.0.31 |
| Promoted HTTP TensorRT | TensorRT 10.10.0.31 in the TensorRT-LLM 0.20 image |
| Data | 46 IndicVoices clips, 12 supported languages, 16 kHz |

GPU 0 was isolated for these runs. GPU 1 continued to host an unrelated Indic
Flex service and was never selected by the Shrutam containers.

## Measurement definition

End-to-end RTFx is:

```text
sum(valid input audio seconds) / client wall-clock seconds
```

It includes request upload on an already-established HTTP connection, server queueing, log-mel preprocessing,
Conformer, SMEAR projection, autoregressive Llama decoding, and response
parsing. File loading, PCM conversion, and connection establishment are outside
the timed interval, matching a persistent server/client deployment. The final
client preconnects all requested logical streams, including all 512 before the
c512 timed interval. The exact-1-second matrix trims or zero-pads each
request to exactly 16,000 samples. Concurrency is client concurrency, not an
assertion that the engine saw the same batch size. Actual microbatches are
recorded in each `server_stats.json`; the adaptive frontend uses batch 32 or 64,
a 5 ms low-load window, and a 50 ms high-load gather window.

Encoder-only RTFx is explicitly separated because it excludes HTTP,
preprocessing, SMEAR, and Llama decoding. TensorRT `trtexec` values use GPU
compute time, CUDA graphs, disabled host transfers, 100 timed iterations, and
exact-1-second shapes.

## Install

All remote work was performed in persistent GNU screen sessions.

```bash
brev shell minimum-fuchsia-mule
screen -S shrutam2_work
mkdir -p /home/nvidia/Optimize-shrutam2
cd /home/nvidia/Optimize-shrutam2
```

Copy this directory to the instance, then build the pinned runtime:

```bash
bash build_runtime.sh
```

Download the pinned model snapshot. No token is required for the currently
public repository; if access policy changes, authenticate on the host without
putting a token into a command or log.

```bash
docker run --rm -v "$PWD:/workspace" nvcr.io/nvidia/nemo:25.02 \
  huggingface-cli download bharatgenai/Shrutam-2 \
  --revision e249bba6f7319c27912847fbbebb4258ead3b848 \
  --local-dir /workspace/model
```

Expected checkpoint hashes are retained in `logs/model_sha256.txt`.

## Prepare the quick dataset

The measured manifest was recovered from cached IndicVoices Arrow data because
the source audio/reference pairs already existed on this host:

```bash
docker run --rm \
  -v /home/nvidia/.cache:/host_cache:ro \
  -v "$PWD:/workspace" -w /workspace shrutam2-runtime:25.02 \
  python prepare_cached_indicvoices.py \
    --cache-root /host_cache/huggingface/datasets/parquet \
    --output-dir /workspace/data/indicvoices_quick \
    --samples-per-language 4
```

The resulting manifest has 46 unique clips across Assamese, Bengali, Gujarati,
Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and Urdu.

## Install and run the maximum-throughput vLLM server/client

The vLLM path was inspired by the Canary/Qwen separation in
[wuxuedaifu/Canary-Qwen-2.5b-vllm](https://github.com/wuxuedaifu/Canary-Qwen-2.5b-vllm):
compute audio prompt embeddings in the ASR frontend, then let vLLM continuously
batch independent decoder requests.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv

mkdir -p baseline
git clone https://github.com/wuxuedaifu/Canary-Qwen-2.5b-vllm \
  baseline/Canary-Qwen-2.5b-vllm
git -C baseline/Canary-Qwen-2.5b-vllm checkout \
  ae29af4502c9ac52eb88dbbbd6df721e4d8cac82

bash install_vllm.sh
source .venv-vllm/bin/activate
python export_vllm_checkpoint.py
python smoke_vllm_prompt_embeds.py \
  --model-dir artifacts/vllm_llm_bf16 \
  --output artifacts/vllm_prompt_embed_smoke.json
```

The export step is mandatory: `model/model.pt` contains 147 fine-tuned
`llm.*` tensors. Pointing vLLM at the released base `model/llm` directory
instead produced 272% WER. The exported BF16 decoder contains 146 expected
tensors and is checksummed in `artifacts/vllm_llm_bf16/export_report.json`.

Start and benchmark in separate persistent remote screen sessions:

```bash
screen -L -Logfile logs/server_vllm_adaptive_final_screen.log \
  -dmS shrutam2_vllm_server bash -lc \
  'cd /home/nvidia/Optimize-shrutam2; ./start_server_vllm.sh'

until curl -fsS http://127.0.0.1:8092/health; do sleep 2; done

screen -L -Logfile logs/run_vllm_adaptive_final_screen.log \
  -dmS shrutam2_vllm_bench bash -lc \
  'cd /home/nvidia/Optimize-shrutam2; ./run_vllm_benchmarks.sh'

tail -f logs/run_vllm_adaptive_final_screen.log
screen -S shrutam2_vllm_server -X quit
```

The BF16-control launcher deploys GPU 0 only: eager BF16 FastConformer/SMEAR,
vLLM BF16 with CUDA graphs and chunked prefill, 512 decoder sequences, adaptive
32/64 frontend microbatches, a 256-request length lookahead, and 120-second HTTP
keep-alive. `benchmark_client.py` rejects empty/invalid responses and records
every latency, transcript, generated-token count, batch size, and retry count.

Build and serve the promoted TensorRT-encoder plus vLLM configuration:

```bash
docker run --rm --gpus device=0 -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 bash -lc \
  'ROOT=/workspace \
   ENGINE=/workspace/artifacts/shrutam2_encoder_bf16_b64.plan \
   CACHE=/workspace/artifacts/shrutam2_encoder_bf16_b64.cache \
   TRTEXEC=/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec \
   PRECISION=bf16 OPT_BATCH=32 MAX_BATCH=64 bash build_trt_encoder.sh'

screen -L -Logfile logs/server_vllm_trt_b64_screen.log \
  -dmS shrutam2_vllm_trt bash -lc \
  'cd /home/nvidia/Optimize-shrutam2; \
   TRT_ENGINE=$PWD/artifacts/shrutam2_encoder_bf16_b64.plan \
   BASE_FRONTEND_BATCH=32 MAX_FRONTEND_BATCH=64 \
   bash start_server_vllm_trt_container.sh'

until curl -fsS http://127.0.0.1:8092/health; do sleep 2; done
screen -L -Logfile logs/bench_vllm_trt_b64_screen.log \
  -dmS shrutam2_vllm_trt_bench bash -lc \
  'cd /home/nvidia/Optimize-shrutam2; \
   VARIANT=vllm_trt_bf16enc_bf16llm_adaptive64_persistent_beam1_final \
   bash run_vllm_benchmarks.sh'
tail -f logs/bench_vllm_trt_b64_screen.log
```

The TensorRT launcher uses the TensorRT 10.10 libraries from the TensorRT-LLM
0.20 image while mounting the tested vLLM 0.12/PyTorch 2.9 virtual environment.
The runner casts each binding to the engine-declared type; in particular, the
exported ONNX length input is INT64.

The valid FP8-weight/BF16-KV experiment can be reproduced with:

```bash
RUNTIME=eager LLM_QUANTIZATION=fp8 KV_CACHE_DTYPE=auto \
  bash start_server_vllm.sh
```

FP8 weights plus FP8 KV cache was also tested with
`KV_CACHE_DTYPE=fp8` and `CALCULATE_KV_SCALES=1`, but failed the real-audio
validity gate and is not a deployable result.

## Run the quality-oriented HTTP server/client benchmark

```bash
RUNTIME=compile PRECISION=bf16 NUM_BEAMS=4 MAX_NEW_TOKENS=200 \
  bash start_server.sh

VARIANT=optimized_compile_bf16_beam4 bash run_http_benchmarks.sh
bash stop_server.sh
```

`start_server.sh` persists the Inductor cache in
`artifacts/torchinductor_cache`. Its readiness gate includes warm-up batches
before clients are admitted. `run_http_benchmarks.sh` executes:

- variable-duration concurrency `1,2,8,32,64,128,256`;
- exact-1-second concurrency `1,2,8,32,64,128,256,512`;
- sequential transcript capture and WER/CER calculation.

For the maximum-throughput, lower-accuracy profile:

```bash
RUNTIME=eager PRECISION=bf16 NUM_BEAMS=1 MAX_NEW_TOKENS=96 bash start_server.sh
VARIANT=optimized_eager_bf16_beam1 bash run_http_benchmarks.sh
bash stop_server.sh
```

## End-to-end HTTP results

RTFx, variable-duration audio:

| Variant | c1 | c2 | c8 | c32 | c64 | c128 | c256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Upstream FP32 Conformer, BF16 LLM, beam 4 | 16.16 | 19.72 | 38.16 | 78.02 | 99.93 | 97.46 | 138.31 |
| Compiled BF16, beam 4 | 19.09 | 26.80 | 47.79 | 101.67 | 115.37 | 113.69 | 120.53 |
| Eager BF16, beam 1 throughput profile | 22.22 | 29.55 | 32.49 | 89.95 | 116.87 | 132.80 | 183.37 |
| vLLM adaptive persistent, BF16 beam 1 control | 32.96 | 57.41 | 97.49 | 164.22 | 314.13 | 423.81 | 550.85 |
| vLLM FP8 weights, BF16 KV, eager BF16 encoder | 33.42 | 51.95 | 102.84 | 184.03 | 301.52 | 394.25 | 449.16 |
| **TensorRT BF16 encoder + vLLM BF16, adaptive 64** | 32.86 | 67.97 | 124.04 | 220.10 | 366.53 | 408.63 | 566.34 |

RTFx, exact-1-second audio:

| Variant | c1 | c2 | c8 | c32 | c64 | c128 | c256 | c512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Upstream FP32 Conformer, BF16 LLM, beam 4 | 9.54 | 24.41 | 46.94 | 90.67 | 73.98 | 84.66 | 48.61 | 83.60 |
| Compiled BF16, beam 4 | 15.30 | 16.82 | 40.17 | 76.44 | 101.55 | 49.77 | 47.56 | 68.03 |
| Eager BF16, beam 1 throughput profile | 17.20 | 28.94 | 53.55 | 120.79 | 117.56 | 92.05 | 48.49 | 73.61 |
| vLLM adaptive persistent, BF16 beam 1 control | 12.56 | 21.12 | 51.25 | 98.01 | 111.85 | 112.91 | 119.98 | 95.59 |
| vLLM FP8 weights, BF16 KV, eager BF16 encoder | 12.51 | 22.74 | 53.19 | 101.62 | 120.25 | 112.15 | 113.40 | 102.20 |
| **TensorRT BF16 encoder + vLLM BF16, adaptive 64** | 17.13 | 27.99 | 60.39 | 135.80 | 113.12 | 106.20 | 158.34 | 138.08 |

The original vLLM row is a single cohesive control run. Against the earlier eager BF16
beam-1 path, variable-duration RTFx improved by 1.48x, 1.94x, 3.00x, 1.83x,
2.69x, 3.19x, and 3.00x at c1 through c256. Exact-1-second throughput favors
the old path below c128, but vLLM improves c128/c256/c512 by 1.23x/2.47x/1.30x.
The promoted TensorRT/vLLM row is another clean cohesive run: all 1,792 timed
matrix responses were valid, with no failures or retries. Relative to the BF16
vLLM control it reached 1.00x/1.18x/1.27x/1.34x/1.17x/0.96x/1.03x on variable
audio and 1.36x/1.33x/1.18x/1.39x/1.01x/0.94x/1.32x/1.44x on one-second audio.
The c128 regressions are reported rather than hidden; the integrated path wins
most of the requested cells and the highest-concurrency endpoints.

The upstream c256 variable run initially had one transport `ReadError` and was
excluded. The table uses the clean 256/256 retry at 138.31 RTFx; both runs are
retained in the artifacts.

## Quick accuracy

| Path | Beams | WER | CER | WER change vs upstream |
|---|---:|---:|---:|---:|
| Upstream FP32 Conformer + BF16 LLM | 4 | 29.412% | 17.334% | — |
| **Compiled BF16 end-to-end** | 4 | **29.630%** | **17.522%** | **+0.218 pp** |
| Eager BF16 throughput profile | 1 | 32.462% | 19.918% | +3.050 pp |
| vLLM adaptive persistent BF16 control | 1 | 31.808% | 18.233% | +2.396 pp |
| vLLM FP8 weights + BF16 KV | 1 | 32.244% | 17.746% | +2.832 pp |
| **TensorRT BF16 encoder + vLLM BF16** | 1 | **32.462%** | **18.233%** | **+3.050 pp** |

The 46-clip set is a quick regression check, not a publication-grade model
evaluation. The compiled BF16 profile remains the quality-oriented result
because it keeps the same four-beam search and changes WER by only 0.218
percentage points. The promoted TensorRT/vLLM profile changes WER by +0.654
percentage points relative to its BF16 vLLM control, inside the accepted 1.5 pp
budget. FP8 weights with BF16 KV also passed that gate (+0.436 pp versus the
control), but did not improve the overall throughput matrix. The control's
generated-token maximum was 49 versus the configured limit of 96, so no
measured response was truncated by that limit.

## Conformer encoder optimization

Build and test the BF16 ONNX/TensorRT engine:

```bash
docker run --rm --gpus device=0 --ipc=host -v "$PWD:/workspace" \
  -w /workspace shrutam2-runtime:25.02 \
  python export_encoder_onnx.py --model-dir /workspace/model

docker run --rm --gpus device=0 --ipc=host -v "$PWD:/workspace" \
  -w /workspace nvcr.io/nvidia/tensorrt-llm/release:0.20.0 \
  bash build_trt_encoder.sh

bash benchmark_trtexec.sh
```

For FP16, export FP32 ONNX weights and set `PRECISION=fp16`, `ENGINE`, and
`CACHE` when invoking `build_trt_encoder.sh`; the exact command is preserved in
`logs/trt_encoder_fp16_build.log`.

Build the AOTInductor package and run the exact-shape comparison:

```bash
docker run --rm --gpus device=0 --ipc=host -v "$PWD:/workspace" \
  -w /workspace shrutam2-runtime:25.02 \
  python export_encoder_aoti.py \
    --model-dir /workspace/model \
    --static-frames 101 --max-batch 256 \
    --output /workspace/artifacts/shrutam2_encoder_bf16_1s.pt2 \
    --report /workspace/artifacts/aoti_1s_build_report.json

bash run_encoder_1s.sh
```

Exact-1-second encoder-only RTFx:

| Runtime | b1 | b2 | b8 | b32 | b64 | b128 | b256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch eager BF16 | 41.51 | 80.97 | 321.20 | 1202.14 | 2484.56 | 3832.08 | 4408.51 |
| `torch.compile` BF16 | 267.24 | 393.33 | 1348.75 | 5030.27 | 4054.32 | 4630.07 | 4989.11 |
| AOTI BF16, static time | 240.27 | 412.79 | 1396.80 | 5198.26 | 4199.20 | 4845.45 | 5205.14 |
| TensorRT BF16 `trtexec` | 224.43 | 444.75 | 1692.17 | 5295.66 | — | — | — |
| TensorRT FP16 `trtexec` | 266.14 | 531.42 | 2054.05 | **7128.30** | — | — | — |

The original TensorRT plan was capped at batch 32. The promoted engine expands
its min/opt/max profile to batches 1/32/64 and 101/501/1001 mel frames. Its
remote-only plan is 1,416,359,964 bytes with SHA-256
`bc3b1ad64041b7038086f52778fede0b8533a2a233ba42d23817780560aa00ef`.
Full dynamic-time AOTI export failed on symbolic divisibility guards introduced
by the 8x convolutional subsampler, so the valid AOTI artifact remains static in
time (101 mel frames) and dynamic only in batch.

The NeMo 25.02 HTTP image contains TensorRT 10.8 and cannot deserialize a 10.10
plan. `start_server_vllm_trt_container.sh` resolves that mismatch by using the
TensorRT-LLM 0.20/TensorRT 10.10 image while mounting the vLLM environment. A
second integration bug was equally important: ONNX exported the length binding
as INT64, while the old runner supplied an INT32 pointer. Casting every binding
to the engine-declared type fixed empty transcripts. The main HTTP tables are
now genuine integrated server/client results; the `trtexec` table remains
encoder-stage-only evidence.

## TensorRT-LLM BF16 and ModelOpt FP8

Shrutam-2 passes projected audio vectors directly to Llama as prefix
embeddings. The engine must therefore enable a prompt-embedding table; a
text-only decoder smoke test is not valid evidence for this model.

```bash
bash probe_trtllm.sh
bash build_trtllm_llm.sh

docker run --rm --gpus device=0 --ipc=host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt-llm/release:0.20.0 \
  python trtllm_audio_smoke.py \
    --model-dir /workspace/model \
    --engine-dir /workspace/artifacts/trtllm_llm_bf16_engine \
    --manifest /workspace/data/indicvoices_quick/manifest.jsonl \
    --output /workspace/results/trtllm_bf16/accuracy_requests_beam4.jsonl \
    --max-new-tokens 200 --num-beams 4

bash build_modelopt_fp8_llm.sh
```

The real-audio prompt-table accuracy results were:

| Decoder engine | WER | CER | FP8 minus BF16 WER |
|---|---:|---:|---:|
| TensorRT-LLM BF16, beam 4 | 31.808% | 18.308% | — |
| TensorRT-LLM ModelOpt FP8 + FP8 KV, beam 4 | 30.719% | 16.061% | -1.089 pp |

FP8 produced no directional WER regression. However, the absolute WER movement
is 1.089 percentage points, narrowly outside a strict `abs(FP8-BF16) < 1 pp`
equivalence rule. The ModelOpt engine and evidence are retained as exploratory,
but this README does not promote it as strict quality-equivalent on such a
small set. Re-run on a larger, fixed evaluation manifest before deployment.

## Result map

- `results/consolidated_results.json`: machine-readable tables and pass/fail
  validation for every reported server/client matrix.
- `results/vllm_continuous_bf16_adaptive_persistent_beam1/`: the BF16 vLLM
  control matrices, accuracy pairs, and server counters.
- `results/vllm_trt_bf16enc_bf16llm_adaptive64_persistent_beam1_final/`: the
  promoted integrated TensorRT/vLLM result.
- `results/vllm_fp8w_bf16kv_adaptive_persistent_beam1/`: valid FP8-weight,
  BF16-KV evidence; the rejected FP8-KV health/log evidence is retained under
  `results/vllm_fp8w_fp8kv_adaptive_persistent_beam1/` and `logs/`.
- `artifacts/provenance.json`: host, revisions, and SHA-256 hashes for every
  large engine/package.
- `results/*/*/requests.jsonl`: per-request latency and transcript evidence.
- `results/*/*/server_stats.json`: actual dynamic-batch histograms.
- `logs/`: complete build, server, client, accuracy, GPU, and failure logs.
- `LEARNIGS.md`: implementation decisions, failures, and boundaries.

Large ONNX, AOTI, TensorRT, and TensorRT-LLM binaries are not duplicated in the
local evidence bundle. Their byte sizes and hashes are recorded in
`artifacts/provenance.json`, and all build scripts needed to reproduce them are
included here.
