# Shrutam-2 Optimization Learnings

This file intentionally keeps the user-requested filename `LEARNIGS.md`.

## Bottom line

Shrutam-2 has a **FastConformer** speech encoder, an SMEAR projector, and a
fine-tuned Llama decoder. The biggest end-to-end gain on
`equivalent-purple-ostrich` came from replacing lockstep Hugging Face generation
with TensorRT-LLM continuous/paged decoding. Decoder FP8 was consistently useful
for variable-duration throughput. TensorRT acceleration of the FastConformer
provided smaller end-to-end gains because the log-mel frontend, SMEAR, request
scheduling, and short autoregressive decode remain outside the encoder plan.

The best variable-audio cell was 1,145.61 RTFx at c128 for the TensorRT
calibrated mixed-FP8 FastConformer plus ModelOpt FP8 TensorRT-LLM decoder and
FP8 KV cache. The HF baseline at the same concurrency was 320.87 RTFx, a 3.57x
increase. On the held-out IndicVoices Hindi evaluation its WER changed from
60.3809% to 59.3285% (-1.0524 percentage points), while CER changed from
51.5645% to 49.4316% (-2.1328 points).

## Controlled comparison discipline

Every throughput row used the same pinned model revision, 48-audio FLEURS
manifest, beam 1, maximum 96 output tokens, server/client timing definition,
and requested concurrency levels. Each row produced:

- 576 variable-duration timed requests at c1, c2, c8, c32, c64, c128, c256;
- 1,216 exact-one-second timed requests through c512;
- a 48-clip FLEURS transcript smoke, retained only as an encoder-calibration
  overlap check;
- raw request records, health, final server counters, client summaries, and GPU
  telemetry.

All 16,128 timed matrix requests across nine rows returned valid non-empty
transcripts. The canonical report independently checks expected levels,
request/valid counts, and nonzero failures before setting
`all_matrices_passed=true`.

Primary WER/CER use the official AI4Bharat IndicVoices Hindi validation file,
not the calibration set. It contained 5,530 source rows. Preparation removed
205 rows whose reference included `unintelligible` case-insensitively
(including `<unintelligible>`) and 370 rows outside the shared 0.5-19.0-second
input envelope. The result is 4,955 clips and 28,249.95 seconds (7.85 hours).
Every one of the nine rows attempted exactly those paths, returned 4,955
non-empty transcripts, and used the same Unicode NFKC plus whitespace
normalization. Across the accuracy comparison this is 44,595/44,595 successful
requests. The finalizer treats failed or empty output as an empty hypothesis,
so bad requests cannot make a row's WER look better through exclusion.

The 19-second upper bound is a serving-contract constraint: the Hindi prompt
uses 17 of the TensorRT-LLM engine's 256 input tokens, leaving 239 projected
audio tokens. An initial 20-second run demonstrated why the limit must be
enforced before adaptive batching: one oversized member caused a whole
microbatch to return HTTP 500. The final server now rejects over-limit audio
individually with HTTP 400 before it enters a batch.

## Fine-tuned Llama export is mandatory

The released base Llama directory is not the final decoder served by Shrutam-2.
The actual fine-tuned `llm.*` tensors live in `model.pt`. The exporter found all
146 expected decoder tensors, ignored only the tied extra `lm_head`, and wrote
a 2,470,662,568-byte BF16 safetensors file with SHA-256
`727c0cf4b68d9cfac69da68e5e796ea4170d697d48a57d937a70b1853e8f81fa`.
Building TensorRT-LLM from the unmodified base LLM would benchmark the wrong
model even if its server were healthy.

## TensorRT-LLM and Model Optimizer

The tested runtime was TensorRT-LLM 0.20.0, TensorRT 10.10.0.31, PyTorch
2.7.0a0 nv25.04, CUDA runtime 12.9, and NVIDIA Model Optimizer 0.29.0 on an
RTX PRO 6000 Blackwell Server Edition (compute capability 12.0).

Four decoder plans were built:

1. BF16 weights and BF16 KV cache;
2. ModelOpt FP8 weights and BF16 KV cache;
3. BF16 weights and ModelOpt FP8 KV cache;
4. ModelOpt FP8 weights and ModelOpt FP8 KV cache.

The FP8 checkpoint metadata records `producer.name=modelopt` and version
0.29.0. Decoder FP8 calibration used 128 CNN/DailyMail sequences, batch 8,
maximum sequence length 512. The engine used max batch 64, max input 256, max
sequence 512, beam 1, removed input padding, context FMHA, multiple profiles,
and paged KV cache.

The prompt-embedding table size is global across a batch. An initial 4,096
setting appeared sufficient for a 256-token per-sequence limit but effectively
allowed only 64 audio tokens at max batch 64. Real FLEURS warm-up failed. The
working build sets both `max_num_tokens` and
`max_prompt_embedding_table_size` to 16,384 (64 x 256), after which 25/25 real
audio smoke cases passed. Synthetic engine load tests would not have exposed
this multimodal-prefix capacity error.

FP8 weights materially improved the variable matrix over BF16 weights with the
same PyTorch BF16 encoder at c1, c2, c8, c64, c128, and c256. The c32 cell was
lower, showing that adaptive request grouping and short-output variability can
dominate a single cell. With a TensorRT BF16 encoder, FP8 decoder weights raised
the variable peak from 875.88 to 1,081.29 RTFx.

FP8 KV cache did not improve every cell, which is expected because short ASR
prompts and outputs do not create the long-lived KV bandwidth pressure seen in
long-text serving. It did help the strongest combined profile: mixed-FP8
FastConformer + FP8 decoder weights peaked at 1,145.61 RTFx with FP8 KV versus
1,105.05 with BF16 KV, a 3.67% gain. On the BF16 encoder/BF16 decoder-weight
control, FP8 KV changed the peak only from 875.88 to 879.97 RTFx. Treat it as a
profile-specific throughput and capacity option, not an automatic speedup.

## ModelOpt FP8 FastConformer

Encoder calibration used the real 48-waveform FLEURS set after Shrutam-2's own
log-mel preprocessor. Calibration tensors retained the ONNX contract:
`audio_signal` FP32 and `lengths` INT64. ModelOpt inserted 340 QuantizeLinear
and 607 DequantizeLinear nodes in the exported graph.

The CUDA ONNX Runtime calibration provider could not load an advanced cuDNN
library in this container. ModelOpt fell back to its requested CPU provider and
completed calibration. This affected build time, not the TensorRT execution
provider used for the final benchmark.

The first fully requested Conv+MatMul FP8 graph could not make every depthwise
convolution executable in TensorRT because some channel dimensions cannot be
padded safely without changing grouped-convolution semantics. The deployable
plan therefore uses FP8 for eligible operators and retains unsupported
depthwise convolutions at higher precision. Documentation calls this
**calibrated mixed-FP8** rather than mislabeling it as all-operator FP8.

The mixed-FP8 plan is 722,717,276 bytes versus 1,435,234,620 bytes for BF16.
At the TensorRT optimization profile, standalone `trtexec` GPU latency changed
from 13.0933 ms BF16 to 10.0633 ms mixed-FP8, about 23% faster. Both plans
passed the integrated real-audio server/client gate. The plan hashes are:

- BF16: `d70c163304847f2a99104e09845f411102c1505a7dc3a4a498cbef8f3098d3d5`;
- mixed-FP8: `54367a7a6589039368fde7250c0ec70546ea42151f8d0b9df7372ec64d0092f9`.

TensorRT plans are runtime- and GPU-specific. Rebuild them when TensorRT,
driver, GPU architecture, shape profiles, or plugin versions change.

## `torch.compile` was not a serving win here

The compiled BF16 HF row was slower than the FP32-encoder HF baseline at most
matrix points and peaked at 313.04 variable RTFx versus 334.69. Its
exact-one-second curve collapsed after c2. Dynamic batch/time shapes caused
recompilation or graph specialization overhead and the lockstep HF decoder
remained the dominant serving limitation. A warmed fixed-shape encoder microtest
can still favor compile, but it is not evidence that the full dynamic server is
faster.

Compiled BF16 quick WER changed by +1.1211 points and stayed within the user's
quality budget. It is retained as the requested measured row, not promoted.

## Server/client behavior matters

RTFx includes queueing and actual HTTP service, not just engine compute.
Persistent clients are preconnected before timing. The server batches the
frontend adaptively, while TensorRT-LLM independently schedules decoder work.
The observed engine batch sizes are preserved in server counters because
client concurrency is not equivalent to GPU batch size.

Variable-duration throughput rose until c128 and then fell at c256 on every
TensorRT-LLM profile. Exact-one-second throughput generally peaked at c32 or
c64 and dropped sharply at c128-c512. At those short durations, connection
fan-out, queue wakeups, Python scheduling, and microbatch formation outweigh
additional GPU parallelism. This is why the repository reports a matrix rather
than calling the largest concurrency the best configuration.

The FastConformer and TensorRT-LLM runners must cast inputs to the engine's
declared binding types. In particular, encoder `lengths` is INT64. Supplying an
INT32 pointer can deserialize and execute without an obvious engine error while
producing bad masks and invalid transcripts.

Readiness is a semantic gate. It performs real-audio warm-ups and exposes
encoder precision, decoder quantization, KV quantization, and engine path in
`/health`. Merely observing a running container or a loaded plan is not enough.

## Accuracy interpretation

All nine beam-1 WER deltas versus the Hugging Face base were within the accepted
1.5-percentage-point budget on the 4,955-clip IndicVoices Hindi validation set:

| Key | WER | CER | WER delta pp | CER delta pp |
|---|---:|---:|---:|---:|
| `base_hf` | 60.3809% | 51.5645% | 0.0000 | 0.0000 |
| `compiled_bf16` | 60.3936% | 51.5708% | +0.0127 | +0.0063 |
| `trtllm_bf16` | 59.8534% | 50.1186% | -0.5274 | -1.4459 |
| `trtllm_fp8` | 60.2287% | 50.5695% | -0.1521 | -0.9949 |
| `trt_bf16_trtllm_bf16` | 59.2436% | 49.5674% | -1.1373 | -1.9970 |
| `trt_bf16_trtllm_fp8` | 60.0195% | 50.2724% | -0.3613 | -1.2920 |
| `trt_bf16_trtllm_bf16_fp8kv` | 60.2389% | 50.4925% | -0.1420 | -1.0720 |
| `trt_fp8_trtllm_fp8` | 59.1142% | 49.0432% | -1.2666 | -2.5213 |
| `trt_fp8_trtllm_fp8_fp8kv` | 59.3285% | 49.4316% | -1.0524 | -2.1328 |

The absolute WER/CER are high and should not be mistaken for production Hindi
quality; they expose a substantial mismatch between Shrutam-2 and this
IndicVoices validation domain under only Unicode/whitespace normalization. The
controlled value of the table is the relative effect of runtime and precision
while sample identity, prompt, beam, and scoring remain fixed. A
customer-representative multilingual evaluation is still required before a
production quality claim.

## Deployment choices

- Maximum variable throughput: calibrated mixed-FP8 TensorRT FastConformer +
  ModelOpt FP8 TensorRT-LLM weights/FP8 KV; 1,145.61 peak RTFx and -1.0524 WER
  points on the held-out Hindi comparison.
- Conservative encoder precision: TensorRT BF16 FastConformer + ModelOpt FP8
  TensorRT-LLM weights/BF16 KV; 1,081.29 peak RTFx and -0.3613 WER points.
- Conservative decoder precision: TensorRT BF16 FastConformer + BF16
  TensorRT-LLM weights/BF16 KV; 875.88 peak RTFx and -1.1373 WER points.
- Do not promote the dynamic compiled HF server for this shape mix.

## Evidence and reproducibility

`results/equivalent_consolidated_results.json` is the canonical source for all
numbers. `results/indicvoices_hindi_valid_accuracy.json` preserves the common
sample hashes, all-sample scoring policy, deltas, and failure counts. The nine
`results/equiv_*` trees contain raw requests, summaries, health, stats, and
transcript pairs. `logs/*equiv*` and `logs/indicvoices*` preserve builds, engine
output, server logs, benchmark clients, accuracy, and GPU telemetry.

The retrieved remote evidence archive is
`artifacts/equivalent_evidence_20260916.tgz`, SHA-256
`a204b400fe1f29c7d22abe5ad4d1e7bdaf371abe22ce2f77b648d93334199f20`.
Weights, ONNX graphs, calibration arrays, and TensorRT plans are not committed;
the README gives the exact commands to regenerate them.
