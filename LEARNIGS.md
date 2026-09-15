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

The best variable-audio cell was 1,105.05 RTFx at c128 for the TensorRT
calibrated mixed-FP8 FastConformer plus ModelOpt FP8 TensorRT-LLM decoder. The
HF baseline at the same concurrency was 320.87 RTFx, a 3.44x increase. Its
quick WER changed from 19.5067% to 19.2825% (-0.2242 percentage points), while
CER changed from 5.9750% to 6.2075% (+0.2326 points).

## Controlled comparison discipline

Every row used the same pinned model revision, 48-audio manifest, beam 1,
maximum 96 output tokens, server/client timing definition, and requested
concurrency levels. Each row produced:

- 576 variable-duration timed requests at c1, c2, c8, c32, c64, c128, c256;
- 1,216 exact-one-second timed requests through c512;
- 64 sequential accuracy requests, from which the 48 unique manifest samples
  were scored;
- raw request records, health, final server counters, client summaries, and GPU
  telemetry.

All 14,336 timed matrix requests across eight rows returned valid non-empty
transcripts. The canonical report independently checks expected levels,
request/valid counts, and nonzero failures before setting
`all_matrices_passed=true`.

This is a quick comparative accuracy gate, not a publishable full-dataset WER.
The original cached IndicVoices audio was unavailable on this fresh host and
the gated source could not be reconstructed without credentials. Substituting
a fixed public FLEURS test set was preferable to silently changing samples
between rows. It contains 48 clips, 496.78 seconds, and four clips each for
Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi,
Tamil, Telugu, and Urdu.

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

Three decoder plans were built:

1. BF16 weights and BF16 KV cache;
2. ModelOpt FP8 weights and BF16 KV cache;
3. BF16 weights and ModelOpt FP8 KV cache.

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

FP8 KV cache did not consistently improve this workload. It preserved baseline
WER and slightly improved quick CER, but short ASR prompts and short generated
sequences do not create the long-lived KV bandwidth pressure seen in long-text
LLM serving. Its throughput was close to, and often below, the BF16-KV control.
It remains a memory-capacity option rather than the promoted speed path.

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

All eight beam-1 WER deltas versus the base were within +1.5 percentage points:

| Key | WER | CER | WER delta pp | CER delta pp |
|---|---:|---:|---:|---:|
| `base_hf` | 19.5067% | 5.9750% | 0.0000 | 0.0000 |
| `compiled_bf16` | 20.6278% | 6.5474% | +1.1211 | +0.5725 |
| `trtllm_bf16` | 20.2915% | 6.4043% | +0.7848 | +0.4293 |
| `trtllm_fp8` | 19.7309% | 6.2791% | +0.2242 | +0.3041 |
| `trt_bf16_trtllm_bf16` | 19.9552% | 6.1896% | +0.4484 | +0.2147 |
| `trt_bf16_trtllm_fp8` | 20.6278% | 6.9410% | +1.1211 | +0.9660 |
| `trt_bf16_trtllm_bf16_fp8kv` | 19.5067% | 5.9571% | 0.0000 | -0.0179 |
| `trt_fp8_trtllm_fp8` | 19.2825% | 6.2075% | -0.2242 | +0.2326 |

Small negative or positive changes on 48 clips should not be interpreted as a
general quality improvement or regression. They establish that no large
precision-induced failure appeared. Before production, repeat on a larger
customer-representative multilingual test set and report language-wise WER/CER.

## Deployment choices

- Maximum variable throughput: calibrated mixed-FP8 TensorRT FastConformer +
  ModelOpt FP8 TensorRT-LLM weights/BF16 KV.
- Conservative encoder precision: TensorRT BF16 FastConformer + ModelOpt FP8
  TensorRT-LLM weights/BF16 KV; 1,081.29 peak RTFx and +1.1211 WER points.
- Baseline-WER option: TensorRT BF16 FastConformer + BF16 TensorRT-LLM weights
  with FP8 KV; no measured WER change, but no consistent speed benefit from KV
  quantization.
- Do not promote the dynamic compiled HF server for this shape mix.

## Evidence and reproducibility

`results/equivalent_consolidated_results.json` is the canonical source for all
numbers. The eight `results/equiv_*` trees contain raw requests, summaries,
health, stats, and transcript pairs. `logs/*equiv*` preserves builds, engine
output, server logs, benchmark clients, accuracy, and GPU telemetry.

The retrieved remote evidence archive is
`artifacts/equivalent_evidence_20260915.tgz`, SHA-256
`2c0c306515f801303c9a6f4aaacfae3904898ac8291f8ab4c63fff2c1cc3981f`.
Weights, ONNX graphs, calibration arrays, and TensorRT plans are not committed;
the README gives the exact commands to regenerate them.
