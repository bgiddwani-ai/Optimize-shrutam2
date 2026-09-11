# Shrutam-2 Optimization Learnings

This file intentionally keeps the requested filename `LEARNIGS.md`.

## What was optimized

Shrutam-2 combines a 24-layer, width-1024 Conformer encoder, an utterance-routed
eight-expert SMEAR projector, and a 16-layer Llama decoder with hidden size
2048 and vocabulary size 128,016. The deployed server keeps the log-mel
frontend in FP32 and runs the Conformer, SMEAR, and Llama in BF16. Keeping STFT
in FP32 avoided a fragile/unsupported BF16 frontend while preserving tensor
core execution for the expensive network layers.

The promoted maximum-throughput HTTP implementation added:

- TensorRT 10.10 BF16 execution for the FastConformer encoder;
- asynchronous adaptive frontend batching at 32/64 with 5/50 ms gather windows;
- length-aware, fairness-anchored request lookahead;
- persistent HTTP streams and a 120-second server keep-alive;
- raw 16 kHz signed-int16 PCM request bodies;
- variable-length masks passed through Conformer and SMEAR;
- fine-tuned prompt embeddings submitted independently to vLLM;
- per-request validity checks, latency records, batch counters, and GPU logs;
- explicit model/download revisions and engine SHA-256 provenance.

The separate quality-oriented four-beam path retains correct left packing and
`torch.compile(mode="reduce-overhead", dynamic=True)` with a persistent
Inductor cache.

## Critical batching correctness fix

The upstream sample is single-item inference. Naively right-padding a batch
placed shorter requests' final prompt position before the common tensor end,
while generation read the shared final position. This caused hundreds of empty
transcripts even though the requests completed successfully.

The fix packs each request so that its valid audio prefix and prompt end at the
same final decoder position. A c64 post-fix gate returned 64/64 valid requests
at 127.91 RTFx with observed microbatches `[4, 28, 32]`. The invalid pre-fix
run is retained in the logs and is excluded from every reported table.

The released checkpoint also registers the SMEAR experts under two module
paths. PyTorch reports 32 missing `encoder_projector.*` alias keys after loading,
but the corresponding `MoELayer_routing.experts.*` tensors are present and
loaded. `artifacts/checkpoint_alias_report.json` documents all 185 checkpoint
tensors and the alias mapping; this is not a missing-weight model.

## End-to-end findings

The four-beam compiled BF16 server changed quick WER from 29.412% to 29.630%, a
+0.218 percentage-point regression on 46 clips. It improved variable-duration
RTFx over the upstream precision profile at c1 through c128, peaking at 115.37
RTFx at c64. At c256 it reached 120.53 RTFx versus the clean upstream retry at
138.31, so compile is not universally faster.

On exact-1-second requests, performance was non-monotonic because request
completion timing determines how the server fills microbatches and each decode
has variable output length. Compiled BF16 won strongly at c1 and c64 but lost
to the upstream profile at several other points. This is why the complete
matrix is preserved instead of presenting a single peak number.

The one-beam eager profile reached 183.37 variable-duration RTFx at c256 and
120.79 exact-1-second RTFx at c32, but WER worsened by 3.050 percentage points.
It is a useful throughput ceiling, not a quality-preserving default.

The first upstream c256 variable run returned 255/256 valid requests due to a
client-side `ReadError`. It was rejected and repeated. Only the clean 256/256
retry at 138.31 RTFx appears in the final table.

## Canary-inspired vLLM continuous decoder

The `wuxuedaifu/Canary-Qwen-2.5b-vllm` design showed the useful architectural
split: retain the speech encoder and projector as a frontend, then submit each
projected prompt independently to vLLM instead of keeping an entire Hugging
Face `generate` batch in lockstep. The inspected fork is pinned at
`ae29af4502c9ac52eb88dbbbd6df721e4d8cac82`.

The released `model/llm` directory is not the decoder that Shrutam-2 actually
serves. `model/model.pt` contains 147 fine-tuned `llm.*` tensors. Using the base
directory with vLLM produced 272% WER. `export_vllm_checkpoint.py` materializes
the 146 expected BF16 decoder tensors, verifies tied embedding/lm-head equality,
and records SHA-256
`727c0cf4b68d9cfac69da68e5e796ea4170d697d48a57d937a70b1853e8f81fa`.
After that correction, prompt-token and prompt-embedding smoke paths produced
identical output, and HTTP accuracy returned to the one-beam range.

The fork's editable and normal precompiled wheel builds both omitted
`vllm._C` on this host. The reproducible installation therefore resolves the
fork dependency set, then overlays the official vLLM 0.12.0 binary wheel and
explicitly imports `vllm._C` as an install gate. The measured stack was PyTorch
2.9.1+cu129. Dynamic `torch.compile` of the frontend stalled for minutes in
symbolic-shape analysis on this stack, so the BF16 vLLM control uses the stable
eager BF16 FastConformer path. The subsequently promoted server replaces that
encoder path with TensorRT. This is a measured compatibility choice, not a
claim that eager is intrinsically faster than a working compiled graph.

Several server/client changes were jointly necessary:

- waveform host-to-device copies and prompt-embedding device-to-host copies
  are consolidated once per frontend microbatch;
- the frontend uses a fair length-aware lookahead to reduce Conformer padding;
- a two-stage gather waits 5 ms below batch 32 and may extend to 50 ms under
  load, selecting adaptive batches up to 64;
- the decoder uses vLLM CUDA graphs, chunked prefill, and independent request
  completion with capacity for 512 sequences;
- PCM conversion and one persistent socket per logical stream are prepared
  before timing, including all 512 sockets for the c512 measurement;
- Uvicorn keep-alive is 120 seconds, because its five-second default expired
  early sockets while high-concurrency connection pools were being formed.

The transport boundary mattered as much as GPU kernels. Before persistent
preconnect, variable c256 was roughly 185-203 RTFx and exact-1-second c256 was
roughly 49-120 RTFx. With the final persistent adaptive path they reached
550.85 and 119.98 RTFx; exact-1-second c512 reached 95.59 RTFx. The complete
final variable matrix was
`32.96,57.41,97.49,164.22,314.13,423.81,550.85` at concurrency
`1,2,8,32,64,128,256`. The exact-1-second matrix was
`12.56,21.12,51.25,98.01,111.85,112.91,119.98,95.59` through c512.
All 1,792 timed matrix responses were valid, failed count was zero, and every
record completed in one transport attempt.

The BF16 vLLM control's 46-clip quick gate was 31.808% WER / 18.233% CER.
Repeated one-beam BF16 runs on this small multilingual set moved between
31.808% and 32.462% WER,
so differences inside that band should not be presented as statistically
meaningful. Use a larger fixed manifest and repeated trials for a production
quality decision. The final request records show a generated-token maximum of
49 versus the configured cap of 96, ruling out measured truncation.

Rejected and intermediate scheduler, transfer, preconnect, and keep-alive runs
remain under `logs/tune_*`, `logs/server_vllm_*`, and the corresponding remote
result directories. They are useful evidence that c128 peaks from short tuning
runs are not a substitute for one cohesive full matrix.

## FP8 vLLM and integrated TensorRT outcome

Online vLLM FP8 weights with BF16 KV cache was functionally valid. Its complete
variable matrix was `33.42,51.95,102.84,184.03,301.52,394.25,449.16`; its
exact-one-second matrix was
`12.51,22.74,53.19,101.62,120.25,112.15,113.40,102.20`. Quick WER/CER was
32.244%/17.746%, only +0.436 WER percentage points versus the BF16 vLLM
control. It passed the requested 1.5 pp gate but lost enough high-concurrency
variable-audio throughput that it was not promoted.

Combining online FP8 weights and FP8 KV cache was not valid. vLLM loaded the
model but warned that attention query/probability scales were uncalibrated, and
the first real-audio warm-up returned an empty transcript even with runtime KV
scale calculation enabled. A successfully loaded server is therefore not an
accuracy gate. The health response and server log are retained, but no RTFx
matrix was run or reported for this configuration.

The integrated TensorRT BF16 encoder plus vLLM BF16 decoder was the best overall
configuration. The final adaptive-64 variable matrix was
`32.86,67.97,124.04,220.10,366.53,408.63,566.34`; its exact-one-second matrix
was `17.13,27.99,60.39,135.80,113.12,106.20,158.34,138.08`. All 1,792 timed
requests were valid with zero failures and zero retries. Relative to the BF16
vLLM control, the ratios were
`1.00,1.18,1.27,1.34,1.17,0.96,1.03` on variable audio and
`1.36,1.33,1.18,1.39,1.01,0.94,1.32,1.44` on one-second audio. The c128
regressions are real and retained. Quick WER/CER was 32.462%/18.233%, a +0.654
WER-point change versus the BF16 vLLM control and inside the accepted 1.5 pp
budget.

Adaptive-32 was also measured cleanly as a control. Its variable matrix was
`32.86,61.08,117.39,199.14,367.67,486.08,541.45`; its one-second matrix was
`15.25,28.49,61.46,126.17,125.41,99.39,148.86,125.00`; WER/CER was
31.373%/17.596%. Adaptive-64 won 9 of 15 throughput cells, including variable
c256 and one-second c256/c512, so it is the promoted general profile. Adaptive
32 remains a useful c128-specific option.

Two integration details were decisive:

- TensorRT 10.10 plans cannot be deserialized by the NeMo image's TensorRT
  10.8 runtime. The working launcher uses the TensorRT-LLM 0.20 image's
  TensorRT 10.10 libraries and mounts the tested vLLM 0.12 environment.
- The ONNX encoder exports `lengths` as INT64. Supplying an INT32 device pointer
  did not fail loudly, but produced widespread empty transcripts. The runner
  now casts every input to the engine-declared dtype before setting addresses.

The promoted engine profile is min/opt/max batch 1/32/64 at 101/501/1001 mel
frames. The 1,416,359,964-byte remote plan has SHA-256
`bc3b1ad64041b7038086f52778fede0b8533a2a233ba42d23817780560aa00ef`.
Preconnecting all 512 client streams before timing materially changed the c512
measurement: a targeted run reached 168.34 RTFx versus 66.57 when only 256
streams had been prepared. The accepted cohesive run reached 138.08 RTFx.
Back-to-back connection-pool teardown also caused run-to-run variation, so the
accepted matrices use a fresh server and cooldown rather than splicing peaks.

## `torch.compile` and AOTInductor

Steady-state encoder compile latency excludes first compilation. The first
dynamic server warm-up compiled batch 1 and batch 2 shapes in roughly 31.6 s
each; the cache is therefore mounted at
`artifacts/torchinductor_cache` for restarts.

On real variable-duration batches, encoder-only BF16 RTFx improved from eager
`86.43, 253.95, 2338.51, 4448.21, 4338.28, 4378.68, 4705.25` to compiled
`348.67, 892.08, 3813.91, 5627.03, 4925.59, 4808.29, 5368.95` at batch
`1,2,8,32,64,128,256`.

Fully dynamic-time AOTI export failed twice. The 8x convolutional subsampler
generated modular/divisibility shape guards that `torch.export` could not prove
for a general mel-frame dimension, even after the minimum was corrected from 8
to 9. The failure reports and tracebacks are saved in
`artifacts/aoti_build_report.json`, `logs/aoti_build.log`, and
`logs/aoti_build_retry.log`.

A useful AOTI artifact was still built for the exact-1-second case: time is
static at 101 mel frames and batch is dynamic through 256. Its encoder-only
RTFx was `240.27, 412.79, 1396.80, 5198.26, 4199.20, 4845.45, 5205.14`.
That is similar to cached `torch.compile` at most sizes and proves AOTI is viable
for a fixed chunk bucket. A production variable-audio service would need
several static-time packages or compiler/export fixes.

## TensorRT Conformer

The first BF16 engine build failed for two independent reasons:

1. `--stronglyTyped` contradicted the explicit BF16 builder request in this
   TensorRT version.
2. A maximum profile of batch 256 by 2,001 mel frames was far larger than the
   real server microbatch and consumed impractical build resources.

The first successful profile matched the initial serving ceiling: minimum
`1x80x101`, optimum `16x80x501`, maximum `32x80x1001`, 16 GiB workspace, and
optimization level 5. Both BF16 and FP16 engines built with TensorRT 10.10.
FP16 was fastest at batch 32 with 4.489 ms GPU compute time, equivalent to
7128.30 encoder-only RTFx. A second BF16 plan widened the profile to batch 64
and was integrated into the live HTTP path as documented above.

The standalone `trtexec` figures remain encoder-stage measurements. Only the
new adaptive-32/adaptive-64 server/client matrices include HTTP, preprocessing,
SMEAR, TensorRT encoder execution, vLLM decode, and response parsing.

## TensorRT-LLM audio-prefix integration

The first TensorRT-LLM smoke test used `tokenizer.vocab_size` when constructing
fake prompt-table token IDs. The engine vocabulary is 128,016, which is larger
than the tokenizer's exposed base size. Those IDs were interpreted as normal
text tokens and produced English hallucinations.

The corrected runner reads the engine vocabulary from its config, allocates
fake IDs at or above 128,016, and supplies the real BF16 Conformer/SMEAR output
as a per-request prompt embedding table. A real Assamese sample then produced
`টোকিও` for reference `টকিঅ'`, confirming the audio-prefix path was active.

The BF16 engine was built with max batch 32, input 256, sequence 512, beam width
4, prompt table 4096, paged input removal, and context FMHA. Its real-audio
beam-4 quick WER/CER was 31.808%/18.308%. This test is offline prompt-table
inference; it is not substituted for the measured FastAPI server/client matrix.

## ModelOpt FP8 decision

ModelOpt quantized the Llama decoder to FP8 with FP8 KV cache using 64
CNN/DailyMail calibration samples, batch 4, maximum sequence length 512. The
same 46 real audio prefixes and four-beam decoder settings produced WER/CER
30.719%/16.061%.

The signed FP8-minus-BF16 WER change is -1.089 percentage points, so there is no
directional quality regression. The absolute change is 1.089 points and fails
a strict `abs(delta) < 1 pp` equivalence test by 0.089 points. Because the
evaluation set is small, the FP8 engine remains an exploratory artifact rather
than the promoted deployment. A larger fixed manifest is required before
concluding that FP8 genuinely improves quality.

## Measurement and operational boundaries

- The baseline Git repository was used as the requested benchmark methodology
  reference and pinned at its measured revision. The upstream performance rows
  are Shrutam-2 itself with its original FP32 Conformer precision and four-beam
  decoding, served through the same HTTP harness; they are not old benchmark
  numbers copied from that repository.
- RTFx counts only valid responses. Each final matrix also requires zero failed
  requests, so silently dropping failures cannot inflate throughput.
- The quick transcript check is separate from throughput and retains every
  reference/hypothesis pair.
- Encoder-only and offline TensorRT-LLM evidence are labeled as such.
- GPU 0 was cleaned before measurement. The unrelated GPU 1 service was left
  untouched.
- Large engines remain on the remote project and are reproducible via scripts;
  the local package contains their sizes and SHA-256 hashes.

## Artifact trail

- `results/consolidated_results.json`: all final numbers and automated matrix
  completeness checks.
- `artifacts/provenance.json`: model/baseline revisions, hardware, engine sizes,
  and SHA-256 hashes.
- `results/upstream_fp32conformer_bf16llm_beam4/variable_c256_retry/`: accepted
  clean retry.
- `results/optimized_compile_bf16_beam4/`: recommended server/client evidence.
- `results/optimized_eager_bf16_beam1/`: throughput-only server/client evidence.
- `results/vllm_continuous_bf16_adaptive_persistent_beam1/`: BF16 vLLM control
  matrices, accuracy pairs, health, and server counters.
- `results/vllm_trt_bf16enc_bf16llm_adaptive64_persistent_beam1_final/`:
  promoted integrated TensorRT/vLLM full matrices and quality evidence.
- `results/vllm_trt_bf16enc_bf16llm_adaptive32_persistent512_beam1_final/`:
  clean adaptive-32 comparison.
- `results/vllm_fp8w_bf16kv_adaptive_persistent_beam1/`: complete valid FP8
  weight/BF16 KV run; `results/vllm_fp8w_fp8kv_adaptive_persistent_beam1/`
  contains the rejected FP8-KV health evidence.
- `logs/`: complete server, client, compiler, engine, accuracy, and GPU logs.
