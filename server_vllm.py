#!/usr/bin/env python3
"""Server/client deployment for Shrutam-2 with vLLM continuous decoding."""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request

from runtime_vllm import ShrutamVLLMRuntime


@dataclass
class Pending:
    waveform: np.ndarray
    language: str
    prompt: str | None
    enqueued_at: float
    future: asyncio.Future


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


class SplitBatcher:
    """Microbatch the frontend; submit decoded rows independently to vLLM."""

    def __init__(
        self,
        model: ShrutamVLLMRuntime,
        max_frontend_batch: int,
        base_frontend_batch: int,
        large_batch_queue_threshold: int,
        length_bucket_lookahead: int,
        base_frontend_delay_ms: float,
        max_frontend_delay_ms: float,
        max_inflight_decode: int,
    ) -> None:
        self.model = model
        self.max_batch = max_frontend_batch
        self.base_batch = min(base_frontend_batch, max_frontend_batch)
        self.large_batch_queue_threshold = large_batch_queue_threshold
        self.length_bucket_lookahead = max(length_bucket_lookahead, max_frontend_batch)
        self.base_delay_s = min(base_frontend_delay_ms, max_frontend_delay_ms) / 1000.0
        self.max_delay_s = max_frontend_delay_ms / 1000.0
        self.queue: asyncio.Queue[Pending] = asyncio.Queue()
        self.backlog: list[Pending] = []
        self.decode_slots = asyncio.Semaphore(max_inflight_decode)
        self.worker: asyncio.Task | None = None
        self.decode_tasks: set[asyncio.Task] = set()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_audio_seconds = 0.0
        self.total_failures = 0
        self.inflight_decode = 0
        self.frontend_batches: collections.Counter[int] = collections.Counter()
        self.frontend_ms: list[float] = []
        self.decode_ms: list[float] = []
        self.queue_ms: list[float] = []
        self.frontend_actual_samples = 0
        self.frontend_padded_samples = 0

    def start(self) -> None:
        self.worker = asyncio.create_task(self._worker())

    async def submit(self, waveform: np.ndarray, language: str, prompt: str | None) -> dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(Pending(waveform, language, prompt, time.perf_counter(), future))
        return await future

    async def _collect(self) -> list[Pending]:
        # Pull a bounded lookahead window and form a length-homogeneous batch.
        # FastConformer pads every row to the longest waveform, so FIFO batches
        # can waste most of the encoder work when short and long clips mix.
        candidates = self.backlog
        self.backlog = []
        if not candidates:
            candidates = [await self.queue.get()]

        collect_started = time.perf_counter()
        base_deadline = collect_started + self.base_delay_s
        large_deadline = collect_started + self.max_delay_s
        while len(candidates) < self.length_bucket_lookahead:
            try:
                candidates.append(self.queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            deadline = (
                large_deadline if len(candidates) >= self.base_batch else base_deadline
            )
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                candidates.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break

        target_batch = (
            self.max_batch
            if len(candidates) >= self.large_batch_queue_threshold
            else self.base_batch
        )
        target_batch = min(target_batch, len(candidates))

        # Anchor each bucket on the oldest request for bounded, fair reordering,
        # then choose the closest waveform lengths to minimize padding.
        oldest = min(candidates, key=lambda item: item.enqueued_at)
        ranked = sorted(
            candidates,
            key=lambda item: (
                abs(item.waveform.shape[0] - oldest.waveform.shape[0]),
                item.enqueued_at,
            ),
        )
        batch = ranked[:target_batch]
        selected = {id(item) for item in batch}
        self.backlog = [item for item in candidates if id(item) not in selected]
        return batch

    async def _decode_one(
        self,
        item: Pending,
        embeddings: torch.Tensor,
        frontend: dict[str, Any],
        batch_size: int,
        queue_ms: float,
    ) -> None:
        try:
            async with self.decode_slots:
                self.inflight_decode += 1
                try:
                    text, decoder = await self.model.decode(embeddings)
                finally:
                    self.inflight_decode -= 1
            if not text:
                raise RuntimeError("empty transcript")
            self.total_requests += 1
            self.total_audio_seconds += item.waveform.shape[0] / 16000.0
            self.decode_ms.append(float(decoder["decode_ms"]))
            if not item.future.done():
                item.future.set_result({
                    "text": text,
                    "queue_ms": queue_ms,
                    "batch_size": batch_size,
                    "frontend_timing": frontend,
                    "decoder_timing": decoder,
                })
        except Exception as exc:
            self.total_failures += 1
            if not item.future.done():
                item.future.set_exception(exc)
        finally:
            self.queue.task_done()

    async def _worker(self) -> None:
        while True:
            items = await self._collect()
            started = time.perf_counter()
            waits = [(started - item.enqueued_at) * 1000.0 for item in items]
            try:
                embeddings, frontend = await asyncio.to_thread(
                    self.model.frontend.prepare_batch,
                    [item.waveform for item in items],
                    [item.language for item in items],
                    [item.prompt for item in items],
                )
                self.frontend_batches[len(items)] += 1
                self.frontend_ms.append(float(frontend["frontend_ms"]))
                self.queue_ms.extend(waits)
                self.frontend_actual_samples += sum(item.waveform.shape[0] for item in items)
                self.frontend_padded_samples += (
                    max(item.waveform.shape[0] for item in items) * len(items)
                )
                for item, prompt_embeddings, wait_ms in zip(items, embeddings, waits):
                    task = asyncio.create_task(
                        self._decode_one(item, prompt_embeddings, frontend, len(items), wait_ms)
                    )
                    self.decode_tasks.add(task)
                    task.add_done_callback(self.decode_tasks.discard)
            except Exception as exc:
                self.total_failures += len(items)
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
                    self.queue.task_done()

    def stats(self) -> dict[str, Any]:
        return {
            **self.model.config,
            "uptime_seconds": time.time() - self.started_at,
            "frontend_queue_depth": self.queue.qsize() + len(self.backlog),
            "max_frontend_batch": self.max_batch,
            "base_frontend_batch": self.base_batch,
            "large_batch_queue_threshold": self.large_batch_queue_threshold,
            "length_bucket_lookahead": self.length_bucket_lookahead,
            "base_frontend_delay_ms": self.base_delay_s * 1000.0,
            "max_frontend_delay_ms": self.max_delay_s * 1000.0,
            "inflight_decode": self.inflight_decode,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_audio_seconds": self.total_audio_seconds,
            "frontend_batch_histogram": dict(sorted(self.frontend_batches.items())),
            "frontend_ms_p50": percentile(self.frontend_ms, 50),
            "frontend_ms_p95": percentile(self.frontend_ms, 95),
            "decode_ms_p50": percentile(self.decode_ms, 50),
            "decode_ms_p95": percentile(self.decode_ms, 95),
            "queue_ms_p50": percentile(self.queue_ms, 50),
            "queue_ms_p95": percentile(self.queue_ms, 95),
            "frontend_padding_efficiency": (
                self.frontend_actual_samples / self.frontend_padded_samples
                if self.frontend_padded_samples
                else 0.0
            ),
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Shrutam-2 vLLM continuous-batching ASR", version="2.0")

    @app.on_event("startup")
    async def startup() -> None:
        app.state.model = ShrutamVLLMRuntime(
            args.model_dir,
            decoder_dir=args.decoder_dir,
            runtime=args.runtime,
            precision=args.precision,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trt_engine=args.trt_engine,
            llm_quantization=(
                None if args.llm_quantization == "none" else args.llm_quantization
            ),
            kv_cache_dtype=args.kv_cache_dtype,
            calculate_kv_scales=args.calculate_kv_scales,
        )
        app.state.warmup = await app.state.model.warmup(args.warmup_batches)
        app.state.batcher = SplitBatcher(
            app.state.model,
            args.max_frontend_batch,
            args.base_frontend_batch,
            args.large_batch_queue_threshold,
            args.length_bucket_lookahead,
            args.base_frontend_delay_ms,
            args.max_frontend_delay_ms,
            args.max_inflight_decode,
        )
        app.state.batcher.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        app.state.model.shutdown()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ready": True,
            "architecture": (
                f"{app.state.model.config['runtime']} FastConformer frontend "
                "+ vLLM continuous decoder"
            ),
            "gpu": torch.cuda.get_device_name(0),
            "config": app.state.model.config,
            "checkpoint_missing": app.state.model.frontend.base.checkpoint_missing,
            "checkpoint_unexpected": app.state.model.frontend.base.checkpoint_unexpected,
            "warmup": app.state.warmup,
        }

    @app.get("/preconnect")
    async def preconnect() -> dict[str, bool]:
        # Keep the socket occupied briefly so an HTTP/1.1 client can establish
        # one persistent connection per logical stream before timed traffic.
        await asyncio.sleep(args.preconnect_hold_ms / 1000.0)
        return {"ready": True}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return app.state.batcher.stats()

    @app.post("/transcribe")
    async def transcribe(
        request: Request,
        x_sample_rate: int = Header(default=16000),
        x_language: str = Header(default="hi"),
        x_prompt: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        if x_sample_rate != 16000:
            raise HTTPException(400, "only 16 kHz PCM is accepted")
        body = await request.body()
        if len(body) < 320 or len(body) % 2:
            raise HTTPException(400, "PCM must be non-empty, even-length signed 16-bit audio")
        pcm = np.frombuffer(body, dtype="<i2")
        waveform = pcm.astype(np.float32) / 32768.0
        if not np.isfinite(waveform).all() or float(np.max(np.abs(waveform))) < 1e-6:
            raise HTTPException(400, "audio is silent or invalid")
        started = time.perf_counter()
        try:
            result = await app.state.batcher.submit(waveform, x_language, x_prompt)
        except Exception as exc:
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
        result.update({
            "request_id": request_id,
            "duration_seconds": waveform.shape[0] / 16000.0,
            "server_total_ms": (time.perf_counter() - started) * 1000.0,
        })
        return result

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/nvidia/Optimize-shrutam2/model")
    parser.add_argument(
        "--decoder-dir", default="/home/nvidia/Optimize-shrutam2/artifacts/vllm_llm_bf16"
    )
    parser.add_argument("--runtime", choices=("eager", "compile", "trt"), default="compile")
    parser.add_argument("--trt-engine")
    parser.add_argument("--precision", choices=("upstream", "bf16"), default="bf16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-frontend-batch", type=int, default=32)
    parser.add_argument("--base-frontend-batch", type=int, default=32)
    parser.add_argument("--large-batch-queue-threshold", type=int, default=128)
    parser.add_argument("--length-bucket-lookahead", type=int, default=256)
    parser.add_argument("--base-frontend-delay-ms", type=float, default=5.0)
    parser.add_argument("--max-frontend-delay-ms", type=float, default=50.0)
    parser.add_argument("--preconnect-hold-ms", type=float, default=50.0)
    parser.add_argument("--max-inflight-decode", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--llm-quantization", choices=("none", "fp8"), default="none")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"),
        default="auto",
    )
    parser.add_argument("--calculate-kv-scales", action="store_true")
    parser.add_argument("--warmup-batches", type=int, nargs="+", default=[1, 2, 8])
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    uvicorn.run(
        build_app(parsed),
        host=parsed.host,
        port=parsed.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
        timeout_keep_alive=120,
        backlog=2048,
    )
