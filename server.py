#!/usr/bin/env python3
"""Dynamically batched raw-PCM HTTP server for Shrutam-2."""

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

from runtime_model import DecodeConfig, ShrutamRuntime


@dataclass
class Pending:
    waveform: np.ndarray
    language: str
    prompt: str | None
    enqueued_at: float
    future: asyncio.Future


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class DynamicBatcher:
    def __init__(self, model: ShrutamRuntime, max_batch_size: int, max_delay_ms: float) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_delay_s = max_delay_ms / 1000.0
        self.queue: asyncio.Queue[Pending] = asyncio.Queue()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_audio_seconds = 0.0
        self.total_failures = 0
        self.batch_sizes: collections.Counter[int] = collections.Counter()
        self.batch_latencies_ms: list[float] = []
        self.queue_latencies_ms: list[float] = []
        self.last_batch: dict[str, Any] = {}
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._worker())

    async def submit(self, waveform: np.ndarray, language: str, prompt: str | None) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(Pending(waveform, language, prompt, time.perf_counter(), future))
        return await future

    async def _collect(self) -> list[Pending]:
        batch = [await self.queue.get()]
        deadline = time.perf_counter() + self.max_delay_s
        while len(batch) < self.max_batch_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    async def _worker(self) -> None:
        while True:
            items = await self._collect()
            started = time.perf_counter()
            queue_ms = [(started - item.enqueued_at) * 1000.0 for item in items]
            try:
                texts, timing = await asyncio.to_thread(
                    self.model.transcribe_batch,
                    [item.waveform for item in items],
                    [item.language for item in items],
                    [item.prompt for item in items],
                )
                batch_ms = (time.perf_counter() - started) * 1000.0
                self.total_requests += len(items)
                self.total_audio_seconds += sum(item.waveform.shape[0] / 16000.0 for item in items)
                self.batch_sizes[len(items)] += 1
                self.batch_latencies_ms.append(batch_ms)
                self.queue_latencies_ms.extend(queue_ms)
                self.last_batch = timing
                for item, text, wait_ms in zip(items, texts, queue_ms):
                    if not item.future.done():
                        item.future.set_result({
                            "text": text,
                            "queue_ms": wait_ms,
                            "batch_size": len(items),
                            "batch_ms": batch_ms,
                            "model_timing": timing,
                        })
            except Exception as exc:
                self.total_failures += len(items)
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in items:
                    self.queue.task_done()

    def stats(self) -> dict[str, Any]:
        return {
            "runtime": self.model.runtime_name,
            "precision": self.model.precision,
            "uptime_seconds": time.time() - self.started_at,
            "queue_depth": self.queue.qsize(),
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_audio_seconds": self.total_audio_seconds,
            "batch_histogram": dict(sorted(self.batch_sizes.items())),
            "batch_latency_p50_ms": percentile(self.batch_latencies_ms, 50),
            "batch_latency_p95_ms": percentile(self.batch_latencies_ms, 95),
            "queue_latency_p50_ms": percentile(self.queue_latencies_ms, 50),
            "queue_latency_p95_ms": percentile(self.queue_latencies_ms, 95),
            "last_batch": self.last_batch,
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Shrutam-2 optimized ASR", version="1.0")

    @app.on_event("startup")
    async def startup() -> None:
        app.state.model = ShrutamRuntime(
            args.model_dir,
            runtime=args.runtime,
            precision=args.precision,
            decode=DecodeConfig(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams),
        )
        app.state.warmup = await asyncio.to_thread(app.state.model.warmup, args.warmup_batches)
        app.state.batcher = DynamicBatcher(app.state.model, args.max_batch_size, args.max_delay_ms)
        app.state.batcher.start()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        model = app.state.model
        return {
            "ready": True,
            "runtime": model.runtime_name,
            "precision": model.precision,
            "encoder_projector_dtype": str(model.dtype),
            "llm_dtype": str(model.llm_dtype),
            "gpu": torch.cuda.get_device_name(0),
            "checkpoint_missing": model.checkpoint_missing,
            "checkpoint_unexpected": model.checkpoint_unexpected,
            "warmup": app.state.warmup,
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return app.state.batcher.stats()

    @app.get("/preconnect")
    async def preconnect() -> dict[str, bool]:
        """Allow benchmark clients to establish persistent sockets untimed."""
        return {"ok": True}

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
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--runtime", choices=("eager", "compile"), default="eager")
    parser.add_argument("--precision", choices=("upstream", "bf16"), default="bf16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-delay-ms", type=float, default=8.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, nargs="+", default=[1, 2, 8])
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    uvicorn.run(build_app(parsed), host=parsed.host, port=parsed.port, log_level=os.getenv("LOG_LEVEL", "info"))
