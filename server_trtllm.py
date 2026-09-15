#!/usr/bin/env python3
"""Server/client TensorRT-LLM deployment for Shrutam-2."""

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

from runtime_trtllm import ShrutamTRTLLMRuntime


@dataclass
class Pending:
    waveform: np.ndarray
    language: str
    prompt: str | None
    enqueued_at: float
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, model: ShrutamTRTLLMRuntime, max_batch_size: int, max_delay_ms: float) -> None:
        self.model = model
        self.max_batch_size = min(max_batch_size, model.max_batch_size)
        self.max_delay_s = max_delay_ms / 1000.0
        self.queue: asyncio.Queue[Pending] = asyncio.Queue()
        self.total_requests = 0
        self.total_failures = 0
        self.total_audio_seconds = 0.0
        self.batch_histogram: collections.Counter[int] = collections.Counter()
        self.last_batch: dict[str, object] = {}

    def start(self) -> None:
        asyncio.create_task(self._worker())

    async def submit(self, waveform: np.ndarray, language: str, prompt: str | None) -> dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
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
                self.batch_histogram[len(items)] += 1
                self.last_batch = timing
                for item, text in zip(items, texts):
                    item.future.set_result({
                        "text": text,
                        "queue_ms": (started - item.enqueued_at) * 1000.0,
                        "batch_size": len(items),
                        "batch_ms": batch_ms,
                        "decoder_timing": timing,
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
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_audio_seconds": self.total_audio_seconds,
            "queue_depth": self.queue.qsize(),
            "batch_histogram": dict(sorted(self.batch_histogram.items())),
            "last_batch": self.last_batch,
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Shrutam-2 TensorRT-LLM server", version="1.0")

    @app.on_event("startup")
    async def startup() -> None:
        app.state.model = ShrutamTRTLLMRuntime(
            args.model_dir,
            args.engine_dir,
            encoder_runtime=args.encoder_runtime,
            encoder_precision=args.encoder_precision,
            trt_engine=args.trt_engine,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            kv_cache_fraction=args.kv_cache_fraction,
            cuda_graph_mode=args.cuda_graph_mode,
        )
        app.state.warmup = await asyncio.to_thread(app.state.model.warmup, args.warmup_batches)
        app.state.max_audio_samples = int(args.max_audio_seconds * 16000)
        app.state.batcher = DynamicBatcher(app.state.model, args.max_batch_size, args.max_delay_ms)
        app.state.batcher.start()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        model = app.state.model
        return {
            "ready": True,
            "gpu": torch.cuda.get_device_name(0),
            "beam": model.num_beams,
            "encoder_runtime": model.encoder_runtime,
            "encoder_precision": model.encoder_precision,
            "engine_dir": str(model.engine_dir),
            "decoder_dtype": model.decoder_dtype,
            "decoder_quant_algo": model.quant_algo,
            "kv_cache_quant_algo": model.kv_cache_quant_algo,
            "max_batch_size": model.max_batch_size,
            "max_audio_seconds": args.max_audio_seconds,
            "warmup": app.state.warmup,
        }

    @app.get("/preconnect")
    async def preconnect() -> dict[str, bool]:
        return {"ok": True}

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
        if x_sample_rate != 16000:
            raise HTTPException(400, "only 16 kHz PCM is accepted")
        body = await request.body()
        if len(body) < 320 or len(body) % 2:
            raise HTTPException(400, "PCM must be non-empty, even-length signed 16-bit audio")
        waveform = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
        if waveform.shape[0] > app.state.max_audio_samples:
            raise HTTPException(
                400,
                f"audio exceeds {args.max_audio_seconds:g}s TensorRT-LLM input profile",
            )
        if not np.isfinite(waveform).all() or float(np.max(np.abs(waveform))) < 1e-6:
            raise HTTPException(400, "audio is silent or invalid")
        started = time.perf_counter()
        try:
            result = await app.state.batcher.submit(waveform, x_language, x_prompt)
        except Exception as exc:
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
        result.update({
            "request_id": str(uuid.uuid4()),
            "duration_seconds": waveform.shape[0] / 16000.0,
            "server_total_ms": (time.perf_counter() - started) * 1000.0,
        })
        return result

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--encoder-runtime", choices=("eager", "compile", "trt"), default="eager")
    parser.add_argument("--encoder-precision", choices=("upstream", "bf16"), default="bf16")
    parser.add_argument("--trt-engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-audio-seconds", type=float, default=19.0)
    parser.add_argument("--max-delay-ms", type=float, default=12.0)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--kv-cache-fraction", type=float, default=0.55)
    parser.add_argument("--cuda-graph-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-batches", type=int, nargs="+", default=[1, 2, 8, 32])
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    uvicorn.run(build_app(parsed), host=parsed.host, port=parsed.port, log_level=os.getenv("LOG_LEVEL", "warning"))
