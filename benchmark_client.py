#!/usr/bin/env python3
"""Integrity-checking asynchronous HTTP benchmark for Shrutam-2."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf


@dataclass
class Sample:
    path: Path
    waveform: np.ndarray
    pcm: bytes
    duration: float
    text: str
    language: str


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else math.nan


def load_manifest(path: Path, fixed_seconds: float | None) -> list[Sample]:
    rows: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            audio_path = Path(item.get("audio_filepath") or item.get("audio") or item.get("path"))
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            elif not audio_path.exists() and str(audio_path).startswith("/workspace/"):
                # Manifests are shared by the Docker and host-venv runners.
                # Rebase the container mount without rewriting benchmark data.
                audio_path = Path(__file__).resolve().parent / audio_path.relative_to("/workspace")
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if sample_rate != 16000:
                raise ValueError(f"{audio_path}: expected 16 kHz, got {sample_rate}")
            if fixed_seconds is not None:
                target = int(round(fixed_seconds * sample_rate))
                if waveform.shape[0] >= target:
                    waveform = waveform[:target]
                else:
                    waveform = np.pad(waveform, (0, target - waveform.shape[0]))
            if waveform.size < 160 or float(np.max(np.abs(waveform))) < 1e-6:
                continue
            rows.append(Sample(
                audio_path,
                np.asarray(waveform, dtype=np.float32),
                pcm_bytes(waveform),
                waveform.shape[0] / sample_rate,
                str(item.get("text") or item.get("transcript") or ""),
                str(item.get("language") or item.get("lang") or "hi"),
            ))
    if not rows:
        raise ValueError(f"no usable audio in {path}")
    return rows


def pcm_bytes(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 0.9999695)
    return (clipped * 32768.0).astype("<i2").tobytes()


async def one_request(
    client: httpx.AsyncClient,
    endpoint: str,
    sample: Sample,
    index: int,
    transport_retries: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    attempts = 0
    while True:
        try:
            attempts += 1
            response = await client.post(
                endpoint,
                content=sample.pcm,
                headers={
                    "content-type": "application/octet-stream",
                    "x-sample-rate": "16000",
                    "x-language": sample.language,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            payload = response.json()
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("empty transcript")
            return {
                "index": index,
                "ok": True,
                "attempts": attempts,
                "path": str(sample.path),
                "duration_seconds": sample.duration,
                "reference": sample.text,
                "language": sample.language,
                "text": text,
                "latency_ms": elapsed_ms,
                "server_batch_size": int(payload.get("batch_size", 0)),
                "server_total_ms": float(payload.get("server_total_ms", 0.0)),
                "generated_tokens": int(payload.get("decoder_timing", {}).get("generated_tokens", 0)),
            }
        except httpx.TransportError as exc:
            if attempts <= transport_retries:
                continue
            error = exc
        except Exception as exc:
            error = exc
        return {
            "index": index,
            "ok": False,
            "attempts": attempts,
            "path": str(sample.path),
            "duration_seconds": sample.duration,
            "reference": sample.text,
            "language": sample.language,
            "error": f"{type(error).__name__}: {error}",
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


async def run_level(
    client: httpx.AsyncClient,
    endpoint: str,
    samples: list[Sample],
    concurrency: int,
    rounds: int,
    seed: int,
    min_requests: int,
    sequential: bool,
    transport_retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed + concurrency)
    count = max(concurrency * rounds, min_requests)
    selected = (
        [samples[index % len(samples)] for index in range(count)]
        if sequential
        else [samples[rng.randrange(len(samples))] for _ in range(count)]
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(index: int, sample: Sample) -> dict[str, Any]:
        async with semaphore:
            return await one_request(client, endpoint, sample, index, transport_retries)

    started = time.perf_counter()
    results = await asyncio.gather(*(guarded(i, sample) for i, sample in enumerate(selected)))
    wall = time.perf_counter() - started
    valid = [row for row in results if row["ok"]]
    latencies = [float(row["latency_ms"]) for row in valid]
    audio_seconds = sum(float(row["duration_seconds"]) for row in valid)
    summary = {
        "concurrency": concurrency,
        "requests": count,
        "valid": len(valid),
        "failed": count - len(valid),
        "wall_seconds": wall,
        "audio_seconds": audio_seconds,
        "rtfx": audio_seconds / wall if wall > 0 else math.nan,
        "requests_per_second": len(valid) / wall if wall > 0 else math.nan,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else math.nan,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "observed_batch_sizes": sorted(set(int(row["server_batch_size"]) for row in valid)),
    }
    return summary, results


async def main_async(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(Path(args.manifest), args.fixed_seconds)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=max(args.concurrency) + 16, max_keepalive_connections=max(args.concurrency))
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = (await client.get(args.url.rstrip("/") + "/health")).json()
        (output / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
        # One untimed request establishes a persistent connection and validates the response contract.
        warm = await one_request(
            client, args.url.rstrip("/") + "/transcribe", samples[0], -1, args.transport_retries
        )
        if not warm["ok"]:
            raise RuntimeError(f"warmup failed: {warm}")
        for concurrency in args.concurrency:
            # Production streams keep their transport open. Establish one
            # HTTP/1.1 socket per stream outside the timed interval.
            preconnect_streams = min(concurrency, args.max_preconnect_streams)
            preconnect_started = time.perf_counter()
            preconnect = await asyncio.gather(*(
                client.get(args.url.rstrip("/") + "/preconnect")
                for _ in range(preconnect_streams)
            ))
            for response in preconnect:
                response.raise_for_status()
            preconnect_ms = (time.perf_counter() - preconnect_started) * 1000.0
            summary, rows = await run_level(
                client,
                args.url.rstrip("/") + "/transcribe",
                samples,
                concurrency,
                args.rounds,
                args.seed,
                args.min_requests,
                args.sequential,
                args.transport_retries,
            )
            summary["preconnected_streams"] = preconnect_streams
            summary["preconnect_ms_untimed"] = preconnect_ms
            summaries.append(summary)
            for row in rows:
                row["concurrency"] = concurrency
            all_rows.extend(rows)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        stats = (await client.get(args.url.rstrip("/") + "/stats")).json()
        (output / "server_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "concurrency", "requests", "valid", "failed", "wall_seconds", "audio_seconds", "rtfx",
            "requests_per_second", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "observed_batch_sizes", "preconnected_streams", "preconnect_ms_untimed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(int(row["failed"]) for row in summaries):
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--min-requests", type=int, default=0)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--fixed-seconds", type=float)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transport-retries", type=int, default=1)
    parser.add_argument("--max-preconnect-streams", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
