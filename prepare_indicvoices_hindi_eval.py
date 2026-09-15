#!/usr/bin/env python3
"""Materialize the official IndicVoices Hindi validation split for held-out ASR evaluation."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly


def reference(row: dict[str, object]) -> str:
    for key in ("normalized", "text", "verbatim", "unsanitized_normalized"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def decode_audio(value: object) -> tuple[np.ndarray, int]:
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw:
            wave, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        elif path:
            wave, rate = sf.read(str(path), dtype="float32", always_2d=False)
        else:
            raise ValueError("audio mapping has neither bytes nor path")
    elif isinstance(value, str):
        wave, rate = sf.read(value, dtype="float32", always_2d=False)
    else:
        raise TypeError(f"unsupported audio value: {type(value)!r}")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        wave = resample_poly(wave, 16000, rate).astype(np.float32, copy=False)
        rate = 16000
    return np.ascontiguousarray(wave, dtype=np.float32), rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/workspace/data/indicvoices_hindi_valid")
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=19.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 materializes every usable row")
    args = parser.parse_args()

    output = Path(args.output_dir)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_parquet = hf_hub_download(
        repo_id="ai4bharat/IndicVoices",
        filename="hindi/valid-00000-of-00001.parquet",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN") or False,
    )
    dataset = load_dataset(
        "parquet", data_files={"valid": source_parquet}, split="valid"
    ).cast_column("audio_filepath", Audio(decode=False))

    rows: list[dict[str, object]] = []
    excluded = {
        "missing_reference": 0,
        "unintelligible": 0,
        "duration": 0,
        "decode": 0,
        "silent": 0,
    }
    for source_index, item in enumerate(dataset):
        text = reference(item)
        if not text:
            excluded["missing_reference"] += 1
            continue
        if "unintelligible" in text.casefold():
            excluded["unintelligible"] += 1
            continue
        duration = float(item.get("duration") or 0.0)
        if not args.min_seconds <= duration <= args.max_seconds:
            excluded["duration"] += 1
            continue
        try:
            wave, rate = decode_audio(item["audio_filepath"])
        except Exception:
            excluded["decode"] += 1
            continue
        measured_duration = wave.shape[0] / rate
        if not args.min_seconds <= measured_duration <= args.max_seconds:
            excluded["duration"] += 1
            continue
        if wave.size < 160 or float(np.max(np.abs(wave))) < 1e-6:
            excluded["silent"] += 1
            continue
        destination = audio_dir / f"hi_{source_index:05d}.wav"
        sf.write(destination, wave, rate, subtype="PCM_16")
        rows.append({
            "audio_filepath": f"/workspace/data/indicvoices_hindi_valid/audio/{destination.name}",
            "duration": measured_duration,
            "text": text,
            "language": "hi",
            "dataset": "ai4bharat/IndicVoices",
            "config": "hindi",
            "split": "valid",
            "source_index": source_index,
            "speaker_id": item.get("speaker_id"),
            "scenario": item.get("scenario"),
        })
        if args.max_samples and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError("no usable Hindi validation audio was materialized")
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "dataset": "ai4bharat/IndicVoices",
        "config": "hindi",
        "split": "valid",
        "source_examples": len(dataset),
        "selection": "all usable rows in source order",
        "filters": {
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "text_exclusion": "case-insensitive substring 'unintelligible' (including <unintelligible>)",
            "duration_reason": "fits the 256-token TensorRT-LLM input contract after the fixed 17-token Hindi prompt",
        },
        "samples": len(rows),
        "audio_seconds": sum(float(row["duration"]) for row in rows),
        "excluded": excluded,
        "manifest": str(manifest),
        "dataset_fingerprint": dataset._fingerprint,
        "source_file": "hindi/valid-00000-of-00001.parquet",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
