#!/usr/bin/env python3
"""Prepare a deterministic public multilingual FLEURS accuracy/throughput set."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly


CONFIGS = {
    "as": "as_in",
    "bn": "bn_in",
    "gu": "gu_in",
    "hi": "hi_in",
    "kn": "kn_in",
    "ml": "ml_in",
    "mr": "mr_in",
    "or": "or_in",
    "pa": "pa_in",
    "ta": "ta_in",
    "te": "te_in",
    "ur": "ur_pk",
}


def decode_audio(value: object) -> tuple[np.ndarray, int]:
    if isinstance(value, dict):
        if value.get("array") is not None:
            wave = np.asarray(value["array"], dtype=np.float32)
            rate = int(value["sampling_rate"])
        elif value.get("bytes"):
            wave, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
        elif value.get("path"):
            wave, rate = sf.read(str(value["path"]), dtype="float32")
        else:
            raise ValueError("audio mapping has no array, bytes, or path")
    elif hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        wave = np.asarray(samples.data, dtype=np.float32)
        rate = int(samples.sample_rate)
    else:
        raise TypeError(f"unsupported audio object: {type(value)!r}")
    if wave.ndim == 2:
        wave = wave.mean(axis=0 if wave.shape[0] <= 2 else 1)
    if rate != 16000:
        wave = resample_poly(wave, 16000, rate).astype(np.float32, copy=False)
        rate = 16000
    return np.ascontiguousarray(wave, dtype=np.float32), rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/workspace/data/fleurs_quick")
    parser.add_argument("--samples-per-language", type=int, default=4)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    output = Path(args.output_dir)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    language_counts: dict[str, int] = {}
    for language, config in CONFIGS.items():
        dataset = load_dataset("google/fleurs", config, split=args.split, streaming=True)
        try:
            dataset = dataset.cast_column("audio", Audio(decode=True, sampling_rate=16000))
        except Exception:
            pass
        accepted = 0
        for source_index, item in enumerate(dataset):
            text = str(item.get("transcription") or item.get("raw_transcription") or "").strip()
            if not text:
                continue
            try:
                wave, rate = decode_audio(item["audio"])
            except Exception:
                continue
            duration = wave.shape[0] / rate
            if not 0.5 <= duration <= 20.0 or float(np.max(np.abs(wave))) < 1e-6:
                continue
            path = audio_dir / f"{language}_{accepted:02d}.wav"
            sf.write(path, wave, rate, subtype="PCM_16")
            rows.append({
                "audio_filepath": f"/workspace/data/fleurs_quick/audio/{path.name}",
                "duration": duration,
                "text": text,
                "language": language,
                "dataset": "google/fleurs",
                "config": config,
                "split": args.split,
                "source_index": source_index,
                "source_id": item.get("id"),
            })
            accepted += 1
            if accepted == args.samples_per_language:
                break
        if accepted != args.samples_per_language:
            raise RuntimeError(f"{config}: selected only {accepted} usable samples")
        language_counts[language] = accepted

    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "dataset": "google/fleurs",
        "split": args.split,
        "selection": "first usable non-silent rows in streaming source order",
        "samples": len(rows),
        "audio_seconds": sum(float(row["duration"]) for row in rows),
        "languages": language_counts,
        "manifest": str(manifest),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
