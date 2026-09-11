#!/usr/bin/env python3
"""Materialize a small multilingual workload from cached IndicVoices Arrow files."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset
from scipy.signal import resample_poly


CODE_ALIASES = {
    "as": "as", "assamese": "as", "bn": "bn", "bengali": "bn",
    "gu": "gu", "gujarati": "gu", "hi": "hi", "hindi": "hi",
    "kn": "kn", "kannada": "kn", "ml": "ml", "malayalam": "ml",
    "mr": "mr", "marathi": "mr", "or": "or", "odia": "or", "oriya": "or",
    "pa": "pa", "punjabi": "pa", "ta": "ta", "tamil": "ta",
    "te": "te", "telugu": "te", "ur": "ur", "urdu": "ur",
}
TARGET_SECONDS = (1.2, 2.5, 5.0, 9.0)


def language_code(value: object) -> str | None:
    text = str(value or "").strip().casefold()
    return CODE_ALIASES.get(text)


def reference(row: dict[str, object]) -> str:
    for key in ("text", "normalized", "unsanitized_normalized", "verbatim"):
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
        elif path and Path(str(path)).exists():
            wave, rate = sf.read(str(path), dtype="float32", always_2d=False)
        else:
            raise FileNotFoundError(f"audio payload has neither bytes nor existing path: {path}")
    elif isinstance(value, str) and Path(value).exists():
        wave, rate = sf.read(value, dtype="float32", always_2d=False)
    else:
        raise TypeError(f"unsupported audio payload: {type(value)!r}")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        wave = resample_poly(wave, 16000, rate).astype(np.float32, copy=False)
        rate = 16000
    return np.asarray(wave, dtype=np.float32), rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/host_cache/huggingface/datasets/parquet")
    parser.add_argument("--output-dir", default="/workspace/data/indicvoices_quick")
    parser.add_argument("--samples-per-language", type=int, default=4)
    args = parser.parse_args()

    output = Path(args.output_dir)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[tuple[float, Path, int]]] = {code: [] for code in sorted(set(CODE_ALIASES.values()))}
    inspected = []
    for arrow in sorted(Path(args.cache_root).rglob("*.arrow")):
        try:
            dataset = Dataset.from_file(str(arrow))
            if not {"audio_filepath", "lang", "duration"}.issubset(dataset.column_names):
                continue
            langs = dataset["lang"]
            durations = dataset["duration"]
            counts: dict[str, int] = {}
            for index, (lang, duration) in enumerate(zip(langs, durations)):
                code = language_code(lang)
                if code is None:
                    continue
                seconds = float(duration or 0.0)
                if 1.0 <= seconds <= 20.0:
                    candidates[code].append((seconds, arrow, index))
                    counts[code] = counts.get(code, 0) + 1
            inspected.append({"arrow": str(arrow), "rows": len(dataset), "supported": counts})
        except Exception as exc:
            inspected.append({"arrow": str(arrow), "error": f"{type(exc).__name__}: {exc}"})

    rows = []
    selected_report: dict[str, object] = {}
    for code, items in candidates.items():
        # Deduplicate caches by duration/index proximity and cover short through long utterances.
        chosen = []
        remaining = list(items)
        targets = TARGET_SECONDS[: args.samples_per_language]
        for target in targets:
            if not remaining:
                break
            best = min(remaining, key=lambda item: abs(item[0] - target))
            chosen.append(best)
            remaining.remove(best)
        language_rows = []
        for ordinal, (duration, arrow, index) in enumerate(chosen):
            dataset = Dataset.from_file(str(arrow)).cast_column("audio_filepath", Audio(decode=False))
            row = dataset[index]
            text = reference(row)
            if not text or "unintelligible" in text.casefold():
                continue
            try:
                wave, rate = decode_audio(row["audio_filepath"])
            except Exception:
                continue
            destination = audio_dir / f"{code}_{ordinal:02d}.wav"
            sf.write(destination, wave, rate, subtype="PCM_16")
            record = {
                "audio_filepath": str(destination),
                "duration": wave.shape[0] / rate,
                "text": text,
                "language": code,
                "source_arrow": str(arrow),
                "source_index": index,
            }
            rows.append(record)
            language_rows.append(record)
        selected_report[code] = {
            "candidates": len(items),
            "selected": len(language_rows),
            "durations": [row["duration"] for row in language_rows],
        }

    if not rows:
        raise RuntimeError("no supported cached audio could be materialized")
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "cache_root": args.cache_root,
        "samples": len(rows),
        "audio_seconds": sum(float(row["duration"]) for row in rows),
        "languages": selected_report,
        "inspected": inspected,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "inspected"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
