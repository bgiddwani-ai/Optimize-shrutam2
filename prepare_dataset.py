#!/usr/bin/env python3
"""Create a deterministic supported-language IndicVoices quick manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


LANGUAGES = {
    "assamese": "as", "bengali": "bn", "gujarati": "gu", "hindi": "hi",
    "kannada": "kn", "malayalam": "ml", "marathi": "mr", "odia": "or",
    "punjabi": "pa", "tamil": "ta", "telugu": "te", "urdu": "ur",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Directory containing <language>/filtered_manifest.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-language", type=int, default=4)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source = Path(args.source_dir)
    selected: list[dict[str, object]] = []
    report: dict[str, object] = {"source_dir": str(source), "languages": {}, "skipped": []}
    for language_name, code in LANGUAGES.items():
        manifest = source / language_name / "filtered_manifest.jsonl"
        if not manifest.exists():
            report["skipped"].append({"language": language_name, "reason": "manifest missing"})
            continue
        candidates = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(row["audio_filepath"])
            duration = float(row.get("duration", 0.0))
            if not audio.exists() or not row.get("text"):
                continue
            if not (args.min_seconds <= duration <= args.max_seconds):
                continue
            info = sf.info(audio)
            if info.samplerate != 16000 or info.frames < 160:
                continue
            candidates.append({
                "audio_filepath": str(audio),
                "duration": info.frames / info.samplerate,
                "text": str(row["text"]),
                "language": code,
                "dataset_config": language_name,
                "source_index": row.get("source_index"),
            })
        # Spread selections across duration rather than taking only the first utterances.
        candidates.sort(key=lambda item: float(item["duration"]))
        if len(candidates) <= args.samples_per_language:
            chosen = candidates
        else:
            indexes = [round(i * (len(candidates) - 1) / (args.samples_per_language - 1)) for i in range(args.samples_per_language)]
            chosen = [candidates[index] for index in indexes]
        selected.extend(chosen)
        report["languages"][language_name] = {
            "available": len(candidates),
            "selected": len(chosen),
            "duration_seconds": sum(float(item["duration"]) for item in chosen),
        }

    if not selected:
        raise RuntimeError(f"no usable samples under {source}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    report["selected_samples"] = len(selected)
    report["selected_audio_seconds"] = sum(float(row["duration"]) for row in selected)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

