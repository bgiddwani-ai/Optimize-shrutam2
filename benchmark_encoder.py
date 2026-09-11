#!/usr/bin/env python3
"""Real-audio encoder-only RTFx sweep; never labels this as end-to-end ASR."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import types
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(manifest: Path, fixed_seconds: float | None = None) -> list[np.ndarray]:
    waves = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row.get("audio_filepath") or row.get("audio") or row.get("path"))
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        wave, rate = sf.read(path, dtype="float32")
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if rate != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {rate}")
        if fixed_seconds is not None:
            target = int(round(fixed_seconds * rate))
            wave = wave[:target] if wave.shape[0] >= target else np.pad(wave, (0, target - wave.shape[0]))
        waves.append(wave)
    return waves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime", choices=("eager", "compile", "aoti", "trt"), default="eager")
    parser.add_argument("--aoti-package")
    parser.add_argument("--trt-engine")
    parser.add_argument("--fixed-seconds", type=float)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    if importlib.util.find_spec("torchaudio") is None:
        sys.modules["torchaudio"] = types.ModuleType("torchaudio")
    from runtime_compat import patch_conformer_mask_dtype
    patch_conformer_mask_dtype()
    from speech_encoder import MelSpectrogramPreprocessor
    from conformer_encoder import ConformerEncoder

    device = torch.device("cuda")
    dtype = torch.bfloat16
    preprocessor = MelSpectrogramPreprocessor().eval().to(device)
    encoder = ConformerEncoder().eval().to(device, dtype=dtype)
    encoder.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=True))
    if args.runtime == "compile":
        encoder = torch.compile(encoder, dynamic=True, mode="reduce-overhead")
    elif args.runtime == "aoti":
        encoder = torch._inductor.aoti_load_package(args.aoti_package)
    elif args.runtime == "trt":
        from trt_encoder import TensorRTEncoder
        encoder = TensorRTEncoder(args.trt_engine, device)
    waves = load_audio(Path(args.manifest), args.fixed_seconds)

    reports = []
    for batch_size in args.batch_sizes:
        selected = [waves[index % len(waves)] for index in range(batch_size)]
        lengths = torch.tensor([wave.shape[0] for wave in selected], device=device, dtype=torch.long)
        audio = torch.zeros((batch_size, int(lengths.max().item())), device=device)
        for index, wave in enumerate(selected):
            audio[index, : wave.shape[0]] = torch.from_numpy(wave).to(device)
        with torch.inference_mode():
            mel, mel_lengths = preprocessor(audio, lengths)
            mel = mel.to(dtype)
            call_lengths = mel_lengths.to(torch.int32) if args.runtime == "trt" else mel_lengths
            for _ in range(args.warmup):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    encoder(mel, call_lengths)
            torch.cuda.synchronize()
            timings = []
            for _ in range(args.runs):
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype):
                    encoded, encoded_lengths = encoder(mel, call_lengths)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
        mean_seconds = statistics.fmean(timings)
        audio_seconds = float(lengths.sum().item()) / 16000.0
        report = {
            "runtime": args.runtime,
            "batch_size": batch_size,
            "mean_latency_ms": mean_seconds * 1000.0,
            "p50_latency_ms": float(np.percentile(timings, 50) * 1000.0),
            "audio_seconds": audio_seconds,
            "rtfx": audio_seconds / mean_seconds,
            "encoded_shape": list(encoded.shape),
            "valid_encoded_frames": int(encoded_lengths.sum().item()),
        }
        print(json.dumps(report), flush=True)
        reports.append(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)


if __name__ == "__main__":
    main()
