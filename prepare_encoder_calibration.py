#!/usr/bin/env python3
"""Create real-audio log-mel calibration tensors for Model Optimizer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from runtime_compat import patch_conformer_mask_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--manifest", default="/workspace/data/fleurs_quick/manifest.jsonl")
    parser.add_argument("--output", default="/workspace/artifacts/encoder_fp8_calibration.npz")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    if importlib.util.find_spec("torchaudio") is None:
        sys.modules["torchaudio"] = types.ModuleType("torchaudio")
    patch_conformer_mask_dtype()
    from speech_encoder import SpeechEncoder

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line]
    waves: list[np.ndarray] = []
    for row in rows:
        path = Path(row["audio_filepath"])
        if not path.exists() and str(path).startswith("/workspace/"):
            path = Path("/workspace") / path.relative_to("/workspace")
        wave, rate = sf.read(path, dtype="float32")
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if rate != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {rate}")
        waves.append(np.ascontiguousarray(wave, dtype=np.float32))
    max_samples = max(wave.shape[0] for wave in waves)
    host = np.zeros((len(waves), max_samples), dtype=np.float32)
    for index, wave in enumerate(waves):
        host[index, : wave.shape[0]] = wave
    audio = torch.from_numpy(host).cuda()
    lengths = torch.tensor([wave.shape[0] for wave in waves], device="cuda", dtype=torch.long)
    preprocessor = SpeechEncoder(str(model_dir / "encoder.pt")).preprocessor.eval().cuda()
    with torch.inference_mode():
        mel, mel_lengths = preprocessor(audio, lengths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        audio_signal=mel.float().cpu().numpy(),
        lengths=mel_lengths.cpu().numpy().astype(np.int64),
    )
    report = {
        "output": str(output),
        "samples": len(waves),
        "audio_signal_shape": list(mel.shape),
        "lengths_min": int(mel_lengths.min().item()),
        "lengths_max": int(mel_lengths.max().item()),
        "source_manifest": args.manifest,
    }
    Path(str(output) + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
