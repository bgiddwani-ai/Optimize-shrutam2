#!/usr/bin/env python3
"""Materialize the fine-tuned Shrutam decoder as a vLLM model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/nvidia/Optimize-shrutam2/model")
    parser.add_argument("--output-dir", default="/home/nvidia/Optimize-shrutam2/artifacts/vllm_llm_bf16")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    source_llm = model_dir / "llm"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    decoder = {
        key.removeprefix("llm."): value.detach().to(torch.bfloat16).contiguous()
        for key, value in checkpoint.items()
        if key.startswith("llm.")
    }
    with safe_open(source_llm / "model.safetensors", framework="pt", device="cpu") as base:
        expected = set(base.keys())

    missing = sorted(expected - decoder.keys())
    extra = sorted(decoder.keys() - expected)
    if missing:
        raise RuntimeError(f"fine-tuned checkpoint is missing decoder keys: {missing}")

    tied_equal = None
    if "lm_head.weight" in decoder and "model.embed_tokens.weight" in decoder:
        tied_equal = bool(torch.equal(decoder["lm_head.weight"], decoder["model.embed_tokens.weight"]))
        if not tied_equal:
            raise RuntimeError("tied lm_head and embedding weights differ")

    output_state = {key: decoder[key] for key in sorted(expected)}
    weights_path = output / "model.safetensors"
    save_file(output_state, weights_path, metadata={"format": "pt"})
    for source in source_llm.iterdir():
        if source.is_file() and source.name != "model.safetensors":
            shutil.copy2(source, output / source.name)

    report = {
        "source_checkpoint": str(model_dir / "model.pt"),
        "tensor_count": len(output_state),
        "missing": missing,
        "extra_ignored": extra,
        "tied_lm_head_matches_embedding": tied_equal,
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256(weights_path),
    }
    (output / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
