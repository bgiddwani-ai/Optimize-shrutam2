#!/usr/bin/env python3
"""Export the Shrutam-2 Conformer encoder with dynamic batch/time axes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--output", default="/workspace/artifacts/shrutam2_encoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    from runtime_compat import patch_conformer_mask_dtype
    patch_conformer_mask_dtype()
    from conformer_encoder import ConformerEncoder

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = ConformerEncoder().eval().cuda().to(dtype)
    state = torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    features = torch.randn((2, 80, 301), device="cuda", dtype=dtype)
    lengths = torch.tensor([301, 287], device="cuda", dtype=torch.long)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (features, lengths),
            str(output),
            input_names=["audio_signal", "lengths"],
            output_names=["encoded", "encoded_lengths"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "mel_frames"},
                "lengths": {0: "batch"},
                "encoded": {0: "batch", 2: "encoded_frames"},
                "encoded_lengths": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print({"onnx": str(output), "bytes": output.stat().st_size})


if __name__ == "__main__":
    main()
