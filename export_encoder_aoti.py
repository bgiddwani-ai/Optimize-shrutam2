#!/usr/bin/env python3
"""Build a BF16 AOTInductor package for dynamic Shrutam-2 encoder shapes."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from torch.export import Dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--output", default="/workspace/artifacts/shrutam2_encoder_bf16.pt2")
    parser.add_argument("--report", default="/workspace/artifacts/aoti_build_report.json")
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=2001)
    parser.add_argument("--static-frames", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    sys.path.insert(0, str(model_dir))
    from runtime_compat import patch_conformer_mask_dtype
    patch_conformer_mask_dtype()
    from conformer_encoder import ConformerEncoder

    model = ConformerEncoder().eval().cuda().to(torch.bfloat16)
    model.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=True))
    example_frames = args.static_frames or 301
    features = torch.randn((2, 80, example_frames), device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([example_frames, max(9, example_frames - 14)], device="cuda", dtype=torch.long)
    batch = Dim("batch", min=1, max=args.max_batch)
    # The 8x convolutional subsampler requires at least nine mel frames; using
    # eight violates the shape guards produced by torch.export.
    frames = Dim("mel_frames", min=9, max=args.max_frames)
    report: dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "max_batch": args.max_batch,
        "max_frames": args.max_frames,
        "static_frames": args.static_frames,
        "aoti_api": hasattr(torch._inductor, "aoti_compile_and_package"),
    }
    try:
        dynamic_shapes = (
            ({0: batch}, {0: batch})
            if args.static_frames
            else ({0: batch, 2: frames}, {0: batch})
        )
        exported = torch.export.export(model, (features, lengths), dynamic_shapes=dynamic_shapes, strict=False)
        report["exported"] = True
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch._inductor.aoti_compile_and_package(exported, package_path=str(output))
        report["packaged"] = output.exists()
        report["package_bytes"] = output.stat().st_size if output.exists() else 0
    except Exception as exc:
        report["exported"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:5000]
        report["traceback"] = traceback.format_exc()[-5000:]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report.get("packaged"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
