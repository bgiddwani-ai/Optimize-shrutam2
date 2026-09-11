#!/usr/bin/env python3
"""Record why the duplicated encoder_projector state names are harmless."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/model/model.pt")
    parser.add_argument("--output", default="/workspace/artifacts/checkpoint_alias_report.json")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    expert = sorted(key for key in state if key.startswith("MoELayer_routing.experts."))
    duplicate = sorted(key for key in state if key.startswith("encoder_projector."))
    report = {
        "checkpoint_tensor_count": len(state),
        "moe_expert_tensor_count": len(expert),
        "duplicate_encoder_projector_tensor_count": len(duplicate),
        "moe_expert_examples": expert[:8],
        "explanation": (
            "ASRModel registers the same projector modules under MoELayer_routing.experts "
            "and encoder_projector. The checkpoint stores the former names, so strict=False "
            "reports the latter alias names as missing even though the shared modules were loaded."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
