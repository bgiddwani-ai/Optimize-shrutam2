#!/usr/bin/env python3
"""Consolidate PyTorch's many ONNX external initializers into one data file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import onnx


def external_locations(model: onnx.ModelProto) -> list[str]:
    locations: list[str] = []
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in tensor.external_data:
            if item.key == "location":
                locations.append(item.value)
    return sorted(set(locations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source = Path(args.model).resolve()
    parent = source.parent
    metadata_only = onnx.load(str(source), load_external_data=False)
    old_locations = external_locations(metadata_only)
    old_files: list[Path] = []
    for location in old_locations:
        candidate = (parent / location).resolve()
        if candidate.parent != parent:
            raise RuntimeError(f"refusing external data outside model directory: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        old_files.append(candidate)

    loaded = onnx.load(str(source), load_external_data=True)
    consolidated = source.with_suffix(".consolidated.onnx")
    data_name = source.name + ".data"
    data_path = parent / data_name
    onnx.save_model(
        loaded,
        str(consolidated),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    verified = onnx.load(str(consolidated), load_external_data=False)
    onnx.checker.check_model(str(consolidated))
    report = {
        "source": str(source),
        "old_external_files": len(old_files),
        "old_external_bytes": sum(path.stat().st_size for path in old_files),
        "consolidated": str(consolidated),
        "consolidated_graph_bytes": consolidated.stat().st_size,
        "consolidated_data_bytes": data_path.stat().st_size,
        "nodes": len(verified.graph.node),
        "initializers": len(verified.graph.initializer),
        "replaced": args.replace,
    }
    if args.replace:
        for path in old_files:
            path.unlink()
        source.unlink()
        os.replace(consolidated, source)
        report["model"] = str(source)
        report["data"] = str(data_path)
    Path(str(source) + ".external_data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
