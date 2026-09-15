#!/usr/bin/env python3
"""Insert calibrated FP8 Q/DQ nodes into the FastConformer ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from modelopt.onnx.quantization import quantize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="/workspace/artifacts/shrutam2_encoder_fp32.onnx")
    parser.add_argument("--calibration", default="/workspace/artifacts/encoder_fp8_calibration.npz")
    parser.add_argument("--output", default="/workspace/artifacts/shrutam2_encoder_modelopt_fp8.onnx")
    parser.add_argument("--log", default="/workspace/logs/equivalent_encoder_modelopt_fp8.log")
    args = parser.parse_args()
    with np.load(args.calibration) as values:
        calibration = {
            "audio_signal": np.ascontiguousarray(values["audio_signal"], dtype=np.float32),
            "lengths": np.ascontiguousarray(values["lengths"], dtype=np.int64),
        }
    quantize(
        args.onnx,
        quantize_mode="fp8",
        calibration_data=calibration,
        calibration_method="max",
        calibration_eps=["cuda:0", "cpu"],
        op_types_to_quantize=["Conv", "MatMul"],
        use_external_data_format=True,
        output_path=args.output,
        log_level="INFO",
        log_file=args.log,
        high_precision_dtype="bf16",
        mha_accumulation_dtype="fp32",
        opset=19,
    )
    model = onnx.load(args.output, load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    report = {
        "source_onnx": args.onnx,
        "calibration": args.calibration,
        "output": args.output,
        "output_bytes": Path(args.output).stat().st_size,
        "quantize_linear_nodes": counts.get("QuantizeLinear", 0),
        "dequantize_linear_nodes": counts.get("DequantizeLinear", 0),
        "operator_counts": counts,
    }
    Path(args.output + ".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
