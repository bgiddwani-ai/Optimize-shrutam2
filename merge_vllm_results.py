#!/usr/bin/env python3
"""Merge the promoted vLLM run into the existing consolidated evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIANT = "vllm_continuous_bf16_adaptive_persistent_beam1"
FP8_WEIGHTS_VARIANT = "vllm_fp8w_bf16kv_adaptive_persistent_beam1"
TRT_B64_VARIANT = "vllm_trt_bf16enc_bf16llm_adaptive64_persistent_beam1_final"
TRT_B32_VARIANT = "vllm_trt_bf16enc_bf16llm_adaptive32_persistent512_beam1_final"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(rows: list[dict], expected: list[int]) -> dict:
    by_concurrency = {int(row["concurrency"]): row for row in rows}
    missing = sorted(set(expected) - set(by_concurrency))
    failures = {
        str(value): int(by_concurrency[value].get("failed", 0))
        for value in expected
        if value in by_concurrency and int(by_concurrency[value].get("failed", 0))
    }
    return {
        "expected": expected,
        "missing": missing,
        "failures": failures,
        "passed": not missing and not failures,
    }


def improvement(current: list[dict], baseline: list[dict]) -> list[dict]:
    base = {int(row["concurrency"]): float(row["rtfx"]) for row in baseline}
    return [
        {
            "concurrency": int(row["concurrency"]),
            "rtfx_ratio": float(row["rtfx"]) / base[int(row["concurrency"])],
        }
        for row in current
    ]


def load_variant(name: str) -> dict:
    directory = RESULTS / name
    return {
        "variant": name,
        "variable": read_json(directory / "variable" / "summary.json"),
        "chunk_1s": read_json(directory / "chunk_1s" / "summary.json"),
        "accuracy": read_json(directory / "quick_accuracy.json"),
        "server_stats_final": read_json(directory / "server_stats_final.json"),
    }


def main() -> None:
    consolidated_path = RESULTS / "consolidated_results.json"
    report = read_json(consolidated_path)
    directory = RESULTS / VARIANT
    variable = read_json(directory / "variable" / "summary.json")
    chunk = read_json(directory / "chunk_1s" / "summary.json")
    accuracy = read_json(directory / "quick_accuracy.json")
    stats = read_json(directory / "server_stats_final.json")

    eager = report["http_server_client"]["throughput_profile_eager_bf16_beam1"]
    report["measurement"]["vllm_transport"] = (
        "PCM conversion and persistent-stream connection establishment are untimed; "
        "timing starts immediately before the request wave and ends after every response"
    )
    report["http_server_client"]["vllm_adaptive_persistent_bf16_beam1"] = {
        "variant": VARIANT,
        "variable": variable,
        "chunk_1s": chunk,
        "accuracy": accuracy,
        "server_stats_final": stats,
        "improvement_vs_eager_bf16_beam1": {
            "variable": improvement(variable, eager["variable"]),
            "chunk_1s": improvement(chunk, eager["chunk_1s"]),
        },
    }
    report["matrix_validation"]["vllm_variable"] = validate(
        variable, [1, 2, 8, 32, 64, 128, 256]
    )
    report["matrix_validation"]["vllm_chunk_1s"] = validate(
        chunk, [1, 2, 8, 32, 64, 128, 256, 512]
    )

    reference_wer = float(accuracy["wer"])
    variants = {
        "vllm_fp8_weights_bf16_kv_adaptive_persistent_beam1": load_variant(
            FP8_WEIGHTS_VARIANT
        ),
        "vllm_trt_bf16_encoder_bf16_llm_adaptive64_persistent_beam1": load_variant(
            TRT_B64_VARIANT
        ),
        "vllm_trt_bf16_encoder_bf16_llm_adaptive32_persistent_beam1": load_variant(
            TRT_B32_VARIANT
        ),
    }
    for key, payload in variants.items():
        delta = float(payload["accuracy"]["wer"]) - reference_wer
        payload["wer_delta_vs_vllm_bf16"] = delta
        payload["quality_gate_abs_delta_lte_0_015"] = abs(delta) <= 0.015
        payload["improvement_vs_vllm_bf16"] = {
            "variable": improvement(payload["variable"], variable),
            "chunk_1s": improvement(payload["chunk_1s"], chunk),
        }
        report["http_server_client"][key] = payload

    report["optimization_trials"] = {
        "vllm_fp8_weights_fp8_kv": {
            "status": "rejected_before_matrix",
            "health": read_json(
                RESULTS
                / "vllm_fp8w_fp8kv_adaptive_persistent_beam1"
                / "accuracy_requests"
                / "health.json"
            ),
            "reason": (
                "first real-audio warmup returned an empty transcript; vLLM logged "
                "uncalibrated FP8 attention q/prob scales"
            ),
        },
        "promoted": TRT_B64_VARIANT,
        "quality_budget_absolute_wer": 0.015,
    }
    for label, payload in (
        ("vllm_fp8_weights", variants["vllm_fp8_weights_bf16_kv_adaptive_persistent_beam1"]),
        ("vllm_trt_b64", variants["vllm_trt_bf16_encoder_bf16_llm_adaptive64_persistent_beam1"]),
        ("vllm_trt_b32", variants["vllm_trt_bf16_encoder_bf16_llm_adaptive32_persistent_beam1"]),
    ):
        report["matrix_validation"][f"{label}_variable"] = validate(
            payload["variable"], [1, 2, 8, 32, 64, 128, 256]
        )
        report["matrix_validation"][f"{label}_chunk_1s"] = validate(
            payload["chunk_1s"], [1, 2, 8, 32, 64, 128, 256, 512]
        )

    report.setdefault("artifact_inventory_additions", {})[
        "artifacts/shrutam2_encoder_bf16_b64.plan"
    ] = {
        "bytes": 1416359964,
        "sha256": "bc3b1ad64041b7038086f52778fede0b8533a2a233ba42d23817780560aa00ef",
        "remote_only": True,
        "tensorrt": "10.10.0.31",
        "profile": {
            "min": "audio_signal:1x80x101,lengths:1",
            "opt": "audio_signal:32x80x501,lengths:32",
            "max": "audio_signal:64x80x1001,lengths:64",
        },
    }

    consolidated_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "results": str(consolidated_path),
        "variable_passed": report["matrix_validation"]["vllm_variable"]["passed"],
        "chunk_1s_passed": report["matrix_validation"]["vllm_chunk_1s"]["passed"],
    }))


if __name__ == "__main__":
    main()
