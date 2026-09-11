#!/usr/bin/env python3
"""Consolidate measured artifacts and record immutable benchmark provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def summary(variant: str, workload: str) -> list[dict]:
    payload = read_json(RESULTS / variant / workload / "summary.json")
    return payload if isinstance(payload, list) else payload.get("results", payload.get("rows", []))


def replace_baseline_retry(rows: list[dict]) -> tuple[list[dict], dict | None]:
    invalid = next((row for row in rows if int(row["concurrency"]) == 256), None)
    retry = summary("upstream_fp32conformer_bf16llm_beam4", "variable_c256_retry")
    clean = retry[0] if retry else None
    return [clean if int(row["concurrency"]) == 256 and clean else row for row in rows], invalid


def validate(rows: list[dict], expected: list[int]) -> dict:
    by_concurrency = {int(row["concurrency"]): row for row in rows}
    missing = sorted(set(expected) - set(by_concurrency))
    failures = {
        str(value): int(by_concurrency[value].get("failed", 0))
        for value in expected
        if value in by_concurrency and int(by_concurrency[value].get("failed", 0))
    }
    return {"expected": expected, "missing": missing, "failures": failures, "passed": not missing and not failures}


def trtexec_rows(precision: str) -> list[dict]:
    rows = []
    directory = RESULTS / "encoder" / f"trtexec_{precision}_1s"
    for batch in (1, 2, 8, 32):
        values = read_json(directory / f"times_b{batch}.json")
        if isinstance(values, dict):
            values = values.get("times", values.get("iterations", []))
        compute = [float(row["computeMs"]) for row in values if "computeMs" in row]
        mean_ms = sum(compute) / len(compute)
        rows.append({"batch_size": batch, "mean_gpu_compute_ms": mean_ms, "rtfx": batch * 1000.0 / mean_ms})
    return rows


def main() -> None:
    baseline_variable_raw = summary("upstream_fp32conformer_bf16llm_beam4", "variable")
    baseline_variable, invalid_baseline_c256 = replace_baseline_retry(baseline_variable_raw)
    baseline_chunk = summary("upstream_fp32conformer_bf16llm_beam4", "chunk_1s")
    compile_variable = summary("optimized_compile_bf16_beam4", "variable")
    compile_chunk = summary("optimized_compile_bf16_beam4", "chunk_1s")
    eager_variable = summary("optimized_eager_bf16_beam1", "variable")
    eager_chunk = summary("optimized_eager_bf16_beam1", "chunk_1s")
    vllm_variable = summary("vllm_continuous_bf16_adaptive_persistent_beam1", "variable")
    vllm_chunk = summary("vllm_continuous_bf16_adaptive_persistent_beam1", "chunk_1s")

    baseline_accuracy = read_json(RESULTS / "upstream_fp32conformer_bf16llm_beam4" / "quick_accuracy.json")
    compile_accuracy = read_json(RESULTS / "optimized_compile_bf16_beam4" / "quick_accuracy.json")
    eager_accuracy = read_json(RESULTS / "optimized_eager_bf16_beam1" / "quick_accuracy.json")
    vllm_accuracy = read_json(
        RESULTS / "vllm_continuous_bf16_adaptive_persistent_beam1" / "quick_accuracy.json"
    )
    trtllm_bf16 = read_json(RESULTS / "trtllm_bf16" / "quick_accuracy_beam4.json")
    trtllm_fp8 = read_json(RESULTS / "trtllm_modelopt_fp8" / "quick_accuracy_beam4.json")
    fp8_delta = float(trtllm_fp8["wer"]) - float(trtllm_bf16["wer"])

    artifact_paths = [
        ARTIFACTS / "shrutam2_encoder.onnx",
        ARTIFACTS / "shrutam2_encoder_bf16.plan",
        ARTIFACTS / "shrutam2_encoder_fp16.plan",
        ARTIFACTS / "shrutam2_encoder_bf16_1s.pt2",
        ARTIFACTS / "trtllm_llm_bf16_engine" / "rank0.engine",
        ARTIFACTS / "trtllm_llm_modelopt_fp8_engine" / "rank0.engine",
    ]
    inventory = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in artifact_paths
        if path.exists()
    }

    gpus = command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ).splitlines()
    provenance = {
        "created_utc": command("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "brev_instance": "minimum-fuchsia-mule",
        "gpus": gpus,
        "benchmark_gpu": 0,
        "model": {
            "repo": "bharatgenai/Shrutam-2",
            "revision": "e249bba6f7319c27912847fbbebb4258ead3b848",
        },
        "baseline": {
            "repo": "https://github.com/bgiddwani-ai/TensorRT-fastconformer",
            "revision": "93586a8d949d364f26c600e2f5213a9cdccf69aa",
        },
        "canary_vllm_reference": {
            "repo": "https://github.com/wuxuedaifu/Canary-Qwen-2.5b-vllm",
            "revision": "ae29af4502c9ac52eb88dbbbd6df721e4d8cac82",
        },
        "vllm_runtime": {
            "vllm": "0.12.0",
            "pytorch": "2.9.1+cu129",
            "precision": "bf16",
            "exported_decoder_bytes": 2470662568,
            "exported_decoder_sha256": (
                "727c0cf4b68d9cfac69da68e5e796ea4170d697d48a57d937a70b1853e8f81fa"
            ),
        },
        "dataset": {
            "manifest": "data/indicvoices_quick/manifest.jsonl",
            "unique_samples": 46,
            "languages": 12,
            "sample_rate_hz": 16000,
        },
        "artifact_inventory": inventory,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    report = {
        "measurement": {
            "end_to_end_rtfx": "sum valid input audio seconds / client wall-clock seconds",
            "chunk_1s": "each request is trimmed or zero-padded to exactly 16000 samples",
            "encoder_rtfx": "encoder stage only; excludes HTTP, preprocessing, projector, and decoding",
            "vllm_transport": "PCM conversion and persistent connection establishment are untimed",
        },
        "http_server_client": {
            "upstream_fp32conformer_bf16llm_beam4": {
                "variable": baseline_variable,
                "chunk_1s": baseline_chunk,
                "accuracy": baseline_accuracy,
                "invalid_original_c256": invalid_baseline_c256,
            },
            "optimized_compile_bf16_beam4": {
                "variable": compile_variable,
                "chunk_1s": compile_chunk,
                "accuracy": compile_accuracy,
            },
            "throughput_profile_eager_bf16_beam1": {
                "variable": eager_variable,
                "chunk_1s": eager_chunk,
                "accuracy": eager_accuracy,
            },
            "vllm_adaptive_persistent_bf16_beam1": {
                "variable": vllm_variable,
                "chunk_1s": vllm_chunk,
                "accuracy": vllm_accuracy,
            },
        },
        "matrix_validation": {
            "upstream_variable": validate(baseline_variable, [1, 2, 8, 32, 64, 128, 256]),
            "upstream_chunk_1s": validate(baseline_chunk, [1, 2, 8, 32, 64, 128, 256, 512]),
            "compile_variable": validate(compile_variable, [1, 2, 8, 32, 64, 128, 256]),
            "compile_chunk_1s": validate(compile_chunk, [1, 2, 8, 32, 64, 128, 256, 512]),
            "eager_variable": validate(eager_variable, [1, 2, 8, 32, 64, 128, 256]),
            "eager_chunk_1s": validate(eager_chunk, [1, 2, 8, 32, 64, 128, 256, 512]),
            "vllm_variable": validate(vllm_variable, [1, 2, 8, 32, 64, 128, 256]),
            "vllm_chunk_1s": validate(vllm_chunk, [1, 2, 8, 32, 64, 128, 256, 512]),
        },
        "encoder_stage": {
            "real_variable_audio": {
                "eager_bf16": read_csv(RESULTS / "encoder" / "eager_bf16.csv"),
                "compile_bf16": read_csv(RESULTS / "encoder" / "compile_bf16.csv"),
            },
            "exact_1s_audio": {
                "eager_bf16": read_csv(RESULTS / "encoder" / "eager_bf16_1s.csv"),
                "compile_bf16": read_csv(RESULTS / "encoder" / "compile_bf16_1s.csv"),
                "aoti_bf16_static_time": read_csv(RESULTS / "encoder" / "aoti_bf16_1s.csv"),
                "tensorrt_bf16_trtexec": trtexec_rows("bf16"),
                "tensorrt_fp16_trtexec": trtexec_rows("fp16"),
            },
        },
        "trtllm_accuracy": {
            "bf16_beam4": trtllm_bf16,
            "modelopt_fp8_beam4": trtllm_fp8,
            "fp8_minus_bf16_wer": fp8_delta,
            "strict_absolute_difference_lt_0_01": abs(fp8_delta) < 0.01,
            "directional_regression_lt_0_01": fp8_delta < 0.01,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "consolidated_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provenance": str(ARTIFACTS / "provenance.json"), "results": str(RESULTS / "consolidated_results.json")}))


if __name__ == "__main__":
    main()
