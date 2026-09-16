#!/usr/bin/env python3
"""Consolidate the controlled nine-row equivalent-purple-ostrich experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
VARIANTS = [
    ("base_hf", "equiv_base_fp32enc_hf_bf16llm_beam1", "FP32 FastConformer + HF BF16 LLM"),
    ("compiled_bf16", "equiv_compile_bf16_hf_bf16llm_beam1", "compiled BF16 FastConformer + HF BF16 LLM"),
    ("trtllm_bf16", "equiv_eager_bf16enc_trtllm_bf16_beam1", "PyTorch BF16 FastConformer + TensorRT-LLM BF16"),
    ("trtllm_fp8", "equiv_eager_bf16enc_trtllm_fp8w_bf16kv_beam1", "PyTorch BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_bf16_trtllm_bf16", "equiv_trt_bf16enc_trtllm_bf16_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM BF16"),
    ("trt_bf16_trtllm_fp8", "equiv_trt_bf16enc_trtllm_fp8w_bf16kv_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_bf16_trtllm_bf16_fp8kv", "equiv_trt_bf16enc_trtllm_bf16w_fp8kv_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM BF16 weights/FP8 KV"),
    ("trt_fp8_trtllm_fp8", "equiv_trt_fp8enc_trtllm_fp8w_bf16kv_beam1", "TensorRT ModelOpt calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_fp8_trtllm_fp8_fp8kv", "equiv_trt_fp8enc_trtllm_fp8w_fp8kv_beam1", "TensorRT ModelOpt calibrated mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights/FP8 KV"),
]
VARIABLE_LEVELS = [1, 2, 8, 32, 64, 128, 256]
CHUNK_LEVELS = [1, 2, 8, 32, 64, 128, 256, 512]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def validate(rows: list[dict], expected: list[int]) -> dict:
    by_level = {int(row["concurrency"]): row for row in rows}
    missing = sorted(set(expected) - set(by_level))
    failures = {str(level): int(by_level[level]["failed"]) for level in expected if level in by_level and int(by_level[level]["failed"])}
    request_total = sum(int(by_level[level]["requests"]) for level in expected if level in by_level)
    valid_total = sum(int(by_level[level]["valid"]) for level in expected if level in by_level)
    return {
        "expected": expected,
        "missing": missing,
        "failures": failures,
        "requests": request_total,
        "valid": valid_total,
        "passed": not missing and not failures and request_total == valid_total,
    }


def main() -> None:
    rows: list[dict] = []
    payload: dict[str, object] = {}
    indicvoices = load(RESULTS / "indicvoices_hindi_valid_accuracy.json")
    indicvoices_rows = {row["key"]: row for row in indicvoices["comparison_table"]}
    for key, directory, label in VARIANTS:
        root = RESULTS / directory
        variable = load(root / "variable" / "summary.json")
        chunk = load(root / "chunk_1s" / "summary.json")
        fleurs_smoke_accuracy = load(root / "quick_accuracy.json")
        accuracy = indicvoices["variants"][key]
        health = load(root / "variable" / "health.json")
        validation = {
            "variable": validate(variable, VARIABLE_LEVELS),
            "chunk_1s": validate(chunk, CHUNK_LEVELS),
        }
        payload[key] = {
            "directory": directory,
            "label": label,
            "beam": 1,
            "variable": variable,
            "chunk_1s": chunk,
            "accuracy": accuracy,
            "fleurs_overlap_smoke_accuracy": fleurs_smoke_accuracy,
            "health": health,
            "validation": validation,
        }
        row = dict(indicvoices_rows[key])
        row["fleurs_overlap_smoke_wer"] = float(fleurs_smoke_accuracy["wer"])
        row["fleurs_overlap_smoke_cer"] = float(fleurs_smoke_accuracy["cer"])
        rows.append(row)

    artifact_paths = [
        ARTIFACTS / "decoder_bf16" / "model.safetensors",
        ARTIFACTS / "shrutam2_encoder_equiv_bf16.plan",
        ARTIFACTS / "shrutam2_encoder_modelopt_fp8.plan",
        ARTIFACTS / "trtllm_equiv_bf16_engine" / "rank0.engine",
        ARTIFACTS / "trtllm_equiv_fp8w_bf16kv_engine" / "rank0.engine",
        ARTIFACTS / "trtllm_equiv_bf16w_fp8kv_engine" / "rank0.engine",
        ARTIFACTS / "trtllm_equiv_fp8w_fp8kv_engine" / "rank0.engine",
    ]
    inventory = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in artifact_paths if path.exists()
    }
    report = {
        "provenance": {
            "created_utc": command("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
            "host": command("hostname"),
            "brev_instance": "equivalent-purple-ostrich",
            "gpu": command("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"),
            "model": "bharatgenai/Shrutam-2",
            "model_revision": "e249bba6f7319c27912847fbbebb4258ead3b848",
            "performance_and_calibration_smoke_dataset": load(ROOT / "data/fleurs_quick/report.json"),
            "accuracy_dataset": indicvoices["dataset"],
            "beam": 1,
        },
        "measurement": {
            "end_to_end_rtfx": "sum valid input audio seconds / client wall-clock seconds",
            "boundary": "persistent HTTP client send through parsed non-empty transcript",
            "variable_concurrency": VARIABLE_LEVELS,
            "exact_1s_concurrency": CHUNK_LEVELS,
            "exact_1s": "trim or zero-pad to exactly 16000 PCM samples",
        },
        "comparison_table": rows,
        "accuracy_evaluation": indicvoices["evaluation"],
        "variants": payload,
        "artifact_inventory": inventory,
        "all_matrices_passed": all(
            item["validation"][workload]["passed"]
            for item in payload.values()
            for workload in ("variable", "chunk_1s")
        ),
    }
    output = RESULTS / "equivalent_consolidated_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "comparison_table": rows, "all_matrices_passed": report["all_matrices_passed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
