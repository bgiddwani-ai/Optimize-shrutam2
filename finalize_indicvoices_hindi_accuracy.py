#!/usr/bin/env python3
"""Score all nine variants on one common IndicVoices Hindi sample set."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from jiwer import cer, wer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONCURRENCY = 32
VARIANTS = [
    ("base_hf", "equiv_base_fp32enc_hf_bf16llm_beam1", "FP32 FastConformer + HF BF16 LLM"),
    ("compiled_bf16", "equiv_compile_bf16_hf_bf16llm_beam1", "compiled BF16 FastConformer + HF BF16 LLM"),
    ("trtllm_bf16", "equiv_eager_bf16enc_trtllm_bf16_beam1", "PyTorch BF16 FastConformer + TensorRT-LLM BF16"),
    ("trtllm_fp8", "equiv_eager_bf16enc_trtllm_fp8w_bf16kv_beam1", "PyTorch BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_bf16_trtllm_bf16", "equiv_trt_bf16enc_trtllm_bf16_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM BF16"),
    ("trt_bf16_trtllm_fp8", "equiv_trt_bf16enc_trtllm_fp8w_bf16kv_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_bf16_trtllm_bf16_fp8kv", "equiv_trt_bf16enc_trtllm_bf16w_fp8kv_beam1", "TensorRT BF16 FastConformer + TensorRT-LLM BF16 weights/FP8 KV"),
    ("trt_fp8_trtllm_fp8", "equiv_trt_fp8enc_trtllm_fp8w_bf16kv_beam1", "TensorRT ModelOpt mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights/BF16 KV"),
    ("trt_fp8_trtllm_fp8_fp8kv", "equiv_trt_fp8enc_trtllm_fp8w_fp8kv_beam1", "TensorRT ModelOpt mixed-FP8 FastConformer + TensorRT-LLM ModelOpt FP8 weights/FP8 KV"),
]


def normalize(text: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).strip().split())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_requests(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    manifest_path = ROOT / "data/indicvoices_hindi_valid/manifest.jsonl"
    manifest = load_requests(manifest_path)
    manifest_paths = [str(row["audio_filepath"]).replace(str(ROOT), "/workspace") for row in manifest]
    manifest_references = {
        path: normalize(row["text"])
        for path, row in zip(manifest_paths, manifest)
    }
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("IndicVoices manifest paths are not unique")

    loaded: dict[str, dict] = {}
    for key, directory, label in VARIANTS:
        request_path = RESULTS / directory / "indicvoices_hindi_valid_accuracy_requests/requests.jsonl"
        summary_path = request_path.parent / "summary.json"
        rows = [row for row in load_requests(request_path) if int(row.get("concurrency", -1)) == CONCURRENCY]
        if len(rows) != len(manifest):
            raise ValueError(f"{key}: expected {len(manifest)} requests, found {len(rows)}")
        by_path: dict[str, dict] = {}
        for row in rows:
            path = str(row.get("path"))
            if path in by_path:
                raise ValueError(f"{key}: duplicate request path {path}")
            by_path[path] = row
        if set(by_path) != set(manifest_paths):
            missing = sorted(set(manifest_paths) - set(by_path))
            unexpected = sorted(set(by_path) - set(manifest_paths))
            raise ValueError(f"{key}: manifest mismatch; missing={missing[:3]} unexpected={unexpected[:3]}")
        successful = {
            path
            for path, row in by_path.items()
            if row.get("ok") and normalize(row.get("reference")) and normalize(row.get("text"))
        }
        loaded[key] = {
            "directory": directory,
            "label": label,
            "rows": rows,
            "by_path": by_path,
            "successful": successful,
            "benchmark": load_json(summary_path),
        }

    scored_paths = manifest_paths
    references = [manifest_references[path] for path in scored_paths]
    if not all(references):
        raise ValueError("cleaned manifest contains an empty reference")
    comparison: list[dict] = []
    variant_reports: dict[str, dict] = {}
    for key, directory, label in VARIANTS:
        item = loaded[key]
        by_path = item["by_path"]
        request_references = [normalize(by_path[path].get("reference")) for path in scored_paths]
        mismatched_references = [
            path for path, reference in zip(scored_paths, request_references)
            if reference and reference != manifest_references[path]
        ]
        if mismatched_references:
            raise ValueError(f"{key}: reference mismatch on {mismatched_references[:3]}")
        hypotheses = [normalize(by_path[path].get("text")) for path in scored_paths]
        failed_rows = [row for row in item["rows"] if str(row.get("path")) not in item["successful"]]
        error_counts = Counter(str(row.get("error") or "empty reference or hypothesis") for row in failed_rows)
        successful_rows = [by_path[path] for path in scored_paths if path in item["successful"]]
        successful_references = [manifest_references[str(row["path"])] for row in successful_rows]
        successful_hypotheses = [normalize(row.get("text")) for row in successful_rows]
        score = {
            "wer": wer(references, hypotheses),
            "cer": cer(references, hypotheses),
            "successful_only_wer": wer(successful_references, successful_hypotheses),
            "successful_only_cer": cer(successful_references, successful_hypotheses),
        }
        report = {
            "directory": directory,
            "label": label,
            "attempted": len(item["rows"]),
            "successful_nonempty": len(item["successful"]),
            "failed_or_empty": len(failed_rows),
            "failure_counts": dict(sorted(error_counts.items())),
            "failure_paths": [str(row.get("path")) for row in failed_rows],
            "scored_samples": len(scored_paths),
            **score,
            "benchmark": item["benchmark"],
        }
        variant_reports[key] = report
        comparison.append({
            "key": key,
            "label": label,
            "beam": 1,
            "samples": len(scored_paths),
            "attempted": len(item["rows"]),
            "failed_or_empty": len(failed_rows),
            **score,
        })

    baseline = comparison[0]
    for row in comparison:
        row["wer_delta_pp_vs_base"] = (row["wer"] - baseline["wer"]) * 100.0
        row["cer_delta_pp_vs_base"] = (row["cer"] - baseline["cer"]) * 100.0
        row["within_1_5pp_wer_budget"] = abs(row["wer_delta_pp_vs_base"]) <= 1.5

    report = {
        "dataset": load_json(ROOT / "data/indicvoices_hindi_valid/report.json"),
        "evaluation": {
            "split": "Hindi validation",
            "beam": 1,
            "concurrency": CONCURRENCY,
            "max_new_tokens": 96,
            "normalization": "Unicode NFKC, strip, collapse whitespace",
            "comparison_policy": "WER/CER for every row use all cleaned manifest samples; failed or empty transcripts are scored as empty hypotheses and therefore receive full deletion penalties",
            "manifest_samples_after_cleaning": len(manifest),
            "scored_samples": len(scored_paths),
            "path_sha256": digest_lines(scored_paths),
            "reference_sha256": digest_lines(references),
        },
        "comparison_table": comparison,
        "variants": variant_reports,
        "all_variants_complete": all(row["attempted"] == len(manifest) for row in comparison),
        "all_requests_successful": all(row["failed_or_empty"] == 0 for row in comparison),
    }
    output = RESULTS / "indicvoices_hindi_valid_accuracy.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "scored_samples": len(scored_paths),
        "comparison_table": comparison,
        "all_variants_complete": report["all_variants_complete"],
        "all_requests_successful": report["all_requests_successful"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
