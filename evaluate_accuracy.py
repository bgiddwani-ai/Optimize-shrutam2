#!/usr/bin/env python3
"""Compute quick Unicode-aware WER/CER from benchmark request artifacts."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from jiwer import cer, wer


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.requests).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("ok") and int(row.get("concurrency", -1)) == args.concurrency and row.get("reference")]
    deduplicated = {}
    for row in rows:
        deduplicated.setdefault(str(row.get("path")), row)
    rows = list(deduplicated.values())
    references = [normalize(str(row["reference"])) for row in rows]
    hypotheses = [normalize(str(row["text"])) for row in rows]
    valid = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
    if not valid:
        raise ValueError("no valid reference/hypothesis pairs")
    references, hypotheses = map(list, zip(*valid))
    report = {
        "samples": len(references),
        "concurrency": args.concurrency,
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "pairs": [{"reference": ref, "hypothesis": hyp} for ref, hyp in zip(references, hypotheses)],
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
