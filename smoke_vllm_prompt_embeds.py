#!/usr/bin/env python3
"""Verify that vLLM prompt embeddings match the token-ID input path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/nvidia/Optimize-shrutam2/model/llm")
    parser.add_argument("--output", default="/home/nvidia/Optimize-shrutam2/artifacts/vllm_prompt_embed_smoke.json")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = "Transcribe speech to Hindi text."
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor("model.embed_tokens.weight")
    prompt_embeddings = F.embedding(torch.tensor(token_ids, dtype=torch.long), weight).contiguous()

    engine = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=8,
        gpu_memory_utilization=0.65,
        enable_prompt_embeds=True,
        enforce_eager=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=16, skip_special_tokens=True)
    outputs = engine.generate(
        [{"prompt_token_ids": token_ids}, {"prompt_embeds": prompt_embeddings}],
        params,
        use_tqdm=False,
    )
    token_path = list(outputs[0].outputs[0].token_ids)
    embed_path = list(outputs[1].outputs[0].token_ids)
    report = {
        "passed": token_path == embed_path,
        "prompt_tokens": len(token_ids),
        "token_path": token_path,
        "embed_path": embed_path,
        "token_text": outputs[0].outputs[0].text,
        "embed_text": outputs[1].outputs[0].text,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("prompt embedding parity failed")


if __name__ == "__main__":
    main()
