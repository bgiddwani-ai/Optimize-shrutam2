#!/usr/bin/env python3
"""Batched Shrutam FastConformer frontend with a TensorRT-LLM decoder."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tensorrt_llm.runtime import ModelRunnerCpp

from runtime_model import DecodeConfig, ShrutamRuntime


class ShrutamTRTLLMRuntime:
    def __init__(
        self,
        model_dir: str,
        engine_dir: str,
        *,
        encoder_runtime: str = "eager",
        encoder_precision: str = "bf16",
        trt_engine: str | None = None,
        max_batch_size: int = 64,
        max_input_len: int = 256,
        max_new_tokens: int = 96,
        num_beams: int = 1,
        kv_cache_fraction: float = 0.55,
        cuda_graph_mode: bool = True,
    ) -> None:
        if num_beams != 1:
            raise ValueError("the controlled benchmark requires beam 1")
        self.model_dir = Path(model_dir).resolve()
        self.engine_dir = Path(engine_dir).resolve()
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.encoder_runtime = encoder_runtime
        self.encoder_precision = encoder_precision
        self.frontend = ShrutamRuntime(
            str(self.model_dir),
            runtime=encoder_runtime,
            precision=encoder_precision,
            decode=DecodeConfig(max_new_tokens=1, num_beams=1),
            trt_engine=trt_engine,
        )
        del self.frontend.llm
        gc.collect()
        torch.cuda.empty_cache()
        self.engine_config = json.loads((self.engine_dir / "config.json").read_text(encoding="utf-8"))
        pretrained = self.engine_config["pretrained_config"]
        build = self.engine_config["build_config"]
        self.engine_vocab_size = int(pretrained["vocab_size"])
        self.decoder_dtype = str(pretrained["dtype"])
        quant = pretrained.get("quantization") or {}
        self.quant_algo = quant.get("quant_algo")
        self.kv_cache_quant_algo = quant.get("kv_cache_quant_algo")
        self.max_batch_size = min(max_batch_size, int(build["max_batch_size"]))
        self.runner = ModelRunnerCpp.from_dir(
            str(self.engine_dir),
            rank=0,
            max_batch_size=self.max_batch_size,
            max_input_len=min(max_input_len, int(build["max_input_len"])),
            max_output_len=max_new_tokens,
            max_beam_width=1,
            kv_cache_free_gpu_memory_fraction=kv_cache_fraction,
            cuda_graph_mode=cuda_graph_mode,
            device_ids=[0],
        )

    @torch.inference_mode()
    def transcribe_batch(
        self,
        waveforms: Sequence[np.ndarray],
        languages: Sequence[str],
        prompts: Sequence[str | None] | None = None,
    ) -> tuple[list[str], dict[str, object]]:
        if not waveforms:
            return [], {"batch_size": 0, "total_ms": 0.0}
        if len(waveforms) > self.max_batch_size:
            raise ValueError(f"batch {len(waveforms)} exceeds engine maximum {self.max_batch_size}")
        if len(waveforms) != len(languages):
            raise ValueError("waveforms and languages must have equal length")
        prompts = prompts or [None] * len(waveforms)
        torch.cuda.synchronize()
        started = time.perf_counter()
        encoded, encoded_lengths = self.frontend._encode(waveforms)
        projected, projected_lengths = self.frontend._project(encoded, encoded_lengths)
        torch.cuda.synchronize()
        frontend_ms = (time.perf_counter() - started) * 1000.0

        audio_lengths = [int(projected_lengths[index].item()) for index in range(len(waveforms))]
        max_audio_tokens = max(audio_lengths)
        prompt_table = torch.zeros(
            (len(waveforms), max_audio_tokens, projected.shape[-1]),
            device="cuda",
            dtype=torch.bfloat16,
        )
        batch_input_ids: list[torch.Tensor] = []
        input_lengths: list[int] = []
        for index, (language, override, audio_len) in enumerate(zip(languages, prompts, audio_lengths)):
            prompt_table[index, :audio_len].copy_(projected[index, :audio_len].to(torch.bfloat16))
            prompt_ids = self.frontend.tokenizer.encode(
                self.frontend.prompt_for(language, override), add_special_tokens=False
            )
            virtual_ids = torch.arange(
                self.engine_vocab_size,
                self.engine_vocab_size + audio_len,
                device="cuda",
                dtype=torch.int32,
            )
            text_ids = torch.tensor(prompt_ids, device="cuda", dtype=torch.int32)
            ids = torch.cat((virtual_ids, text_ids))
            batch_input_ids.append(ids)
            input_lengths.append(int(ids.shape[0]))

        outputs = self.runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=self.max_new_tokens,
            end_id=self.frontend.tokenizer.eos_token_id,
            pad_id=self.frontend.tokenizer.pad_token_id,
            num_beams=1,
            repetition_penalty=1.3,
            length_penalty=0.8,
            prompt_table=prompt_table,
            prompt_tasks=",".join(str(index) for index in range(len(waveforms))),
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        output_ids = outputs["output_ids"]
        sequence_lengths = outputs.get("sequence_lengths")
        texts: list[str] = []
        generated_counts: list[int] = []
        for index, input_len in enumerate(input_lengths):
            end = int(sequence_lengths[index, 0].item()) if sequence_lengths is not None else output_ids.shape[-1]
            generated = output_ids[index, 0, input_len:end].tolist()
            texts.append(self.frontend.tokenizer.decode(generated, skip_special_tokens=True).strip())
            generated_counts.append(len(generated))
        total_ms = (time.perf_counter() - started) * 1000.0
        return texts, {
            "encoder_runtime": self.encoder_runtime,
            "encoder_precision": self.encoder_precision,
            "decoder_dtype": self.decoder_dtype,
            "decoder_quant_algo": self.quant_algo,
            "kv_cache_quant_algo": self.kv_cache_quant_algo,
            "batch_size": len(waveforms),
            "frontend_ms": frontend_ms,
            "decode_ms": total_ms - frontend_ms,
            "total_ms": total_ms,
            "max_audio_samples": max(wave.shape[0] for wave in waveforms),
            "max_audio_tokens": max_audio_tokens,
            "generated_tokens": sum(generated_counts),
        }

    def warmup(self, batch_sizes: Sequence[int]) -> list[dict[str, object]]:
        reports: list[dict[str, object]] = []
        original = self.max_new_tokens
        self.max_new_tokens = min(original, 8)
        try:
            samples = np.arange(16000, dtype=np.float32)
            tone = (0.01 * np.sin(2.0 * np.pi * 220.0 * samples / 16000.0)).astype(np.float32)
            for batch in batch_sizes:
                _, report = self.transcribe_batch([tone] * batch, ["hi"] * batch)
                reports.append(report)
        finally:
            self.max_new_tokens = original
        return reports
