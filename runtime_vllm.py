#!/usr/bin/env python3
"""Shrutam-2 audio frontend plus a continuously-batched vLLM decoder.

The released model is a FastConformer/projector followed by a standard Llama
decoder.  The original Hugging Face ``generate`` call batches all rows in
lockstep.  This runtime keeps only the frontend and token embedding table in
the API process and submits each completed prompt embedding to vLLM, allowing
finished requests to leave the decode batch immediately.
"""

from __future__ import annotations

import asyncio
import gc
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from runtime_model import DecodeConfig, ShrutamRuntime


class ShrutamPromptFrontend:
    """Produce unpadded audio-plus-instruction embeddings for vLLM."""

    def __init__(
        self,
        model_dir: str,
        runtime: str = "compile",
        precision: str = "bf16",
        trt_engine: str | None = None,
    ) -> None:
        self.base = ShrutamRuntime(
            model_dir,
            runtime=runtime,
            precision=precision,
            decode=DecodeConfig(max_new_tokens=1, num_beams=1),
            trt_engine=trt_engine,
        )
        # Retain the tied input embedding table, but release the HF decoder.
        # vLLM owns the only decoder used for inference.
        self.token_embedding = self.base.llm.get_input_embeddings()
        del self.base.llm
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def prepare_batch(
        self,
        waveforms: Sequence[np.ndarray],
        languages: Sequence[str],
        prompts: Sequence[str | None],
    ) -> tuple[list[torch.Tensor], dict[str, float | int | str]]:
        if not waveforms:
            return [], {"batch_size": 0, "frontend_ms": 0.0}
        started = time.perf_counter()
        encoded, encoded_lengths = self.base._encode(waveforms)
        projected, projected_lengths = self.base._project(encoded, encoded_lengths)

        gpu_results: list[torch.Tensor] = []
        for index, (language, override) in enumerate(zip(languages, prompts)):
            token_ids = self.base.tokenizer.encode(
                self.base.prompt_for(language, override), add_special_tokens=False
            )
            ids = torch.tensor(token_ids, device=self.base.device, dtype=torch.long)
            text_embeddings = self.token_embedding(ids)
            audio_len = int(projected_lengths[index].item())
            prompt_embeddings = torch.cat(
                (projected[index, :audio_len], text_embeddings), dim=0
            ).to(dtype=torch.bfloat16)
            gpu_results.append(prompt_embeddings.contiguous())

        # AsyncLLM's input process transports prompt embeddings to the engine
        # process.  Pad temporarily so the entire microbatch uses one D2H copy
        # instead of one synchronous transfer per request.
        prompt_lengths = [item.shape[0] for item in gpu_results]
        padded = torch.nn.utils.rnn.pad_sequence(gpu_results, batch_first=True)
        host_batch = padded.cpu()
        results = [host_batch[i, :length].contiguous() for i, length in enumerate(prompt_lengths)]

        torch.cuda.synchronize(self.base.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return results, {
            "runtime": self.base.runtime_name,
            "precision": self.base.precision,
            "batch_size": len(waveforms),
            "frontend_ms": elapsed_ms,
            "max_audio_samples": max(w.shape[0] for w in waveforms),
            "max_prompt_tokens": max(t.shape[0] for t in results),
        }


class ShrutamVLLMRuntime:
    """Own the vLLM engine and the independent Shrutam audio frontend."""

    def __init__(
        self,
        model_dir: str,
        *,
        decoder_dir: str | None = None,
        runtime: str = "compile",
        precision: str = "bf16",
        max_new_tokens: int = 96,
        repetition_penalty: float = 1.3,
        max_model_len: int = 1024,
        max_num_seqs: int = 512,
        max_num_batched_tokens: int = 32768,
        gpu_memory_utilization: float = 0.65,
        trt_engine: str | None = None,
        llm_quantization: str | None = None,
        kv_cache_dtype: str = "auto",
        calculate_kv_scales: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.decoder_dir = Path(decoder_dir).resolve() if decoder_dir else self.model_dir / "llm"
        self.max_new_tokens = max_new_tokens
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True,
        )
        engine_args = AsyncEngineArgs(
            model=str(self.decoder_dir),
            tokenizer=str(self.decoder_dir),
            dtype="bfloat16",
            quantization=llm_quantization,
            kv_cache_dtype=kv_cache_dtype,
            calculate_kv_scales=calculate_kv_scales,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
            enable_chunked_prefill=True,
            trust_remote_code=True,
            enforce_eager=False,
            disable_log_stats=False,
        )
        # Create the engine before touching CUDA in this process.  vLLM starts
        # an isolated engine process and reserves its own KV-cache budget.
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.frontend = ShrutamPromptFrontend(
            str(self.model_dir), runtime, precision, trt_engine=trt_engine
        )
        self.config = {
            "runtime": runtime,
            "precision": precision,
            "decoder_dir": str(self.decoder_dir),
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trt_engine": trt_engine,
            "llm_quantization": llm_quantization or "none",
            "kv_cache_dtype": kv_cache_dtype,
            "calculate_kv_scales": calculate_kv_scales,
        }

    async def decode(self, prompt_embeddings: torch.Tensor) -> tuple[str, dict[str, float | int]]:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        final = None
        async for output in self.engine.generate(
            {"prompt_embeds": prompt_embeddings}, self.sampling_params, request_id
        ):
            final = output
        if final is None or not final.outputs:
            raise RuntimeError("vLLM returned no output")
        text = final.outputs[0].text.strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return text, {
            "decode_ms": elapsed_ms,
            "generated_tokens": len(final.outputs[0].token_ids),
            "prompt_tokens": int(prompt_embeddings.shape[0]),
        }

    async def warmup(self, batch_sizes: Sequence[int] = (1, 2, 8)) -> list[dict[str, object]]:
        reports: list[dict[str, object]] = []
        original = self.sampling_params
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=min(8, self.max_new_tokens),
            repetition_penalty=1.3,
            skip_special_tokens=True,
        )
        try:
            for batch_size in batch_sizes:
                # A low-amplitude tone exercises the real preprocessing path.
                x = np.arange(16000, dtype=np.float32)
                wave = (0.01 * np.sin(2.0 * np.pi * 220.0 * x / 16000.0)).astype(np.float32)
                embeddings, frontend = await asyncio.to_thread(
                    self.frontend.prepare_batch,
                    [wave] * batch_size,
                    ["hi"] * batch_size,
                    [None] * batch_size,
                )
                decoded = await asyncio.gather(*(self.decode(item) for item in embeddings))
                reports.append({
                    "batch_size": batch_size,
                    "frontend": frontend,
                    "decode_ms_max": max(item[1]["decode_ms"] for item in decoded),
                })
        finally:
            self.sampling_params = original
        return reports

    def shutdown(self) -> None:
        self.engine.shutdown()
