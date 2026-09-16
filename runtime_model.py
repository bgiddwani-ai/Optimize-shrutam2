#!/usr/bin/env python3
"""Shrutam-2 inference runtime with correct variable-length microbatching.

The upstream sample performs single-item inference and does not carry encoder
lengths into SMEAR routing.  This runtime keeps the log-mel frontend in FP32,
runs the Conformer/projector/LLM in BF16, masks padding during SMEAR routing,
and packs each item's audio prefix immediately before its text prompt.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGE_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "kn": "Kannada", "or": "Odia", "bn": "Bengali",
    "ur": "Urdu", "as": "Assamese", "gu": "Gujarati", "pa": "Punjabi",
}


@dataclass
class DecodeConfig:
    max_new_tokens: int = 96
    num_beams: int = 1
    repetition_penalty: float = 1.3
    length_penalty: float = 0.8


def _load_python_model_config(model_dir: Path):
    config_file = model_dir / "inference_config.py"
    spec = importlib.util.spec_from_file_location("shrutam_inference_config", config_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inference_config, module.model_config


class ShrutamRuntime:
    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        device: str = "cuda",
        runtime: str = "eager",
        precision: str = "bf16",
        decode: DecodeConfig | None = None,
        trt_engine: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.runtime_name = runtime
        if precision not in {"upstream", "bf16"}:
            raise ValueError(f"unknown precision: {precision}")
        self.precision = precision
        self.decode = decode or DecodeConfig()
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.llm_dtype = torch.bfloat16
        self._load(trt_engine=trt_engine)

    def _load(self, *, trt_engine: str | None) -> None:
        sys.path.insert(0, str(self.model_dir))
        # The released speech_encoder imports torchaudio but never uses it in
        # preprocessing or Conformer inference. NeMo 25.02 intentionally omits
        # torchaudio, so satisfy only that incidental import without replacing
        # NVIDIA's pinned torch build.
        if importlib.util.find_spec("torchaudio") is None:
            sys.modules["torchaudio"] = types.ModuleType("torchaudio")
        from runtime_compat import patch_conformer_mask_dtype
        patch_conformer_mask_dtype()
        from asr_model import ASRModel, EncoderDownsamplerCov1d, EncoderProjectorLinear
        from speech_encoder import SpeechEncoder

        inference_config, model_config = _load_python_model_config(self.model_dir)
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "llm")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Decoder-only generation reads the final prompt position. Mixed audio
        # lengths therefore need left padding; right padding makes shorter
        # rows decode from padding and can terminate immediately.
        self.tokenizer.padding_side = "left"
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_dir / "llm",
            torch_dtype=self.llm_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.llm.config.use_cache = True
        self.llm.generation_config.use_cache = True
        encoder = SpeechEncoder(str(self.model_dir / "encoder.pt"))
        projectors = [EncoderProjectorLinear(model_config) for _ in range(model_config.num_experts)]
        downsampler = EncoderDownsamplerCov1d(model_config)
        full = ASRModel(
            encoder=encoder,
            llm=self.llm,
            encoder_projector=projectors,
            down_sampler=downsampler,
            tokenizer=self.tokenizer,
            train_config=inference_config,
            model_config=model_config,
        )
        checkpoint = torch.load(self.model_dir / "model.pt", map_location="cpu", weights_only=True)
        incompatible = full.load_state_dict(checkpoint, strict=False)
        self.checkpoint_missing = list(incompatible.missing_keys)
        self.checkpoint_unexpected = list(incompatible.unexpected_keys)

        self.preprocessor = encoder.preprocessor.eval().to(self.device, dtype=torch.float32)
        self.encoder = encoder.model.eval().to(self.device, dtype=self.dtype)
        self.downsampler = full.down_sampler.eval().to(self.device, dtype=self.dtype)
        self.moe = full.MoELayer_routing.eval().to(self.device, dtype=self.dtype)
        self.llm = full.llm.eval().to(self.device, dtype=self.llm_dtype)
        del full

        if self.runtime_name == "compile":
            self.encoder = torch.compile(
                self.encoder,
                dynamic=True,
                fullgraph=False,
                mode=os.getenv("SHRUTAM_COMPILE_MODE", "reduce-overhead"),
            )
        elif self.runtime_name == "trt":
            if not trt_engine:
                raise ValueError("trt runtime requires --trt-engine")
            from trt_encoder import TensorRTEncoder
            self.encoder = TensorRTEncoder(trt_engine, self.device)
        elif self.runtime_name != "eager":
            raise ValueError(f"unknown runtime: {self.runtime_name}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def prompt_for(self, language: str, override: str | None = None) -> str:
        if override:
            instruction = override
        else:
            name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES["hi"])
            instruction = f"Transcribe speech to {name} text."
        return self.model_config.prompt_template.format(instruction)

    @torch.inference_mode()
    def _encode(self, waveforms: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        host_lengths = [w.shape[0] for w in waveforms]
        max_samples = max(host_lengths)
        # One contiguous H2D transfer is materially cheaper than one transfer
        # per request at c128/c256/c512.
        host_audio = np.zeros((len(waveforms), max_samples), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            host_audio[index, : waveform.shape[0]] = waveform
        audio = torch.from_numpy(host_audio).to(self.device)
        lengths = torch.tensor(host_lengths, device=self.device, dtype=torch.long)
        log_mel, mel_lengths = self.preprocessor(audio, lengths)
        encoder_input = log_mel.to(self.dtype)
        # Upstream ConvolutionModule multiplies BF16 activations by
        # ``mask.float()``. Autocast converts that temporary FP32 tensor back to
        # BF16 for the following depthwise convolution without changing math or
        # checkpoint weights.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            if self.runtime_name == "trt":
                encoded, encoded_lengths = self.encoder(encoder_input, mel_lengths.to(torch.int32))
            else:
                encoded, encoded_lengths = self.encoder(encoder_input, mel_lengths)
        if encoded.ndim != 3:
            raise RuntimeError(f"encoder produced invalid shape {tuple(encoded.shape)}")
        # PyTorch and TensorRT exports emit [B,D,T].
        encoded = encoded.transpose(1, 2).contiguous()
        return encoded.to(self.dtype), encoded_lengths.to(torch.long)

    @torch.inference_mode()
    def _project(self, encoded: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected_input = self.downsampler(encoded)
        # Current checkpoint uses stride 1. Keep the formula correct if that changes.
        stride = int(self.downsampler.conv2.stride[0])
        kernel = int(self.downsampler.conv2.kernel_size[0])
        lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        lengths = lengths.clamp(min=1, max=projected_input.shape[1])
        time = torch.arange(projected_input.shape[1], device=self.device).unsqueeze(0)
        valid = time < lengths.unsqueeze(1)
        projected, _, _ = self.moe(projected_input, mask=valid)
        return projected, lengths

    @torch.inference_mode()
    def transcribe_batch(
        self,
        waveforms: Sequence[np.ndarray],
        languages: Sequence[str],
        prompts: Sequence[str | None] | None = None,
    ) -> tuple[list[str], dict[str, float | int | str]]:
        if not waveforms:
            return [], {"batch_size": 0, "total_ms": 0.0}
        if len(waveforms) != len(languages):
            raise ValueError("waveforms and languages must have equal length")
        prompts = prompts or [None] * len(waveforms)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        encoded, encoded_lengths = self._encode(waveforms)
        projected, projected_lengths = self._project(encoded, encoded_lengths)
        torch.cuda.synchronize(self.device)
        encoder_ms = (time.perf_counter() - started) * 1000.0

        prompt_ids = [
            self.tokenizer.encode(self.prompt_for(language, override), add_special_tokens=False)
            for language, override in zip(languages, prompts)
        ]
        total_lengths = [int(projected_lengths[i].item()) + len(ids) for i, ids in enumerate(prompt_ids)]
        max_total = max(total_lengths)
        batch = len(waveforms)
        token_ids = torch.full(
            (batch, max_total), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long
        )
        attention = torch.zeros((batch, max_total), device=self.device, dtype=torch.long)
        offsets: list[int] = []
        for index, ids in enumerate(prompt_ids):
            audio_len = int(projected_lengths[index].item())
            prompt = torch.tensor(ids, device=self.device, dtype=torch.long)
            offset = max_total - total_lengths[index]
            offsets.append(offset)
            token_ids[index, offset + audio_len : offset + audio_len + len(ids)] = prompt
            attention[index, offset : offset + audio_len + len(ids)] = 1

        embeddings = self.llm.get_input_embeddings()(token_ids)
        for index in range(batch):
            audio_len = int(projected_lengths[index].item())
            offset = offsets[index]
            embeddings[index, offset : offset + audio_len] = projected[index, :audio_len]

        generated = self.llm.generate(
            inputs_embeds=embeddings,
            attention_mask=attention,
            max_new_tokens=self.decode.max_new_tokens,
            num_beams=self.decode.num_beams,
            do_sample=False,
            use_cache=True,
            repetition_penalty=self.decode.repetition_penalty,
            length_penalty=self.decode.length_penalty,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize(self.device)
        total_ms = (time.perf_counter() - started) * 1000.0
        texts = [text.strip() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
        return texts, {
            "runtime": self.runtime_name,
            "precision": self.precision,
            "batch_size": batch,
            "encoder_projector_ms": encoder_ms,
            "decode_ms": total_ms - encoder_ms,
            "total_ms": total_ms,
            "max_audio_samples": max(w.shape[0] for w in waveforms),
            "max_encoder_tokens": int(projected_lengths.max().item()),
        }

    def warmup(self, batch_sizes: Sequence[int] = (1, 2, 8)) -> list[dict[str, float | int | str]]:
        reports = []
        previous = self.decode.max_new_tokens
        self.decode.max_new_tokens = min(previous, 8)
        try:
            for batch in batch_sizes:
                waves = [np.zeros(16000, dtype=np.float32) for _ in range(batch)]
                _, report = self.transcribe_batch(waves, ["hi"] * batch)
                reports.append(report)
        finally:
            self.decode.max_new_tokens = previous
        return reports
