#!/usr/bin/env python3
"""Run real Shrutam audio-prefix decoding through a TensorRT-LLM engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer

from runtime_compat import patch_conformer_mask_dtype
from runtime_model import LANGUAGE_NAMES


def load_model_config(model_dir: Path):
    spec = importlib.util.spec_from_file_location("shrutam_config", model_dir / "inference_config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load inference_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_config


class Frontend:
    def __init__(self, model_dir: Path) -> None:
        sys.path.insert(0, str(model_dir))
        if importlib.util.find_spec("torchaudio") is None:
            sys.modules["torchaudio"] = types.ModuleType("torchaudio")
        patch_conformer_mask_dtype()
        from asr_model import EncoderDownsamplerCov1d, EncoderProjectorLinear
        from smear import MoELayer_SMEAR
        from speech_encoder import SpeechEncoder

        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.config = load_model_config(model_dir)
        released = SpeechEncoder(str(model_dir / "encoder.pt"))
        projectors = [EncoderProjectorLinear(self.config) for _ in range(self.config.num_experts)]
        self.preprocessor = released.preprocessor.eval().to(self.device, dtype=torch.float32)
        self.encoder = released.model.eval().to(self.device, dtype=self.dtype)
        self.downsampler = EncoderDownsamplerCov1d(self.config).eval()
        self.moe = MoELayer_SMEAR(torch.nn.ModuleList(projectors), input_dim=self.config.encoder_dim).eval()
        state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        down_state = {key.removeprefix("down_sampler."): value for key, value in state.items()
                      if key.startswith("down_sampler.")}
        moe_state = {key.removeprefix("MoELayer_routing."): value for key, value in state.items()
                     if key.startswith("MoELayer_routing.")}
        self.downsampler.load_state_dict(down_state)
        self.moe.load_state_dict(moe_state)
        self.downsampler.to(self.device, dtype=self.dtype)
        self.moe.to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        audio = torch.from_numpy(waveform).to(self.device, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([waveform.shape[0]], device=self.device, dtype=torch.long)
        mel, mel_lengths = self.preprocessor(audio, lengths)
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            encoded, encoded_lengths = self.encoder(mel.to(self.dtype), mel_lengths)
            encoded = encoded.transpose(1, 2).contiguous()
            projected_input = self.downsampler(encoded)
            time_axis = torch.arange(projected_input.shape[1], device=self.device).unsqueeze(0)
            valid = time_axis < encoded_lengths.unsqueeze(1)
            projected, _, _ = self.moe(projected_input, mask=valid)
        return projected[0, : int(encoded_lengths[0].item())].contiguous()


def load_rows(manifest: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def read_wave(path: Path) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32")
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        raise ValueError(f"expected 16 kHz, got {rate}: {path}")
    return np.ascontiguousarray(wave, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/model")
    parser.add_argument("--engine-dir", default="/workspace/artifacts/trtllm_llm_bf16_engine")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "llm", local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm_config = json.loads((model_dir / "llm" / "config.json").read_text(encoding="utf-8"))
    engine_vocab_size = int(llm_config["vocab_size"])
    frontend = Frontend(model_dir)
    runner = ModelRunnerCpp.from_dir(
        args.engine_dir, rank=0, max_batch_size=32, max_input_len=256,
        max_output_len=args.max_new_tokens, max_beam_width=args.num_beams,
        kv_cache_free_gpu_memory_fraction=0.5, device_ids=[0],
    )

    output_rows = []
    for index, row in enumerate(load_rows(Path(args.manifest), args.limit)):
        path = Path(row.get("audio_filepath") or row.get("path") or row.get("audio"))
        wave = read_wave(path)
        language = str(row.get("language") or row.get("lang") or "hi")
        prompt = frontend.config.prompt_template.format(
            f"Transcribe speech to {LANGUAGE_NAMES.get(language, 'Hindi')} text."
        )
        started = time.perf_counter()
        table = frontend(wave)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        fake_ids = torch.arange(
            engine_vocab_size,
            engine_vocab_size + table.shape[0],
            device="cuda", dtype=torch.int32,
        )
        input_ids = torch.cat((fake_ids, torch.tensor(prompt_ids, device="cuda", dtype=torch.int32)))
        outputs = runner.generate(
            batch_input_ids=input_ids.unsqueeze(0),
            max_new_tokens=args.max_new_tokens,
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id,
            num_beams=args.num_beams,
            repetition_penalty=1.3,
            length_penalty=0.8,
            prompt_table=table.unsqueeze(0),
            prompt_tasks="0",
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()
        generated = outputs["output_ids"][0][0, input_ids.shape[0]:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        output_rows.append({
            "index": index, "ok": bool(text), "path": str(path),
            "duration_seconds": wave.shape[0] / 16000.0,
            "reference": str(row.get("text") or row.get("transcript") or ""),
            "language": language, "text": text,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "concurrency": 1, "audio_prefix_tokens": int(table.shape[0]),
        })
        print(json.dumps(output_rows[-1], ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    if not all(row["ok"] for row in output_rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
