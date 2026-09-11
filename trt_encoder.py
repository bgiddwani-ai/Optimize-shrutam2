#!/usr/bin/env python3
"""Zero-copy torch-tensor TensorRT 10 runner for the Conformer engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


class TensorRTEncoder:
    def __init__(self, engine_path: str, device: torch.device) -> None:
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        blob = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

    def __call__(self, audio_signal: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = {"audio_signal": audio_signal.contiguous(), "lengths": lengths.contiguous()}
        for name in self.inputs:
            expected_dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            tensor = tensors[name]
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
                tensors[name] = tensor
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"shape rejected for {name}: {tuple(tensor.shape)}")
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs: dict[str, torch.Tensor] = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"unresolved output shape for {name}: {shape}")
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            output = torch.empty(shape, device=self.device, dtype=dtype)
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return outputs["encoded"], outputs["encoded_lengths"]
