"""Narrow inference compatibility fixes for the released Shrutam-2 source."""

from __future__ import annotations

import torch.nn.functional as F


def patch_conformer_mask_dtype() -> None:
    """Keep the convolution padding mask in the activation dtype.

    Upstream calls ``mask.float()`` unconditionally. That makes a BF16 encoder
    fail at the following depthwise convolution. The operation is otherwise
    identical; padding values remain exactly zero.
    """
    from conformer_encoder import ConvolutionModule

    if getattr(ConvolutionModule, "_shrutam_bf16_mask_patch", False):
        return

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if mask is not None:
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        x = self.depthwise_conv(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return self.dropout(x.transpose(1, 2))

    ConvolutionModule.forward = forward
    ConvolutionModule._shrutam_bf16_mask_patch = True

