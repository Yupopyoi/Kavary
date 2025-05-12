# -*- coding: utf-8 -*-

# This software is based on original work provided under the MIT License:
# - Copyright (c) 2023 liujing04
# - Copyright (c) 2023 源文雨
# - Copyright (c) 2023 Ftps
#
# The current implementation includes modifications and extensions beyond the original code.
#
# Modified and maintained by:
#
# - Copyright (c) 2025 Yupopyoi
#
# See the LICENSE file for more details.

import torch

def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    Initialize weights for convolutional layers using normal distribution.
    Only affects modules whose class name contains 'Conv'.
    """
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Compute padding size to keep output length same as input length.
    """
    return int((kernel_size * dilation - dilation) / 2)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor
) -> torch.Tensor:
    """
    Fused activation combining tanh and sigmoid functions, used in gated layers.
    Assumes input_a and input_b are shape [B, 2C, T] and n_channels is [1] or scalar tensor.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


def convert_pad_shape(pad_shape : list[list[int]]) -> list[int]:
    """
    Convert nested padding shape list [[x1, x2], [y1, y2], ...] into flat list for F.pad.
    E.g., [[0, 1], [2, 3]] -> [2, 3, 0, 1] (reverse + flatten)
    """
    reversed_shape = pad_shape[::-1]
    return [item for sublist in reversed_shape for item in sublist]


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """
    Generate a boolean mask for each sequence in the batch.
    True where index < sequence length.
    """
    if max_length is None:
        max_length = length.max().item()
    pos = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return pos.unsqueeze(0) < length.unsqueeze(1)

