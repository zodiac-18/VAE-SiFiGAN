# -*- coding: utf-8 -*-

# Copyright 2026 Kenichi Ogita (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""WaveNet modules.

References:
    - https://github.com/jaywalnut310/vits
    - https://github.com/r9y9/wavenet_vocoder

"""

import torch
import torch.nn as nn


class WaveNet(nn.Module):
    """Non-causal WaveNet module."""

    def __init__(self, channels, kernel_size, dilation_rate, num_layers):
        """Initialize WaveNet module.

        Args:
            channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of dilated convolution layers.
            dilation_rate (int): Base of the exponentially growing dilation factors.
            num_layers (int): Number of dilated convolution layers.

        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        self.channels = channels
        self.num_layers = num_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_rate**i
            self.in_layers += [
                nn.Conv1d(
                    channels,
                    2 * channels,
                    kernel_size,
                    dilation=dilation,
                    padding=(kernel_size * dilation - dilation) // 2,
                )
            ]
            # the last layer has no residual connection
            res_skip_channels = 2 * channels if i < num_layers - 1 else channels
            self.res_skip_layers += [nn.Conv1d(channels, res_skip_channels, 1)]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Sum of the skip connection outputs (B, channels, T).

        """
        output = torch.zeros_like(x)
        for i in range(self.num_layers):
            acts = self.in_layers[i](x)
            acts = torch.tanh(acts[:, : self.channels]) * torch.sigmoid(
                acts[:, self.channels :]
            )
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                x = x + res_skip_acts[:, : self.channels]
                output = output + res_skip_acts[:, self.channels :]
            else:
                output = output + res_skip_acts
        return output
