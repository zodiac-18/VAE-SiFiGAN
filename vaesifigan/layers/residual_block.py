# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual block modules of the SiFi-GAN generator.

References:
    - https://github.com/chomeyama/SiFiGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/jik876/hifi-gan

"""

import torch
import torch.nn as nn


def pd_indexing(x, d, dilation):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Pitch-dependent dilation factors (B, 1, T).
        dilation (int): Base dilation size.

    Returns:
        Tensor: Past samples (B, C, T).
        Tensor: Future samples (B, C, T).

    """
    B, C, T = x.size()
    batch_index = torch.arange(B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)
    idx_base = torch.arange(T, dtype=torch.long, device=x.device).reshape(1, 1, T)

    # Indices out of the sequence are reflected back into it.
    idx_past = (idx_base - dilations).abs() % T
    idx_future = idx_base + dilations
    overflowed = idx_future >= T
    idx_future[overflowed] = -(idx_future[overflowed] % T)

    return x[batch_index, ch_index, idx_past], x[batch_index, ch_index, idx_future]


class ResidualBlock(nn.Module):
    """Residual block module in HiFi-GAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilated convolution layers.
            channels (int): Number of channels for convolution layers.
            dilations (List[int]): List of dilation factors.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_additional_convs (bool): Whether to use additional convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1 += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    nn.Sequential(
                        getattr(nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x


class AdaptiveResidualBlock(nn.Module):
    """Residual block module with pitch-dependent dilated convolutions."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 2, 4),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize AdaptiveResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilated convolution layers.
            channels (int): Number of channels for convolution layers.
            dilations (List[int]): List of dilation factors.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_additional_convs (bool): Whether to use additional convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        assert kernel_size == 3, "Currently only kernel_size = 3 is supported."
        self.use_additional_convs = use_additional_convs
        self.channels = channels
        self.dilations = dilations
        self.nonlinears = nn.ModuleList()
        self.convsC = nn.ModuleList()
        self.convsP = nn.ModuleList()
        self.convsF = nn.ModuleList()
        if use_additional_convs:
            self.convsA = nn.ModuleList()
        for _ in dilations:
            self.nonlinears += [
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            self.convsC += [nn.Conv1d(channels, channels, 1, bias=bias)]
            self.convsP += [nn.Conv1d(channels, channels, 1, bias=bias)]
            self.convsF += [nn.Conv1d(channels, channels, 1, bias=bias)]
            if use_additional_convs:
                self.convsA += [
                    nn.Sequential(
                        getattr(nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            d (Tensor): Pitch-dependent dilation factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for i, dilation in enumerate(self.dilations):
            xt = self.nonlinears[i](x)
            xP, xF = pd_indexing(xt, d, dilation)
            xt = self.convsC[i](xt) + self.convsP[i](xP) + self.convsF[i](xF)
            if self.use_additional_convs:
                xt = self.convsA[i](xt)
            x = xt + x
        return x
