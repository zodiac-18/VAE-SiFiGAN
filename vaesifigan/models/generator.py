# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""SiFi-GAN generator modules.

References:
    - https://github.com/chomeyama/SiFiGAN
    - https://github.com/jik876/hifi-gan

"""

import torch.nn as nn

from vaesifigan.layers import AdaptiveResidualBlock, ResidualBlock


class SiFiGANGenerator(nn.Module):
    """SiFi-GAN generator module."""

    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(5, 4, 3, 2),
        upsample_kernel_sizes=(10, 8, 6, 4),
        source_network_params={
            "resblock_kernel_size": 3,
            "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
            "use_additional_convs": True,
        },
        filter_network_params={
            "resblock_kernel_sizes": (3, 5, 7),
            "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            "use_additional_convs": False,
        },
        share_upsamples=False,
        share_downsamples=False,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize SiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layers.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            source_network_params (dict): Parameters for the source-network.
            filter_network_params (dict): Parameters for the filter-network.
            share_upsamples (bool): Whether to share up-sampling transposed CNNs.
            share_downsamples (bool): Whether to share down-sampling CNNs.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)

        self.num_upsamples = len(upsample_kernel_sizes)
        self.filter_network_params = filter_network_params
        self.share_upsamples = share_upsamples
        self.share_downsamples = share_downsamples
        self.sn = nn.ModuleDict()
        self.fn = nn.ModuleDict()
        self.input_conv = nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        self.sn["upsamples"] = nn.ModuleList()
        self.fn["upsamples"] = nn.ModuleList()
        self.sn["blocks"] = nn.ModuleList()
        self.fn["blocks"] = nn.ModuleList()
        for i in range(self.num_upsamples):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.sn["upsamples"] += [
                self._upsample_layer(
                    channels // (2**i),
                    channels // (2 ** (i + 1)),
                    upsample_kernel_sizes[i],
                    upsample_scales[i],
                    bias,
                    nonlinear_activation,
                    nonlinear_activation_params,
                )
            ]
            if not share_upsamples:
                self.fn["upsamples"] += [
                    self._upsample_layer(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        bias,
                        nonlinear_activation,
                        nonlinear_activation_params,
                    )
                ]
            self.sn["blocks"] += [
                AdaptiveResidualBlock(
                    kernel_size=source_network_params["resblock_kernel_size"],
                    channels=channels // (2 ** (i + 1)),
                    dilations=source_network_params["resblock_dilations"][i],
                    bias=bias,
                    use_additional_convs=source_network_params["use_additional_convs"],
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
            for j in range(len(filter_network_params["resblock_kernel_sizes"])):
                self.fn["blocks"] += [
                    ResidualBlock(
                        kernel_size=filter_network_params["resblock_kernel_sizes"][j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=filter_network_params["resblock_dilations"][j],
                        bias=bias,
                        use_additional_convs=filter_network_params[
                            "use_additional_convs"
                        ],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]

        out_conv_channels = channels // (2**self.num_upsamples)
        self.sn["output_conv"] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                out_conv_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.fn["output_conv"] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                out_conv_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Tanh(),
        )

        # sine embedding layer
        self.sn["emb"] = nn.Conv1d(
            1,
            out_conv_channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )

        # down-sampling CNNs
        self.sn["downsamples"] = nn.ModuleList()
        for i in reversed(range(self.num_upsamples)):
            self.sn["downsamples"] += [
                self._downsample_layer(
                    channels // (2 ** (i + 1)),
                    channels // (2**i),
                    upsample_kernel_sizes[i],
                    upsample_scales[i],
                    bias,
                    nonlinear_activation,
                    nonlinear_activation_params,
                )
            ]
        if not share_downsamples:
            self.fn["downsamples"] = nn.ModuleList()
            for i in reversed(range(self.num_upsamples)):
                self.fn["downsamples"] += [
                    self._downsample_layer(
                        channels // (2 ** (i + 1)),
                        channels // (2**i),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        bias,
                        nonlinear_activation,
                        nonlinear_activation_params,
                    )
                ]

    @staticmethod
    def _upsample_layer(
        in_channels,
        out_channels,
        kernel_size,
        scale,
        bias,
        nonlinear_activation,
        nonlinear_activation_params,
    ):
        return nn.Sequential(
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                scale,
                padding=scale // 2 + scale % 2,
                output_padding=scale % 2,
                bias=bias,
            ),
        )

    @staticmethod
    def _downsample_layer(
        in_channels,
        out_channels,
        kernel_size,
        scale,
        bias,
        nonlinear_activation,
        nonlinear_activation_params,
    ):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                scale,
                padding=scale - (kernel_size % 2 == 0),
                bias=bias,
            ),
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
        )

    def forward(self, x, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input sine signal (B, 1, T).
            c (Tensor): Input latent representation (B, in_channels, T // hop_size).
            d (List): F0-dependent dilation factors [(B, 1, T) x num_upsamples].

        Returns:
            Tensor: Output waveform (B, out_channels, T).
            Tensor: Output source excitation signal (B, out_channels, T).

        """
        c = self.input_conv(c)
        e = c

        # source-network forward
        x = self.sn["emb"](x.float())
        embs = [x]
        for i in range(self.num_upsamples - 1):
            x = self.sn["downsamples"][i](x)
            embs += [x]
        for i in range(self.num_upsamples):
            e = self.sn["upsamples"][i](e) + embs[-i - 1]
            e = self.sn["blocks"][i](e, d[i])
        source = self.sn["output_conv"](e)

        # filter-network forward
        embs = [e]
        for i in range(self.num_upsamples - 1):
            if self.share_downsamples:
                e = self.sn["downsamples"][i](e)
            else:
                e = self.fn["downsamples"][i](e)
            embs += [e]
        num_blocks = len(self.filter_network_params["resblock_kernel_sizes"])
        for i in range(self.num_upsamples):
            if self.share_upsamples:
                c = self.sn["upsamples"][i](c) + embs[-i - 1]
            else:
                c = self.fn["upsamples"][i](c) + embs[-i - 1]
            cs = 0.0
            for j in range(num_blocks):
                cs += self.fn["blocks"][i * num_blocks + j](c)
            c = cs / num_blocks
        c = self.fn["output_conv"](c)

        return c, source
