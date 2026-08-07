# -*- coding: utf-8 -*-

# Copyright 2026 Kenichi Ogita (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""VAE-SiFiGAN modules.

References:
    - https://github.com/chomeyama/SiFiGAN
    - https://github.com/jaywalnut310/vits

"""

import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from vaesifigan.layers import WaveNet
from vaesifigan.models.generator import SiFiGANGenerator
from vaesifigan.utils import FeatureExtractor, dilated_factor, sine_excitation


class PosteriorEncoder(nn.Module):
    """VAE posterior encoder module."""

    def __init__(
        self,
        in_channels=80,
        hidden_channels=192,
        out_channels=30,
        num_layers=16,
        kernel_size=7,
        dilation_rate=1,
    ):
        """Initialize PosteriorEncoder module.

        Args:
            in_channels (int): Number of mel-spectrogram bins.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of latent channels.
            num_layers (int): Number of WaveNet layers.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Base of the WaveNet dilation factors.

        """
        super().__init__()
        self.out_channels = out_channels
        self.preprocess = nn.Conv1d(in_channels, hidden_channels, 1)
        self.wavenet = WaveNet(hidden_channels, kernel_size, dilation_rate, num_layers)
        self.projection = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, mel):
        """Calculate forward propagation.

        Args:
            mel (Tensor): Normalized log mel-spectrogram (B, in_channels, T).

        Returns:
            Tensor: Mean of the latent distribution (B, out_channels, T).
            Tensor: Log scale of the latent distribution (B, out_channels, T).

        """
        x = self.wavenet(self.preprocess(mel))
        return torch.split(self.projection(x), self.out_channels, dim=1)


class VAESiFiGAN(nn.Module):
    """VAE-SiFiGAN vocoder module."""

    def __init__(self, encoder, decoder, features, excitation):
        """Initialize VAE-SiFiGAN module.

        Args:
            encoder (dict): Keyword arguments of PosteriorEncoder.
            decoder (dict): Keyword arguments of SiFiGANGenerator.
            features (dict): Keyword arguments of FeatureExtractor.
            excitation (dict): Parameters of the sine excitation signal, namely
                dense_factors, sine_amp and noise_amp.

        """
        super().__init__()
        self.encoder = PosteriorEncoder(**encoder)
        self.decoder = SiFiGANGenerator(**decoder)
        self.feature_extractor = FeatureExtractor(**features)
        self.excitation = dict(excitation)
        self.upsample_scales = list(decoder["upsample_scales"])

        # Statistics of the training data, overwritten by the loaded state dict.
        num_mels = encoder["in_channels"]
        self.register_buffer("logmsp_mean", torch.zeros(num_mels, dtype=torch.float64))
        self.register_buffer("logmsp_scale", torch.ones(num_mels, dtype=torch.float64))

    @classmethod
    def from_pretrained(cls, checkpoint_path, device="cpu"):
        """Build a model from a pre-trained checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint exported for inference.
            device (str): Device to place the model on.

        Returns:
            VAE-SiFiGAN: Model in evaluation mode.

        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
        return model.eval().to(device)

    @property
    def sample_rate(self):
        """int: Sampling rate the model operates at."""
        return self.feature_extractor.sample_rate

    @torch.no_grad()
    def resynthesize(
        self, audio, sample_rate=None, f0_factor=1.0, f0_floor=50.0, f0_ceil=1000.0
    ):
        """Analyze a waveform and resynthesize it with a scaled F0.

        Args:
            audio (str, Path or ndarray): Path to a wav file, or a waveform (T,).
            sample_rate (int): Sampling rate of the waveform. Required only when
                a waveform is given directly.
            f0_factor (float): Scaling factor applied to the extracted F0.
            f0_floor (float): Lower bound of the F0 search range in Hz.
            f0_ceil (float): Upper bound of the F0 search range in Hz.

        Returns:
            ndarray: Generated waveform (T,) at the model sampling rate.

        """
        if isinstance(audio, (str, os.PathLike)):
            audio, sample_rate = sf.read(audio)
        elif sample_rate is None:
            raise ValueError("sample_rate is required when a waveform is given.")
        logmsp, cf0 = self.feature_extractor(audio, sample_rate, f0_floor, f0_ceil)
        return self.synthesize(logmsp, cf0, f0_factor=f0_factor)

    @torch.no_grad()
    def synthesize(self, logmsp, cf0, f0_factor=1.0):
        """Generate a waveform from a log mel-spectrogram and a continuous F0.

        Args:
            logmsp (ndarray): Log mel-spectrogram in dB (T, num_mels).
            cf0 (ndarray): Continuous F0 sequence in Hz (T, 1).
            f0_factor (float): Scaling factor applied to the F0.

        Returns:
            ndarray: Generated waveform (T * hop_size,).

        """
        device = self.logmsp_mean.device
        sample_rate = self.feature_extractor.sample_rate
        hop_size = self.feature_extractor.hop_size
        cf0 = np.asarray(cf0, dtype=np.float64).reshape(-1, 1) * f0_factor

        # PitchSs-dependent dilation factors for each up-sampling stage
        dilation_factors = [
            torch.from_numpy(
                np.repeat(dilated_factor(cf0, sample_rate, dense_factor), scale)
            )
            .float()
            .view(1, 1, -1)
            .to(device)
            for dense_factor, scale in zip(
                self.excitation["dense_factors"], np.cumprod(self.upsample_scales)
            )
        ]

        cf0 = torch.from_numpy(cf0).float().view(1, 1, -1).to(device)
        source = sine_excitation(
            cf0,
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=self.excitation["sine_amp"],
            noise_amp=self.excitation["noise_amp"],
        )

        mel = torch.from_numpy(np.asarray(logmsp, dtype=np.float64)).to(device)
        mel = ((mel - self.logmsp_mean) / self.logmsp_scale).float()
        mel = mel.transpose(1, 0).unsqueeze(0)

        mean, logs = self.encoder(mel)
        z = mean + torch.randn_like(mean) * torch.exp(logs)
        waveform, _ = self.decoder(source, z, dilation_factors)

        return waveform.view(-1).cpu().numpy()
