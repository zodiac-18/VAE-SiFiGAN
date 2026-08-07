# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature-related functions.

References:
    - https://github.com/chomeyama/SiFiGAN
    - https://github.com/bigpon/QPPWG

"""

import librosa
import numpy as np
import pyworld
import torch
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter
from torch.nn.functional import interpolate

NUM_TAPS = 255


def low_cut_filter(x, sample_rate, cutoff=70):
    """Apply a high-pass FIR filter that removes low frequency noise.

    Args:
        x (ndarray): Waveform sequence (T,).
        sample_rate (int): Sampling rate.
        cutoff (float): Cut-off frequency in Hz.

    Returns:
        ndarray: Filtered waveform sequence (T,).

    """
    fil = firwin(NUM_TAPS, cutoff / (sample_rate // 2), pass_zero=False)
    return lfilter(fil, 1, x)


def low_pass_filter(x, frame_rate, cutoff):
    """Apply a low-pass FIR filter that smooths a frame-level sequence.

    Args:
        x (ndarray): Frame-level sequence (T,).
        frame_rate (int): Number of frames per second.
        cutoff (float): Cut-off frequency in Hz.

    Returns:
        ndarray: Filtered sequence (T,).

    """
    fil = firwin(NUM_TAPS, cutoff / (frame_rate // 2))
    x_pad = np.pad(x, (NUM_TAPS, NUM_TAPS), "edge")
    return lfilter(fil, 1, x_pad)[NUM_TAPS + NUM_TAPS // 2 : -NUM_TAPS // 2]


def to_continuous_f0(f0):
    """Linearly interpolate the unvoiced regions of an F0 sequence.

    Args:
        f0 (ndarray): F0 sequence with unvoiced frames set to zero (T,).

    Returns:
        ndarray: Continuous F0 sequence (T,).

    """
    if (f0 == 0).all():
        return f0.copy()

    # pad the head and the tail with the first and the last voiced values
    voiced = np.where(f0 != 0)[0]
    cf0 = f0.copy()
    cf0[: voiced[0]] = f0[voiced[0]]
    cf0[voiced[-1] :] = f0[voiced[-1]]

    nonzero = np.where(cf0 != 0)[0]
    return interp1d(nonzero, cf0[nonzero])(np.arange(cf0.shape[0]))


def smooth_continuous_f0(cf0, frame_rate, cutoff=20):
    """Smooth a continuous F0 sequence while keeping it non-negative.

    Args:
        cf0 (ndarray): Continuous F0 sequence (T,).
        frame_rate (int): Number of frames per second.
        cutoff (float): Initial cut-off frequency in Hz.

    Returns:
        ndarray: Smoothed continuous F0 sequence (T,).

    """
    nyquist = frame_rate // 2
    smoothed = low_pass_filter(cf0, frame_rate, cutoff)
    # relax the cut-off until the contour does not undershoot below zero
    cutoff = 70
    while not (smoothed >= 0).all() and cutoff < nyquist:
        smoothed = low_pass_filter(cf0, frame_rate, cutoff)
        cutoff *= 2
    return smoothed


def log_mel_spectrogram(
    audio,
    sample_rate,
    fft_size=1024,
    hop_size=120,
    win_length=1024,
    window="hann",
    num_mels=80,
    fmin=0,
    fmax=None,
):
    """Extract a log mel-spectrogram in decibels.

    Args:
        audio (ndarray): Waveform sequence (T,).
        sample_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency of the mel basis.
        fmax (int): Maximum frequency of the mel basis. Defaults to the Nyquist rate.

    Returns:
        ndarray: Log mel-spectrogram in dB (#frames, num_mels).

    """
    spc = np.abs(
        librosa.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            pad_mode="reflect",
        )
    ).T
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=sample_rate / 2 if fmax is None else fmax,
    )
    return 20 * np.log10(np.clip(np.dot(spc, mel_basis.T), 1e-7, None))


def dilated_factor(f0, sample_rate, dense_factor):
    """Calculate pitch-dependent dilation factors.

    Args:
        f0 (ndarray): F0 sequence (T, 1).
        sample_rate (int): Sampling rate.
        dense_factor (int): Number of taps in one F0 cycle.

    Returns:
        ndarray: Pitch-dependent dilation factors (T, 1).

    """
    f0 = np.where(f0 == 0, sample_rate / dense_factor, f0)
    return sample_rate / dense_factor / f0


def sine_excitation(f0, sample_rate, hop_size, sine_amp=0.1, noise_amp=0.003):
    """Generate a sine wave that drives the source-network.

    Args:
        f0 (Tensor): F0 sequence (B, 1, T).
        sample_rate (int): Sampling rate.
        hop_size (int): Hop size of the F0 sequence.
        sine_amp (float): Amplitude of the sine wave.
        noise_amp (float): Amplitude of the additive Gaussian noise.

    Returns:
        Tensor: Sine excitation signal (B, 1, T * hop_size).

    """
    batch_size, _, num_frames = f0.size()
    length = num_frames * hop_size
    vuv = interpolate((f0 > 0) * torch.ones_like(f0), length)
    radians = (interpolate(f0.to(torch.float64), length) / sample_rate) % 1
    sine = vuv * torch.sin(torch.cumsum(radians, dim=2) * 2 * np.pi) * sine_amp
    if noise_amp > 0:
        amplitude = vuv * noise_amp + (1.0 - vuv) * noise_amp / 3.0
        sine = sine + torch.randn((batch_size, 1, length), device=f0.device) * amplitude
    return sine


class FeatureExtractor:
    """Feature extractor module for VAE-SiFiGAN."""

    def __init__(
        self,
        sample_rate=24000,
        hop_size=120,
        fft_size=1024,
        win_length=1024,
        window="hann",
        num_mels=80,
        fmin=0,
        fmax=None,
        highpass_cutoff=70,
    ):
        """Initialize FeatureExtractor.

        Args:
            sample_rate (int): Sampling rate the model operates at.
            hop_size (int): Hop size in samples.
            fft_size (int): FFT size.
            win_length (int): Window length.
            window (str): Window function type.
            num_mels (int): Number of mel basis.
            fmin (int): Minimum frequency of the mel basis.
            fmax (int): Maximum frequency of the mel basis.
            highpass_cutoff (float): Cut-off frequency of the low-cut filter in Hz.
                Set to 0 to disable the filter.

        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.win_length = win_length
        self.window = window
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.highpass_cutoff = highpass_cutoff

    def __call__(self, audio, sample_rate, f0_floor=50.0, f0_ceil=1000.0):
        """Extract features from a waveform.

        Args:
            audio (ndarray): Waveform sequence (T,) or (T, #channels).
            sample_rate (int): Sampling rate of the waveform.
            f0_floor (float): Lower bound of the F0 search range in Hz.
            f0_ceil (float): Upper bound of the F0 search range in Hz.

        Returns:
            ndarray: Log mel-spectrogram in dB (#frames, num_mels).
            ndarray: Continuous F0 sequence in Hz (#frames, 1).

        """
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.sample_rate
            )
        if self.highpass_cutoff > 0:
            audio = low_cut_filter(audio, self.sample_rate, self.highpass_cutoff)

        f0, _ = pyworld.harvest(
            audio,
            fs=self.sample_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=1000 * self.hop_size / self.sample_rate,
        )
        cf0 = to_continuous_f0(f0)
        if not (f0 == 0).all():
            cf0 = smooth_continuous_f0(cf0, self.sample_rate // self.hop_size)

        logmsp = log_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            num_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        num_frames = min(len(cf0), len(logmsp))
        return logmsp[:num_frames], cf0[:num_frames, np.newaxis]
