# -*- coding: utf-8 -*-

# Copyright 2026 Kenichi Ogita (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding script for VAE-SiFiGAN.

References:
    - https://github.com/chomeyama/SiFiGAN

"""

import argparse
import logging
import time
from pathlib import Path

import soundfile as sf
import torch

from vaesifigan.models import VAESiFiGAN


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Resynthesize wav files with VAE-SiFiGAN."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to the pre-trained model."
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="directory containing input wav files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="directory to save generated wav files.",
    )
    parser.add_argument(
        "--f0-factors",
        type=float,
        nargs="+",
        default=[1.0],
        help="F0 scaling factors to synthesize with.",
    )
    parser.add_argument(
        "--f0-floor",
        type=float,
        default=50.0,
        help="lower bound of F0 search range in Hz.",
    )
    parser.add_argument(
        "--f0-ceil",
        type=float,
        default=1000.0,
        help="upper bound of F0 search range in Hz.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run the model on.",
    )
    parser.add_argument("--seed", type=int, default=100, help="random seed.")
    return parser


def main():
    """Run decoding process."""
    args = get_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s) %(message)s"
    )
    torch.manual_seed(args.seed)

    wav_files = sorted(Path(args.in_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav file was found in {args.in_dir}.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VAESiFiGAN.from_pretrained(args.checkpoint, device=args.device)
    logging.info(f"Loaded {args.checkpoint} on {args.device}.")

    total_rtf = 0.0
    for wav_file in wav_files:
        for f0_factor in args.f0_factors:
            start = time.time()
            wav = model.resynthesize(
                wav_file,
                f0_factor=f0_factor,
                f0_floor=args.f0_floor,
                f0_ceil=args.f0_ceil,
            )
            rtf = (time.time() - start) / (len(wav) / model.sample_rate)
            total_rtf += rtf

            out_file = out_dir / f"{wav_file.stem}_f{f0_factor:.2f}.wav"
            sf.write(out_file, wav, model.sample_rate, "PCM_16")
            logging.info(f"{out_file} (RTF = {rtf:.3f})")

    num_generated = len(wav_files) * len(args.f0_factors)
    logging.info(
        f"Finished generation of {num_generated} utterances "
        f"(average RTF = {total_rtf / num_generated:.3f})."
    )


if __name__ == "__main__":
    main()
