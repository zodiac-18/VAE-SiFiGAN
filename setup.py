# -*- coding: utf-8 -*-

# Copyright 2026 Kenichi Ogita (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Setup VAE-SiFiGAN inference library."""

from setuptools import find_packages, setup

setup(
    name="vaesifigan",
    version="1.0.0",
    url="https://github.com/zodiac-18/VAE-SiFiGAN",
    author="Kenichi Ogita",
    author_email="ogita.kenichi@g.sp.m.is.nagoya-u.ac.jp",
    description="Inference code for VAE-SiFiGAN, an F0-controllable neural vocoder",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT License",
    packages=find_packages(include=["vaesifigan*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.20.0",
        "librosa>=0.9.0",
        "pyworld>=0.3.0",
        "soundfile>=0.10.2",
        "scipy>=1.6.0",
    ],
    entry_points={
        "console_scripts": [
            "vaesifigan-decode=vaesifigan.bin.decode:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
