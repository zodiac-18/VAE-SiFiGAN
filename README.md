# VAE-SiFiGAN

This repo provides the inference code and a pretrained model of VAE-SiFiGAN.
For more information, please see our papers cited below.

## Environment setup

```bash
$ git clone https://github.com/zodiac-18/VAE-SiFiGAN.git
$ cd VAE-SiFiGAN
$ pip install -e .
```

Python 3.8 or later and PyTorch 1.13 or later are required. The remaining dependencies (`librosa`, `pyworld`, `soundfile`, `scipy` and `numpy`) are installed automatically.

## Folder architecture

- **pretrained**: The folder to place the pretrained model.
- **vaesifigan**: The folder of the source codes.

## Run

### Pretrained model

A pretrained model on 24 kHz speech and singing datasets is available [HERE](https://github.com/zodiac-18/VAE-SiFiGAN/releases/latest). This is the model evaluated in Ogita et al. (EUSIPCO 2026), the second article cited below. We used 535 hours of speech and singing voices from 1491 speakers in nine languages, from which the 9.1% of the clips presumably containing F0 extraction errors were excluded by the data selection described in the article. Download it and place it under `pretrained/`.

```bash
$ curl -L -o pretrained/vaesifigan.pkl \
    https://github.com/zodiac-18/VAE-SiFiGAN/releases/latest/download/vaesifigan.pkl
```

The checkpoint holds the model parameters, the model configuration and the statistics of the training data, so there is nothing else to configure.

### Inference

```bash
# Decode every wav file in the input directory with several F0 scaling factors
$ vaesifigan-decode \
    --checkpoint pretrained/vaesifigan.pkl \
    --in-dir your_own_input_wav_dir \
    --out-dir your_own_output_wav_dir \
    --f0-factors 0.5 1.0 2.0
```

The generated files are named `<input name>_f<F0 factor>.wav`. Run `vaesifigan-decode --help` for the other options such as the F0 search range and the device.

### Python API

```python
import soundfile as sf
from vaesifigan import VAESiFiGAN

model = VAESiFiGAN.from_pretrained("pretrained/vaesifigan.pkl", device="cuda")

# Resynthesize a wav file with the F0 raised by one octave
wav = model.resynthesize("input.wav", f0_factor=2.0)
sf.write("output.wav", wav, model.sample_rate)
```

`resynthesize` also accepts a waveform together with its sampling rate. If you have your own features, `model.synthesize(logmsp, cf0)` generates a waveform from a log mel-spectrogram in dB `(#frames, 80)` and a continuous F0 sequence in Hz `(#frames, 1)`.

### Notes

- Inputs are resampled to 24 kHz internally, so wav files of any sampling rate can be given. The generated waveform is always 24 kHz.
- The F0 is extracted with WORLD Harvest. Narrowing `--f0-floor` and `--f0-ceil` around the range of the target voice makes the extraction more reliable, which matters especially for large F0 scaling factors.

## Citation

If you find the code is helpful, please cite the following articles.

```bibtex
@inproceedings{ogita2025vaesifigan,
    author = {Ogita, Kenichi and Yoneyama, Reo and Huang, Wen-Chin and Toda, Tomoki},
    title = {{VAE-SiFiGAN}: Source-Filter {HiFi-GAN} Based on Variational Autoencoder Representations with Enhanced Pitch Controllability},
    booktitle = {Proc. EUSIPCO},
    pages = {531--535},
    year = {2025},
}

@inproceedings{ogita2026vaesifigan,
    author = {Ogita, Kenichi and Yoneyama, Reo and Huang, Wen-Chin and Toda, Tomoki},
    title = {Evaluating {VAE-SiFiGAN} under Large-Scale Training and Noisy Conditions with Data Selection Using {F0} Extraction Error Estimation},
    booktitle = {Proc. EUSIPCO},
    year = {2026},
}
```

## Authors

Development: Kenichi Ogita @ Nagoya University, Japan<br>
E-mail: `ogita.kenichi@g.sp.m.is.nagoya-u.ac.jp`

Advisors:<br>
Reo Yoneyama @ Nagoya University, Japan<br>
Wen-Chin Huang @ Nagoya University, Japan<br>
Tomoki Toda @ Nagoya University, Japan

The network architecture is based on [SiFi-GAN](https://github.com/chomeyama/SiFiGAN) by Reo Yoneyama, which is distributed under the MIT license.
