# Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

### Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon

In our recent [paper](https://arxiv.org/abs/2005.11129), we propose Glow-TTS: a generative flow for text-to-speech via monotonic alignment search.

Recently, text-to-speech (TTS) models such as FastSpeech and ParaNet have been proposed to generate mel-spectrograms from text in parallel. Despite the advantages, the parallel TTS models cannot be trained without guidance from autoregressive TTS models as their external aligners. In this work, we propose Glow-TTS, a flow-based generative model for parallel TTS that does not require any external aligner. We introduce Monotonic Alignment Search (MAS), an internal alignment search algorithm for training Glow-TTS. By leveraging the properties of flows, MAS searches for the most probable monotonic alignment between text and the latent representation of speech. Glow-TTS obtains an order-of-magnitude speed-up over the autoregressive TTS model, Tacotron 2, at synthesis with comparable speech quality, requiring only 1.5 seconds to synthesize one minute of speech in end-to-end. We further show that our model can be easily extended to a multi-speaker setting.

Visit our [demo](https://jaywalnut310.github.io/glow-tts-demo/index.html) for audio samples.

<table style="width:100%">
  <tr>
    <th>Glow-TTS at training</th>
    <th>Glow-TTS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="Glow-TTS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="Glow-TTS at inference" height="400"></td>
  </tr>
</table>

## 1. Environments we use

* Python3.6.9
* pytorch1.2.0
* cython0.29.12
* librosa0.7.1
* numpy1.16.4
* scipy1.3.0

For Mixed-precision training, we use [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4


## 2. Pre-requisites

a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`


b) Initialize WaveGlow submodule: `git submodule init; git submodule update`

Don't forget to download pretrained WaveGlow model and place it into the waveglow folder.

c) Build Monotonic Alignment Search Code (Cython): `cd monotonic_align; python setup.py build_ext --inplace`


## 3. Training Example


```sh
sh train_ddi.sh configs/base.json base
```

## 4. Inference Example

See [inference.ipynb](./inference.ipynb)
