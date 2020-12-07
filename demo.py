import os
os.system('git submodule init; git submodule update')
os.system('cd monotonic_align; python setup.py build_ext --inplace; cd ..')
os.system('wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/3/zip -O waveglow_ljs_256channels_3.zip')
os.system('mkdir waveglow/')
os.system('unzip waveglow_ljs_256channels_3.zip -d waveglow/')
os.system('mkdir pretrained/')
os.system('cd pretrained; gdown --id 1JiCMBVTG4BMREK8cT3MYck1MgYvwASL0; cd ..')

PRETRAINED_GLOW_TTS_PATH = "./pretrained/pretrained.pth"
WAVEGLOW_PATH = './waveglow/waveglow_256channels_ljs_v3.pt' # or change to the latest version of the pretrained WaveGlow.

import matplotlib.pyplot as plt
import IPython.display as ipd

import sys
sys.path.append('./waveglow/')

import librosa
import numpy as np
import glob
import json

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils
import gradio as gr

# load WaveGlow
waveglow_path = WAVEGLOW_PATH
waveglow = torch.load(waveglow_path, map_location=torch.device('cpu'))['model']
waveglow = waveglow.remove_weightnorm(waveglow)
_ = waveglow.eval()
# from apex import amp
# waveglow, _ = amp.initialize(waveglow, [], opt_level="O3") # Try if you want to boost up synthesis speed.

# If you are using a provided pretrained model
hps = utils.get_hparams_from_file("./configs/base.json")
checkpoint_path = PRETRAINED_GLOW_TTS_PATH

model = models.FlowGenerator(
    len(symbols) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model)

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

# normalizing & type casting
def normalize_audio(x, max_wav_value=hps.data.max_wav_value):
    return np.clip((x / np.abs(x).max()) * max_wav_value, -32768, 32767).astype("int16")
    
def predict(tst_stn):
  if getattr(hps.data, "add_blank", False):
      text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
      text_norm = commons.intersperse(text_norm, len(symbols))
  else: # If not using "add_blank" option during training, adding spaces at the beginning and the end of utterance improves quality
      tst_stn = " " + tst_stn.strip() + " "
      text_norm = text_to_sequence(tst_stn.strip(), ['english_cleaners'], cmu_dict)
  sequence = np.array(text_norm)[None, :]
  x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
  x_tst_lengths = torch.tensor([x_tst.shape[1]])

  with torch.no_grad():
    noise_scale = .667
    length_scale = 1.0
    (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
    audio = waveglow.infer(y_gen_tst, sigma=.666)
    audio = normalize_audio(audio[0].clamp(-1,1).data.cpu().float().numpy())

    return hps.data.sampling_rate, np.stack([audio, audio], 1)
    
gr.Interface(predict, "text", "audio", 
  examples=[["It's so easy to run a text-to-speech model!"], ["Joseph Biden was elected President of the United States."], ["How're you liking the conference so far?"]], title="Glow-TTS: A Generative Flow for Text-to-Speech", description="Try your own predictions with Jaehyeon Kim et al.'s model to be presented at NeurIPS 2020. (A typical sentence will take 10-20 seconds to run on CPU).").launch(share=True, debug=True)
