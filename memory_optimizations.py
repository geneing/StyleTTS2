# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import os.path as osp
import glob
import re
import sys
import json
import yaml
import shutil
import numpy as np
import torch
from socket import gethostname
import warnings
warnings.simplefilter('ignore')

# load packages
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import *
from meldataset import build_dataloader
from utils import *
from losses import *
from optimizers import build_optimizer

import pylab as plt

# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = "cuda"

torch.cuda.memory._record_memory_history(
       max_entries=1000000, enabled='all', stacks="python"
   )

# %%
config_path = 'Configs/config.yml' #'Models/LibriTTS/config.yml' #'Configs/config_libritts_espeak.yml'

config = yaml.safe_load(open(config_path))
sr = config['preprocess_params'].get('sr', 24000)

log_dir = config['log_dir']

torch.cuda.memory._record_memory_history(max_entries=1000000, stacks='python')
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

# %%
config['model_params']['skip_downsamples'] = False
model_params = recursive_munch(config['model_params'])
multispeaker = model_params.multispeaker
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].to(device) for key in model]

# %%
# sr = config['preprocess_params'].get('sr', 24000)
# # wrapped losses for compatibility with mixed precision
# stft_loss = MultiResolutionSTFTLoss().to(device)
# gl = GeneratorLoss(model.mpd, model.msd).to(device)
# dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
# wl = WavLMLoss(model_params.slm.model, 
#                    model.wd, 
#                    sr, 
#                    model_params.slm.sr).to(device)

# %%
scheduler_params = {
    "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
    "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
    "epochs": 1, #epochs,
    "steps_per_epoch": 1, # batch_manager.get_step_count(),
}

optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                  scheduler_params_dict= {key: scheduler_params.copy() for key in model},
                               lr=float(config['optimizer_params'].get('lr', 1e-4)))

pretrained_model = "Models/LJSpeech_original/epoch_2nd_00100.pth" #"Models/LibriTTS/epochs_2nd_00020.pth"
model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, pretrained_model,
                                        load_only_params=True)
stft_loss = MultiResolutionSTFTLoss().to(device)

gl = GeneratorLoss(model.mpd, model.msd).to(device)

dl = DiscriminatorLoss(model.mpd, model.msd).to(device)

wl = WavLMLoss(model_params.slm.model, 
                model.wd, 
                sr, 
                model_params.slm.sr).to(device)

_ = [model[key].train() for key in model]

# %%
max_len = 400
root_path = "../"    
train_list, val_list = get_data_path_list('Data/train_list_LibriTTSR_espeak.txt.verysmall.txt', 'Data/train_list_LibriTTSR_espeak.txt.verysmall.txt')
train_dataloader = build_dataloader(train_list,
                                      root_path,
                                      OOD_data='Data/OOD_texts_LibriTTSR_espeak.txt.verysmall.txt',
                                      # min_length=min_length,
                                      batch_size=4,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})

batch = next(train_dataloader.__iter__())
waves = batch[0]
batch = [b.to(device) for b in batch[1:]]
texts, input_lengths, _, _, mels, mel_input_length, _ = batch



# %%
n_down = model.text_aligner.n_down
mask = (length_to_mask(mel_input_length // (2 ** n_down))).to(device)
text_mask = length_to_mask(input_lengths).to(device)

text_aligner = model.text_aligner.to(device)
ppgs, s2s_pred, s2s_attn = text_aligner(mels, mask, texts)
s2s_attn = s2s_attn.transpose(-1, -2)
s2s_attn = s2s_attn[..., 1:]
s2s_attn = s2s_attn.transpose(-1, -2)
with torch.no_grad():
    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
    attn_mask = (attn_mask < 1)
s2s_attn.masked_fill_(attn_mask, 0.0)

with torch.no_grad():
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

text_encoder = model.text_encoder.to(device)
t_en = text_encoder(texts, input_lengths, text_mask)
asr = (t_en @ s2s_attn_mono)

# get clips
mel_input_length_all = (mel_input_length) # for balanced load
mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), max_len // 2])
mel_len_st = int(mel_input_length.min().item() / 2 - 1)

en = []
gt = []
wav = []
st = []
for bib in range(len(mel_input_length)):
    mel_length = int(mel_input_length[bib].item() / 2)

    random_start = np.random.randint(0, mel_length - mel_len)
    en.append(asr[bib, :, random_start:random_start+mel_len])
    gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

    y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
    wav.append(torch.from_numpy(y).to(device))
    
    # style reference (better to be different from the GT)
    random_start = np.random.randint(0, mel_length - mel_len_st)
    st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])


en = torch.stack(en)
gt = torch.stack(gt).detach()
st = torch.stack(st).detach()

wav = torch.stack(wav).float().detach()

pitch_extractor = model.pitch_extractor.to(device)
with torch.no_grad():    
    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
    F0_real, _, _ = pitch_extractor(gt.unsqueeze(1))

style_encoder = model.style_encoder.to(device)
s = style_encoder(st.unsqueeze(1)) # if multispeaker else gt.unsqueeze(1))

y_rec = model.decoder(en, F0_real, real_norm, s)



# %%
# optimizer.zero_grad()   
# d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
# d_loss.backward()

# %%

optimizer.zero_grad()   
d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach())
d_loss.backward()
torch.cuda.memory._dump_snapshot(f"memory.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# %%
# d=dl(wav.detach().unsqueeze(1).float(), y_rec.detach())


# # %%
# plt.plot(wav.detach().cpu().numpy()[0,:])
# # plt.plot(y_rec.detach().cpu().numpy()[0,0,:])
# # for name, param in model.text_aligner.to(device).named_parameters():
# #     print(name, param.device)

# # %%
# plt.plot(F0_real.cpu()[0,:])

# # %%



