import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import re
import yaml
import warnings
import argparse
from munch import Munch
import numpy as np
import scipy.io.wavfile
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars

import spacy
from spacy.lang.en import English

import phonemizer
from pedalboard.io import AudioFile

from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from models import *
from utils import *
from text_utils import TextCleaner

textclenaer = TextCleaner()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

#

def compute_style(path, model):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def load_components(config):
    # load phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    return global_phonemizer, text_aligner, pitch_extractor, plbert

def build_tts_model(device, model_params, text_aligner, pitch_extractor, plbert):
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
    params = params_whole['net']

    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
            # load params
                model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]
    model.sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    return model


def LFinference(model, model_params, tokens, s_prev, ref_s, alpha=0.7, beta=0.7, diffusion_steps=5, embedding_scale=1):
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = model.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)
        
        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = alpha * s_prev + (1 - alpha) * s_pred
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new
        
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def phoneme_tokenize(global_phonemizer, text_words):
    ps = global_phonemizer.phonemize([text_words])
    # ps = ' '.join(ps)
    tokens = textclenaer(ps[0])
    tokens.insert(0, 0)
    return tokens


# Function to split text into tokens, if the text generates more than 512 tokens (downstream network cannot process more than 512 tokens at once)
# We first split the text into pieces by punctuation. 
# If the pieces are still too long, we'll split on spaces into equal size pieces.
# If it's still too long, we truncate the tokens, to avoid the network failing.
class CommaPoint(PunktLanguageVars):
    sent_end_chars = ('.','?','!',',',';', ':', '-')
punkt_tokenizer = PunktSentenceTokenizer(lang_vars = CommaPoint())

def subtokenize(global_phonemizer, s):
    txt = punkt_tokenizer.tokenize(s.strip())  #maybe switch to spacy

    if len(txt)>1:
        token_list=[]
        for t in txt:
            token_list += subtokenize(global_phonemizer, t)
        return token_list
    else:
        token = phoneme_tokenize(global_phonemizer, txt[0])
        if len(token) < 511:
            print('\t\t',txt[0],'\n\n')
            return [token]
        else:
            #last ditch effort - split on spaces somewhere in the middle of the remaining chunk. 
            t = re.split(r"[ ]", txt[0])
            length = len(t) # number of words
            if length == 1:
                token = phoneme_tokenize(global_phonemizer, t[0])
                if len(token) > 511:
                    print("still too long {token}")
                    return [token[:511]]
                else :
                    return [token]
            else:
                return subtokenize(" ".join(t[0:length//2])) + subtokenize(" ".join(t[length//2:]))


#generator
def tokenize_text(global_phonemizer, text):

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer') 
    doc = nlp(text)

    overflow = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) == 0:
            continue
        # text_words=list(nlp.tokenizer(s))
        tokens = phoneme_tokenize(global_phonemizer, s)
        if len(tokens) < 511:
            print(s,'\n')
            tokenlist = [tokens]
        else:
            tokenlist=subtokenize(global_phonemizer, s)

        # batch short text sequences, which may cause problems with generation. 
        for t in tokenlist:
            overflow+=t
            if len(overflow)>40:
                temp = overflow
                overflow = []
                yield temp

    if len(overflow)>0:
        yield overflow

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(
        prog="EpubToAudiobook",
        description="Read an epub (or other source) to audiobook format",
    )
    parser.add_argument("sourcefile", type=str, default="testtext.txt", nargs="?", help="The epub or text file to process")
    parser.add_argument("alpha", type=float, const=0.7, nargs="?", help="alpha value for the diffusion process")
    parser.add_argument("diffusion_steps", type=int, const=10, nargs="?", help="Number of diffusion steps")
    parser.add_argument("embedding_scale", type=float, const=1.5, nargs="?", help="Embedding scale")
    args = parser.parse_args()
    print(args)

    config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
    model_params = recursive_munch(config['model_params'])

    global_phonemizer, text_aligner, pitch_extractor, plbert = load_components(config)
    model = build_tts_model(device, model_params, text_aligner, pitch_extractor, plbert)
    model.global_phonemizer = global_phonemizer

    wavs = []
    s_prev = None

    with open(args.sourcefile, 'r') as sourcefile:
        text = sourcefile.read()
        text = text.replace('\n', ' ').replace('\r', ' ')


    s_ref = compute_style("Demo/reference_audio/1789_142896_000022_000005.wav", model)

    with AudioFile(
        f"output/libri_book.mp3",
        "w",
        samplerate=24000,
        num_channels=1,
        quality=64,  # kilobits per second
        ) as f:
        for i, token in enumerate(tokenize_text(global_phonemizer, text)):
            # print(i, token)
            wav, s_prev = LFinference(model, model_params, token, s_prev, s_ref, alpha=0.7, diffusion_steps=10, embedding_scale=3.5)
            f.write(wav)

if __name__ == "__main__":
    main()

#sshfs eingerman@rnd24.itw:Projects/DeepLearning/TTS/StyleTTS2/output/ target
    