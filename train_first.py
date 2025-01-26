import os
import os.path as osp
import glob
import re
import sys
import yaml
import shutil
import traceback
import numpy as np
import torch
import click
from socket import gethostname
import warnings
warnings.simplefilter('ignore')

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torch.autograd.profiler as profiler

import librosa

from models import *
from meldataset import build_dataloader
from utils import *
from losses import *
from optimizers import build_optimizer
import time

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

import logging
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="DEBUG")

# The flag below controls whether to allow TF32 on matmul.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def tuple_range(start, length, step):
    t = tuple(range(start, length, step))
    if len(t)>1 and length-t[-1] < 8*1024:    #make sure that the last segment is not too small
        t=(t[:-1] + (length,))
    return t



class MEM_PROBE(object):
    def __init__(self):
        self.last_mem = torch.cuda.memory_allocated()

    def __call__(self):
        current_mem = torch.cuda.memory_allocated()
        mem_delta = current_mem - self.last_mem
        self.last_mem = current_mem
        logger.info(f"MEM:{sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno} {mem_delta/(1024*1024*1024):.2f} : {current_mem/(1024*1024*1024):.2f} GB")
        return None

@click.command()
@click.option('-p', '--config_path', default='Configs/config_libritts_espeak.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    #memory use debugging
    torch.cuda.memory._record_memory_history(max_entries=100000)
    memory_probe = MEM_PROBE()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    hostname = gethostname()
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('{asctime} %s {name}: {levelname:8} {message}' % hostname,
        datefmt='%b %d %H:%M:%S',
        style='{'))
    logger.logger.addHandler(file_handler)
    
    batch_size = config.get('batch_size', 10)
    chunk_size = config.get('chunk_size', 20480)

    device = accelerator.device
    
    epochs = config.get('epochs_1st', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']
    
    max_len = config.get('max_len', 200)
    
    checkpoint_path = ""
    all_checkpoints = list(filter(osp.isfile, glob.glob(osp.join(log_dir, "checkpoint_*.pth"))))
    if all_checkpoints:
        all_checkpoints.sort(key=lambda x: osp.getmtime(x))
        checkpoint_path = all_checkpoints[-1]

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    memory_probe()
    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        memory_probe()

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        memory_probe()

        # load BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)
        memory_probe()

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    memory_probe()

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    
    for k in model:
        model[k] = accelerator.prepare(model[k])

    memory_probe()
    
    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )
    memory_probe()
    
    _ = [model[key].to(device) for key in model]
    memory_probe()

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                  scheduler_params_dict= {key: scheduler_params.copy() for key in model},
                               lr=float(config['optimizer_params'].get('lr', 1e-4)))
    memory_probe()
    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    memory_probe()
    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))
        elif checkpoint_path != "":
            try:
                model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, checkpoint_path,
                                            load_only_params=False)           
                print(f"Loading {epoch=}/{iters=} from checkpoint {checkpoint_path}")
            except Exception as e:
                print(f"Loading from checkpoint {checkpoint_path} failed")
                print(e)
        else:
            start_epoch = 0
            iters = 0

    memory_probe()
    # in case not distributed
    try:
        n_down = model.text_aligner.module.n_down
    except:
        n_down = model.text_aligner.n_down
    
    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    memory_probe()
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    memory_probe()

    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    memory_probe()
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)
    memory_probe()
    try:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir + "/tensorboard"),
            record_shapes=True, profile_memory=True, with_modules=True,
            with_stack=True)
        prof.start()
        for epoch in range(start_epoch, epochs):
            running_loss = 0
            start_time = time.time()

            _ = [model[key].train() for key in model]

            for i, batch in enumerate(train_dataloader):
                prof.step()
                memory_probe()
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
        
                memory_probe()
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch
                
                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                memory_probe()
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

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                memory_probe()

                # 50% of chance of using monotonic version
                if bool(random.getrandbits(1)):
                    asr = (t_en @ s2s_attn)
                else:
                    asr = (t_en @ s2s_attn_mono)
                memory_probe()

                # get clips
                mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
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

                # clip too short to be used by the style encoder
                if gt.shape[-1] < 80:
                    continue
                    
                with torch.no_grad():    
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                memory_probe()
        
                s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
                memory_probe()
                
                y_rec = model.decoder(en, F0_real, real_norm, s)
                memory_probe() 
                
                # print(f"{y_rec.shape=}")
                # discriminator loss
                
                if epoch >= TMA_epoch:

                    optimizer.zero_grad()

                    w = wav.detach().unsqueeze(1).float()
                    y = y_rec.detach()
                    idx = tuple_range(chunk_size, w.shape[2], chunk_size)
                    for w_chunk, y_chunk in zip(torch.tensor_split(w, idx, dim=2), torch.tensor_split(y, idx, dim=2)):
                        d_loss = dl(w_chunk.detach(), y_chunk.detach())
                        accelerator.backward(d_loss)
                    memory_probe()

                    optimizer.step('msd')
                    memory_probe()
                    optimizer.step('mpd')
                    memory_probe()
                else:
                    d_loss = 0

                # generator loss
                optimizer.zero_grad()
                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())
                memory_probe()

                if epoch >= TMA_epoch: # start TMA training
                    loss_s2s = 0
                    for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                        loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                    loss_s2s /= texts.size(0)
                    memory_probe()

                    loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                    memory_probe()
                    
                    w = wav.detach().unsqueeze(1).float()
                    idx = tuple_range(chunk_size, w.shape[2], chunk_size)
                    for w_chunk, y_chunk in zip(torch.tensor_split(w, idx, dim=2), torch.tensor_split(y_rec, idx, dim=2)):
                        loss_gen_all = gl(w_chunk.detach(), y_chunk)
                        accelerator.backward(loss_gen_all, retain_graph=True)  #TODO: check if this is correct
                    memory_probe()
                    
                    for w_chunk, y_chunk in zip(torch.tensor_split(wav.detach(), idx, dim=1), torch.tensor_split(y_rec, idx, dim=2)):
                        loss_slm = wl(w_chunk.detach(), y_chunk)
                        accelerator.backward(loss_slm, retain_graph=True)  #TODO: check if this is correct
                    memory_probe()

                    g_loss = loss_params.lambda_mel * loss_mel + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s + \
                    loss_params.lambda_gen * loss_gen_all + \
                    loss_params.lambda_slm * loss_slm

                else:
                    loss_s2s = 0
                    loss_mono = 0
                    loss_gen_all = 0
                    loss_slm = 0
                    g_loss = loss_mel
                
                running_loss += accelerator.gather(loss_mel).mean().item()
                memory_probe()
                accelerator.backward(g_loss)
                memory_probe()
                optimizer.step('text_encoder')
                optimizer.step('style_encoder')
                optimizer.step('decoder')
                memory_probe()

                if epoch >= TMA_epoch: 
                    optimizer.step('text_aligner')
                    optimizer.step('pitch_extractor')
                    memory_probe()
                
                iters = iters + 1
                
                if (i+1)%log_interval == 0 and accelerator.is_main_process:
                    log_print ('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f'
                            %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm), logger)
                    
                    writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                    writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                    writer.add_scalar('train/d_loss', d_loss, iters)
                    writer.add_scalar('train/mono_loss', loss_mono, iters)
                    writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                    writer.add_scalar('train/slm_loss', loss_slm, iters)

                    running_loss = 0
                    
                    print('Time elasped:', time.time()-start_time)
                                    
            loss_test = 0

            _ = [model[key].eval() for key in model]
            memory_probe()

            with torch.no_grad():
                iters_test = 0
                for batch_idx, batch in enumerate(val_dataloader):
                    optimizer.zero_grad()

                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        text_mask = length_to_mask(input_lengths).to(texts.device)
                        attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                        attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                        attn_mask = (attn_mask < 1)
                        s2s_attn.masked_fill_(attn_mask, 0.0)

                    # encode
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    
                    asr = (t_en @ s2s_attn)

                    # get clips
                    mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
                    mel_len = min([int(mel_input_length.min().item() / 2 - 1), max_len // 2])
                    
                    en = []
                    gt = []
                    wav = []
                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    gt = torch.stack(gt).detach()

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                    y_rec = model.decoder(en, F0_real, real_norm, s)

                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    loss_test += accelerator.gather(loss_mel).mean().item()
                    iters_test += 1
            memory_probe()

            if accelerator.is_main_process:
                print('Epochs:', epoch + 1)
                log_print('Validation loss: %.3f' % (loss_test / iters_test) + '\n\n\n\n', logger)
                print('\n\n\n')
                writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
                attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
                writer.add_figure('eval/attn', attn_image, epoch)
                
                with torch.no_grad():
                    for bib in range(len(asr)):
                        mel_length = int(mel_input_length[bib].item())
                        gt = mels[bib, :, :mel_length].unsqueeze(0)
                        en = asr[bib, :, :mel_length // 2].unsqueeze(0)
                                            
                        F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                        F0_real = F0_real.unsqueeze(0)
                        s = model.style_encoder(gt.unsqueeze(1))
                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        
                        y_rec = model.decoder(en, F0_real, real_norm, s)
                        
                        writer.add_audio('eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                        if epoch == 0:
                            writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)
                        
                        if bib >= 6:
                            break

                if epoch % saving_epoch == 0:
                    if (loss_test / iters_test) < best_loss:
                        best_loss = loss_test / iters_test
                    print('Saving..')
                    state = {
                        'net':  {key: model[key].state_dict() for key in model}, 
                        'optimizer': optimizer.state_dict(),
                        'iters': iters,
                        'val_loss': loss_test / iters_test,
                        'epoch': epoch,
                    }
                    
                    torch.cuda.memory._dump_snapshot(osp.join(log_dir, 'memory_snapshot_%05d.pickle' % epoch))
                    
                    save_path = osp.join(log_dir, f"checkpoint_{epoch}.pth")
                    torch.save(state, save_path)
                                    
        if accelerator.is_main_process:
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            torch.save(state, save_path)
    except Exception as e:
        prof.stop()
        # prof.export_chrome_trace(osp.join(log_dir, 'memory_trace_%05d.json' % epoch))
        prof.export_stacks(osp.join(log_dir, 'memory_stacks_%05d.pickle' % epoch))
        prof.export_memory_timeline(osp.join(log_dir, 'memory_timeline_%05d.pickle' % epoch))
        traceback.print_exc()
        print(f"Exception: {e}")
        torch.cuda.memory._dump_snapshot(osp.join(log_dir, 'memory_exception_%05d.pickle' % epoch))
        exit(0)
    prof.stop()
        
        
    
if __name__=="__main__":
    main()
