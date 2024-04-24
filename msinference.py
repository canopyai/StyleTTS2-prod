from cached_path import cached_path
import nltk
nltk.download('punkt')
from scipy.io.wavfile import write
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from collections import OrderedDict

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import random
import yaml
from munch import Munch
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()
from Utils.PLBERT.util import load_plbert
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule







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

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)






def initialize_and_load_model(device='cpu'):
    # Load configuration
    config_path = "hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml"
    with open(str(cached_path(config_path)), 'r') as file:
        config = yaml.safe_load(file)

    # Load models based on configuration
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # Prepare model parameters
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Set models to evaluation mode and move them to the specified device
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # Load model weights
    params_path = "hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
    params_whole = torch.load(str(cached_path(params_path)), map_location=device)
    params = params_whole['net']

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except RuntimeError as e:
                # Handle module prefix if necessary
                state_dict = params[key]
                new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
                model[key].load_state_dict(new_state_dict, strict=False)

    # Initialize DiffusionSampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    return model, sampler, model_params

# Now you can use this function to initialize and load your model and sampler
model, sampler, model_params = initialize_and_load_model("cuda:0")


def get_device_names():
    device_names = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            device_names.append(device)
    else:
        device_names.append("cpu")
    return device_names

device_names = get_device_names()

model_dicts = []

for device_name in device_names:    
    model, sampler, model_params = initialize_and_load_model(device_name)
    model_dicts.append({
        "device_name": device_name,
        "model": model,
        "sampler": sampler,
        "model_params": model_params
    })







def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1, speed = 1, use_gruut=False, device_index=0):
        
        device = device_names[device_index]
        
        model = model_dicts[0]["model"]
        sampler = model_dicts[0]["sampler"]
        model_params = model_dicts[0]["model_params"]
        

        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)



        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)


            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze() * speed).clamp(min=1)


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

        torch.cuda.synchronize() 
        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


