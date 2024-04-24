import torch
import yaml
from cached_path import cached_path
from models import build_model  # Assuming build_model is properly defined
import numpy as np
from nltk.tokenize import word_tokenize

# Setup configuration and utilities
num_gpus = 8  # Total number of GPUs available
models = {}   # Dictionary to store models by GPU index

def initialize_models():
    # Load configuration from a hosted YAML file
    config_path = cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml")
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])

    # Load parameters from a hosted checkpoint file
    params_path = cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth")
    params_whole = torch.load(params_path, map_location='cpu')
    params = params_whole['net']

    # Initialize and load a model on each GPU
    for i in range(num_gpus):
        device = f'cuda:{i}'
        model = build_model(model_params).to(device)
        model.eval()  # Set model to evaluation mode

        # Load model weights
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key], strict=False)
                except RuntimeError as e:
                    print(f"Error loading state dict for {key}: {e}")

        models[device] = model
        print(f'Model loaded on {device}')

# Define the inference function
def inference(text, ref_s, selected_device=0):
    device = f'cuda:{selected_device}'
    model = models[device]
    
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

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=1,  # Assuming embedding_scale is constant
                         features=ref_s,  # Ensure ref_s is on the correct device
                         num_steps=5).squeeze(1)  # Assuming fixed number of diffusion steps

        # Continue with your specific processing logic here
        # Ensure all tensor operations are performed on `device`
        # Process s_pred and generate final output as per your model's design

# Initialize models on all GPUs
initialize_models()

# Example of running inference
# You need to prepare `ref_s_tensor` and ensure it's on the correct device
# For example:
# ref_s_tensor = torch.randn(1, 256).to('cuda:3')  # Example tensor, use actual data
# output = inference("Hello world", ref_s_tensor, selected_device=3)
