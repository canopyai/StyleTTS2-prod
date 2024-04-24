import torch
import yaml
from cached_path import cached_path
from munch import Munch
import models  # Make sure this imports your model building script

# Assuming you have the appropriate imports for your text and phonemizer utilities
from text_utils import TextCleaner
from phonemizer import Phonemizer

# Recursive munch to handle nested configurations
def recursive_munch(data):
    """
    Recursively convert a nested dictionary into a Munch object.
    """
    if isinstance(data, dict):
        return Munch({k: recursive_munch(v) for k, v in data.items()})
    return data

# Initialize global utilities
global_phonemizer = Phonemizer()  # Initialize this with actual parameters if necessary
textclenaer = TextCleaner()

# Load the configurations from the YAML file
config_path = cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml")
config = yaml.safe_load(open(config_path))
model_params = recursive_munch(config['model_params'])

# Load additional required modules specified in the configuration
def load_required_modules(config):
    # Load ASR model
    ASR_path = config['ASR_path']
    ASR_config = config['ASR_config']
    text_aligner = models.load_ASR_models(ASR_path, ASR_config)

    # Load F0 model
    F0_path = config['F0_path']
    pitch_extractor = models.load_F0_models(F0_path)

    # Load BERT model
    BERT_path = config['PLBERT_dir']
    bert = models.load_plbert(BERT_path)

    return text_aligner, pitch_extractor, bert

text_aligner, pitch_extractor, bert = load_required_modules(config)

# Initialize and load models onto GPUs
num_gpus = 8  # Total number of GPUs available
models_dict = {}  # Dictionary to store models by GPU index

def initialize_models():
    params_path = cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth")
    params_whole = torch.load(params_path, map_location='cpu')
    params = params_whole['net']

    for i in range(num_gpus):
        device = f'cuda:{i}'
        model = models.build_model(model_params, text_aligner, pitch_extractor, bert).to(device)
        model.eval()  # Set model to evaluation mode

        # Load model weights
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key], strict=False)
                except RuntimeError as e:
                    print(f"Error loading state dict for {key}: {e}")

        models_dict[device] = model
        print(f'Model loaded on {device}')

initialize_models()
