import torch

def get_device_names():
    device_names = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.cuda.get_device_name(i)
            device_names.append(f"cuda:{i}")
    else:
        device_names.append("cpu")
    return device_names
