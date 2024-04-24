# save this as list_cuda_devices.py
import torch

def list_cuda_devices():
    print("Number of CUDA Devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))

if __name__ == "__main__":
    list_cuda_devices()
