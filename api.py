import eventlet
eventlet.monkey_patch()
import io
from scipy.io.wavfile import write
import numpy as np
import msinference
from flask_cors import CORS
import time
import torch
from fastapi import FastAPI, Response
import base64
from pydantic import BaseModel


class TextToSpeechRequest(BaseModel):
    text: str
    voice: str
    steps: int
    alpha: float
    beta: float
    speed: float
    embedding_scale: float


def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000 * 10**6
    o = bytes("RIFF", "ascii")
    o += (datasize + 36).to_bytes(4, "little")
    o += bytes("WAVE", "ascii")
    o += bytes("fmt ", "ascii")
    o += (16).to_bytes(4, "little")
    o += (1).to_bytes(2, "little")
    o += (channels).to_bytes(2, "little")
    o += (sampleRate).to_bytes(4, "little")
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, "little")
    o += (channels * bitsPerSample // 8).to_bytes(2, "little")
    o += (bitsPerSample).to_bytes(2, "little")
    o += bytes("data", "ascii")
    o += (datasize).to_bytes(4, "little")
    return o

voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
voices = {}

for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')

print("Finished Computing Voice Embeddings")

app = FastAPI()


def synthesize(text, steps = 10, alpha_ = 0.1, beta_ = 0.1, voice = 'm-us-3', speed = 1.0, embedding_scale = 1.0, device_index = 0):
    return msinference.inference(text, voices[voice][device_index], alpha=alpha_, beta=beta_, diffusion_steps=steps, embedding_scale=embedding_scale, speed=speed, device_index =device_index)

@app.get("/ping")
async def ping():
    return "Pong"


device_index_tracker = 0

number_of_devices = torch.cuda.device_count()

def get_device_index():
    global device_index_tracker

    device_index_tracker = device_index_tracker +1
    if(device_index_tracker == number_of_devices):
        device_index_tracker = 0

    return device_index_tracker
    




@app.post("/api/v1/static")
async def serve_wav(request: TextToSpeechRequest):
    startTime = time.time()
    synth_audio = synthesize(
        request.text, request.steps, request.alpha, request.beta, 
        request.voice, request.speed, request.embedding_scale, get_device_index()
    )
    write('result.wav', 24000, synth_audio)
    sample_rate = 24000
    buffer = io.BytesIO()
    write(buffer, sample_rate, synth_audio)
    buffer.seek(0)
    base64_encoded = base64.b64encode(buffer.read()).decode('utf-8')
    print(f"Time taken: {time.time() - startTime} seconds")
    return {"audio_base64": base64_encoded}