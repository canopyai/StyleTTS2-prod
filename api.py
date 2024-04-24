import eventlet
eventlet.monkey_patch()

import io

from scipy.io.wavfile import write
import numpy as np
import msinference
from flask_cors import CORS
import time
import torch
from fastapi import FastAPI 
from pydantic import BaseModel




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

class SynthesizeRequest(BaseModel):
    text: str
    steps: int
    alpha: float
    beta: float
    speed: float
    device_index: int = 0
    embedding_scale: float = 1.0
    voice: str = 'm-us-3'

print("Computing voices")
for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')
print("Starting Flask app")

app = FastAPI()


def synthesize(text, steps = 10, alpha_ = 0.1, beta_ = 0.1, voice = 'm-us-3', speed = 1.0, embedding_scale = 1.0, device_index = 0):
    return msinference.inference(text, voices[voice][device_index], alpha=alpha_, beta=beta_, diffusion_steps=steps, embedding_scale=embedding_scale, speed=speed, device_index =device_index)

@app.get("/ping")
async def ping():
    return "Pong"



@app.post("/api/v1/simulate")
async def simulate_inference():
    if 'text' not in request.form or 'voice' not in request.form:
        error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
        return jsonify(error_response), 400
    text = request.form['text'].strip()
    steps = int(request.form.get('steps'))
    alpha_ = float(request.form.get('alpha')) 
    beta_ = float(request.form.get('beta'))
    speed = float(request.form.get('speed'))
    device_index = int(request.form.get('device_index'))
    embedding_scale = float(request.form.get('embedding_scale'))
    audios = []
    sleep_time = 0.3
    time.sleep(sleep_time)
    return "response"


@app.post("/api/v1/static")
async def serve_wav(request_data: SynthesizeRequest):
    start_time = time.time()
    
    # Debug: print received data
    print(f"Received request data: {request_data}")

    # Call the synthesizer function (You'll need to define this elsewhere in your application)
    synth_audio = synthesize(
        request_data.text, 
        request_data.steps, 
        request_data.alpha, 
        request_data.beta, 
        request_data.voice, 
        request_data.speed, 
        embedding_scale=request_data.embedding_scale, 
        device_index=request_data.device_index
    )

    # Debug: print time taken for audio synthesis
    synth_audio_time = time.time()
    print(f"Time taken to synthesize audio: {synth_audio_time - start_time} seconds")

    # Concatenate audio and write to buffer
    audios = [synth_audio]
    output_buffer = io.BytesIO()
    write(output_buffer, 24000, np.concatenate(audios))

    # Create response with audio content
    response_content = output_buffer.getvalue()
    return {
        "content_type": "audio/wav",
        "audio_data": response_content
    }
