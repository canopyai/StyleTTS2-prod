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
async def serve_wav():
    text = "Hello there, can you hear me?"
    steps = 10
    alpha = 0.1
    beta = 0.1
    speed = 1.0
    device_index = 0
    embedding_scale = 1.0
    voice = "m-us-3"

    parse_request_time = time.time()
    synth_audio = synthesize(text, steps, alpha, beta, voice, speed, embedding_scale, device_index)
    print(synth_audio)

    synth_audio_time = time.time()

    print(f"Time taken to synthesize audio: {synth_audio_time - parse_request_time} seconds")

    output_buffer = io.BytesIO()
    # Assume audio sample rate is 24000 Hz, update accordingly
    write(output_buffer, 24000, synth_audio.astype(np.int16))
    output_buffer.seek(0)  # Rewind buffer to the beginning

    return Response(content=output_buffer.getvalue(), media_type="audio/wav")