import eventlet
eventlet.monkey_patch()
import io
from flask import Flask, Response, request, jsonify
from scipy.io.wavfile import write
import numpy as np
import msinference
from flask_cors import CORS
import time
import torch

print(torch.cuda.Stream())
num_streams = 4
streams = [torch.cuda.Stream() for _ in range(num_streams)]
stream_usage = [False] * num_streams  # False means available

print("all streams are", streams)

def get_available_stream():
    for i in range(num_streams):
        if not stream_usage[i]:
            stream_usage[i] = True
            return streams[i], i
    return None, None

def release_stream(index):
    stream_usage[index] = False

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
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
print("Computing voices")
for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')
print("Starting Flask app")

app = Flask(__name__)
cors = CORS(app)


def synthesize(text, steps = 10, alpha_ = 0.1, beta_ = 0.1, voice = 'm-us-3', speed = 1.0, embedding_scale = 1.0, stream = streams[0]):
    return msinference.inference(text, voices[voice], alpha=alpha_, beta=beta_, diffusion_steps=steps, embedding_scale=embedding_scale, speed=speed, stream =stream)

@app.route("/ping", methods=['GET'])
def ping():
    return "Pong"

@app.route("/api/v1/static", methods=['POST'])
def serve_wav(): 
    print("Received request")
    startTime = time.time()
    if 'text' not in request.form or 'voice' not in request.form:
        error_response = {'error': 'Missing required fields. Please include "text" and "voice" in your request.'}
        return jsonify(error_response), 400
    text = request.form['text'].strip()
    steps = int(request.form.get('steps'))
    alpha_ = float(request.form.get('alpha')) 
    beta_ = float(request.form.get('beta'))
    speed = float(request.form.get('speed'))
    embedding_scale = float(request.form.get('embedding_scale'))
    parseRequestTime = time.time()
    audios = []
    stream, index = get_available_stream()
    print("stream is", stream)
    synth_audio = synthesize(text, steps, alpha_, beta_, request.form['voice'], speed, embedding_scale=1.0, stream=stream)
    synth_audio_time = time.time()
    if stream is None:
        return jsonify({"error": "All streams are busy"}), 503  # Service Unavailable

    print(f"Time taken to synthesize audio: {synth_audio_time - parseRequestTime} seconds")
    audios.append(synth_audio)
    output_buffer = io.BytesIO()
    write(output_buffer, 24000, np.concatenate(audios))
    response = Response(output_buffer.getvalue())
    response.headers["Content-Type"] = "audio/wav"
    endTime = time.time()
    writeTime = time.time()
    print(f"Time taken to write audio: {writeTime - synth_audio_time} seconds")
    print(f"Time taken: {endTime - startTime} seconds")
    release_stream(index)
    return response
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)