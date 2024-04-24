import eventlet
eventlet.monkey_patch()

import io
from flask import Flask, Response, request, jsonify
from scipy.io.wavfile import write
import numpy as np
import msinference
from flask_cors import CORS
import time
from multiprocessing import Process, Queue




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

app = Flask(__name__)
cors = CORS(app)


def synthesize(text, steps = 10, alpha_ = 0.1, beta_ = 0.1, voice = 'm-us-3', speed = 1.0, embedding_scale = 1.0, device_index = 0):
    return msinference.inference(text, voices[voice][device_index], alpha=alpha_, beta=beta_, diffusion_steps=steps, embedding_scale=embedding_scale, speed=speed, device_index =device_index)

@app.route("/ping", methods=['GET'])
def ping():
    return "Pong"



@app.route("/api/v1/simulate", methods=['POST'])
def simulate_inference():
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



def handle_inference(queue, text, steps, alpha_, beta_, voice, speed, embedding_scale, device_index):
    try:
        print("Starting the synthesis process")
        startTime = time.time()
        # Synthesize audio (dummy function used here)
        # Replace this with the actual function call
        synth_audio = synthesize(text, steps, alpha_, beta_, voice, speed, embedding_scale, device_index)
        endTime = time.time()
        print(f"Synthesis completed in {endTime - startTime} seconds")

        output_buffer = io.BytesIO()
        write(output_buffer, 24000, np.array(synth_audio, dtype=np.int16))
        queue.put(output_buffer.getvalue())
    except Exception as e:
        queue.put(str(e))



@app.route("/api/v1/static", methods=['POST'])
def serve_wav():
    if 'text' not in request.form or 'voice' not in request.form:
        return jsonify({'error': 'Missing required fields. Please include "text" and "voice" in your request.'}), 400

    queue = Queue()
    text = request.form['text'].strip()
    steps = int(request.form.get('steps', 5))
    alpha_ = float(request.form.get('alpha', 0.3))
    beta_ = float(request.form.get('beta', 0.7))
    speed = float(request.form.get('speed', 1.0))
    device_index = int(request.form.get('device_index', 0))
    embedding_scale = float(request.form.get('embedding_scale', 1.0))

    # Create a new process to handle the inference
    proc = Process(target=handle_inference, args=(queue, text, steps, alpha_, beta_, request.form['voice'], speed, embedding_scale, device_index))
    proc.start()
    proc.join()

    # Retrieve the result from the queue
    result = queue.get()

    # Check if the result is an error message
    if isinstance(result, str):
        return jsonify({'error': result}), 500

    response = Response(result)
    response.headers["Content-Type"] = "audio/wav"
    return response





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)