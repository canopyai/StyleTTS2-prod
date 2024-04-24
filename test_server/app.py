from fastapi import FastAPI, Response

from starlette.responses import StreamingResponse
import numpy as np
import io
import time
from scipy.io.wavfile import write
import base64


app = FastAPI()



@app.post("/api/v1/static")
async def serve_wav():
    # Audio parameters and dummy synthesis
    synth_audio = np.random.randn(24000 * 5).astype(np.float32)  # 5 seconds of random audio
    sample_rate = 24000  # 24 kHz sample rate

    # Create an in-memory WAV file
    buffer = io.BytesIO()
    write(buffer, sample_rate, synth_audio)
    buffer.seek(0)  # Rewind the buffer to the beginning

    # Read WAV file from buffer and encode to base64
    base64_encoded = base64.b64encode(buffer.read()).decode('utf-8')

    # Return base64 string as JSON
    return {"audio_base64": base64_encoded}

