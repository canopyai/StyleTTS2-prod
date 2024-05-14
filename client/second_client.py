import requests
import base64
import soundfile as sf

# API URL for the speaker processing endpoint
api_url = 'http://34.91.134.10:8080/api/v1/speaker'

# Path to the audio file you want to send
audio_file_path = './brit.wav'

def send_speaker_request(audio_file_path):
    try:
        # Read the audio file and encode it to base64

        audio, sr = sf.read(audio_file_path)

        audio_resampled = sf.resample(audio, sr, 24000)

        # Encode the resampled audio to base64
        audio_base64 = base64.b64encode(audio_resampled.tobytes()).decode('utf-8')

        data = {
            "b64": audio_base64
        }
                

        # Send the POST request to the server
        response = requests.post(api_url, json=data)

        # Check the response
        if response.status_code == 200:
            print("Successfully processed the speaker audio.")
            print("Response:", response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Send the speaker request
send_speaker_request(audio_file_path)
