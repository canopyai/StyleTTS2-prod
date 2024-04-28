import requests
import time
import concurrent.futures
import uuid
import os
import base64

api_url = 'http://34.91.134.10:8080/api/v1/static'
# api_url = 'http://127.0.0.1:8080/api/v1/static'

#v100s
# api_url = 'http://35.204.162.196:8080/api/v1/static'



text = '''That's "great news"! Congratulations man!'''




def send_text_to_speech_request(stream_index):
    # sleep_time = 0.1 * stream_index
    # time.sleep(sleep_time)
    startTime = time.time()
    data = {
        'text': "That seems like it worked pretty darn well.", 
        'voice': "m-us-1",
        'steps': 7,
        'alpha': 0.2,
        'beta': 0.7,
        'speed': 0.8,
        "embedding_scale": 1,
    }

    try:
        response = requests.post(api_url, json=data)  # Use json=data if API expects JSON
        if response.status_code == 200:
            audio_base64 = response.json()['audio_base64']
            audio_data = base64.b64decode(audio_base64)
            random_file_name = f"outputs/{uuid.uuid4().hex}.wav"
            os.makedirs(os.path.dirname(random_file_name), exist_ok=True)
            with open(random_file_name, 'wb') as f:
                f.write(audio_data)
            end_time = time.time()
            print(f"Time taken: {end_time - startTime} seconds")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Modified function to include timing
def timed_send_text_to_speech_request(start_time, stream_index):
    send_text_to_speech_request(stream_index)
    return time.time() - start_time

def spam_requests(num_requests):
    startTime = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(timed_send_text_to_speech_request, startTime, _) for _ in range(num_requests)]


# Number of concurrent requests
num_requests = 1

# Calling the function to spam requests and measure latency
spam_requests(num_requests)