import requests

# URL of the Flask API
url = 'http://34.32.135.27:8080/api/v1/static'

# Data to be sent to the API
data = {
    'text': 'Hello, this is a test of the TTS system.',
    'voice': 'm-us-1'
}

# Send a POST request
response = requests.post(url, data=data)

# Check the status code to determine if the request was successful
if response.status_code == 200:
    # Save the audio file
    with open('output.wav', 'wb') as f:
        f.write(response.content)
    print("Audio saved successfully.")
else:
    # Print error message
    print(f"Failed to generate audio: {response.json()['error']}")

