import requests
import time
import concurrent.futures

api_url = 'http://34.69.125.197:8080/api/v1/static'
# api_url = 'http://127.0.0.1:8080/api/v1/static'
ping_url = 'http://34.141.243.146:8080/ping'

text = '''That's "great news"! Congratulations man!'''

def send_text_to_speech_request():
    # Parameters to send to the API
    data = {
        'text': text,
        'voice': "m-us-3", 
        'steps': 13, 
        'alpha': 0.2,
        'beta': 0.7,
        'speed': 0.8, 
        "embedding_scale":1
    }
    
    # Send POST request to the API
    response = requests.post(api_url, data=data)
   
    print("***",response)


# Modified function to include timing
def timed_send_text_to_speech_request(start_time):
    send_text_to_speech_request()
    print(f"Time taken: {time.time() - start_time} seconds")
    return time.time() - start_time

def spam_requests(num_requests):
    startTime = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        # Submitting the timed version of the function and collecting future objects
        futures = [executor.submit(timed_send_text_to_speech_request, startTime) for _ in range(num_requests)]
        # Retrieving results as they complete
        latencies = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Calculating average latency
    
    endTime = time.time()

    print(f"Total time taken: {endTime - startTime} seconds")

    return latencies

# Number of concurrent requests
num_requests = 10

# Calling the function to spam requests and measure latency
spam_requests(num_requests)