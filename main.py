import sys
import requests
import time

def query_model(inputs, candidate_labels, model_url, token):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": inputs,
        "parameters": {"candidate_labels": candidate_labels}
    }
    response = requests.post(model_url, headers=headers, json=payload)
    return response.json()

if __name__ == "__main__":
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    TOKEN = "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"

    inputs = sys.argv[1] if len(sys.argv) > 1 else input("Enter the text: ")
    candidate_labels = ["racist", "xenophobic"]

    start_time = time.time()  # Record start time

    output = query_model(inputs, candidate_labels, API_URL, TOKEN)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

    print(output['labels'][0])
