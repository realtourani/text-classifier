import warnings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import time

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./model/').to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define the prediction function
def get_prediction(text):
    start_time = time.time()  # Start time
    
    # Tokenize input text
    encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    # Calculate probabilities
    probs = torch.nn.Sigmoid()(logits.squeeze().cpu()).numpy()
    label = np.argmax(probs, axis=-1)

    # Map label to class
    result = {
        'label': 'racist' if label == 1 else 'xenophobic',
        'probability': probs[label]
    }

    end_time = time.time()  # End time
    response_time = end_time - start_time

    return result, response_time

# Get user input
text = input("Enter text for prediction: ")
prediction, response_time = get_prediction(text)

print(f"Prediction: {prediction}")
print(f"Response time: {response_time:.4f} seconds")
