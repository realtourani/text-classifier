# Text Classifier using BERT

This repository contains code for a text classifier using BERT (Bidirectional Encoder Representations from Transformers) for sequence classification tasks. The model is fine-tuned on a specific dataset to classify text into two categories: "racist" or "xenophobic".

## Installation

1. Clone this repository:
```bash
git clone https://github.com/realtourani/text-classifier.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. First, download the trained model from here:
   https://drive.google.com/file/d/1G5LfE0iyMrc1WEMxT05RhTLnt0JQSAUP/view?usp=sharing

3. After that, put the downloaded model into the 'model' folder in the project directory, now the structure of the `model` folder is like this:
   - config.json
   - model.safetensors

4. Run the following Python script to make predictions:
```bash
python main.py
```

4. Enter the text for prediction when prompted.

## Code Structure
`main.py`: Python script for making predictions. It loads the pre-trained BERT model and tokenizer, tokenizes input text, performs inference, and returns the predicted label and response time.

## How it Works
1. The input text is tokenized using the BERT tokenizer with padding and truncation.
2. The pre-trained BERT model is loaded and used for inference on the input text.
3. The model outputs logits, which are passed through a sigmoid function to obtain probabilities.
4. The predicted label is determined based on the highest probability.
5. The predicted label and response time are returned as output.

## Model Information
The BERT model used is `bert-base-uncased` from Hugging Face Transformers.

