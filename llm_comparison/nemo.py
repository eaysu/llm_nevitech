import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from accelerate import Accelerator
import torch        

def run_nemo():
    # Authenticate with Hugging Face
    login(token="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")
    # Initialize the accelerator
    accelerator = Accelerator()

    language_model = "mistralai/Mistral-Nemo-Instruct-2407"

    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModelForCausalLM.from_pretrained(language_model)

    # Prepare the model for distributed training/inference
    model = accelerator.prepare(model)

    # Define context and query
    context_text = "Masamın üstünde bir suluk, bir bilgisayar ve iki kalem var."
    query_text = "Masamın üstünde ne var?"

    print(f"context text: {context_text}\n\nquery text: {query_text}\n\n")

    # Prepare the prompt with context and query
    messages = [
        {"role": "system", "content": f"Sen verilen bağlama göre soruları türkçe cevaplayan bir dil modelisin: {context_text}\n"},
        {"role": "user", "content": f"Verilen bağlama göre bu soruyu cevapla: {query_text}\n"},
    ]

    # Combine the messages into a single input string
    input_text = messages[0]['content'] + "\n" + messages[1]['content']
    inputs = tokenizer(input_text, return_tensors="pt")

    # Remove 'token_type_ids' from inputs if present
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

    # Generate the answer
    outputs = model.generate(**inputs, max_new_tokens=256)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer from the response text
    # This assumes the answer follows the query text directly
    start_index = response_text.find(query_text) + len(query_text)
    response_text = response_text[start_index:].strip()

if __name__ == '__main__':
    run_nemo()