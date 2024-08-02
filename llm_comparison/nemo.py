import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from accelerate import Accelerator
import deepspeed

def run_nemo():
    # Authenticate with Hugging Face
    login(token="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")

    # Initialize the accelerator
    accelerator = Accelerator()

    language_model = "mistralai/Mistral-Nemo-Instruct-2407"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModelForCausalLM.from_pretrained(language_model)

    # Configure DeepSpeed for multi-GPU inference
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            }
        }
    }

    model = deepspeed.initialize(model=model, config=ds_config)[0]

    # Define context and query
    context_text = "Masamın üstünde bir suluk, bir bilgisayar ve iki kalem var."
    query_text = "Masamın üstünde ne var?"

    print(f"context text: {context_text}\n\nquery text: {query_text}\n\n")

    # Prepare the prompt with context and query
    input_text = f"Sen verilen bağlama göre soruları türkçe cevaplayan bir dil modelisin: {context_text}\n\nVerilen bağlama göre bu soruyu cevapla: {query_text}\n"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Move inputs to the appropriate device
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

    # Generate the answer
    outputs = model.generate(**inputs, max_new_tokens=256)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the response
    print(f"Response: {response_text}")

if __name__ == '__main__':
    run_nemo()
