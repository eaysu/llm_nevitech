from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import deepspeed
import torch

torch.cuda.empty_cache()

accelerator = Accelerator()

language_model = 'mistralai/Mistral-Nemo-Instruct-2407'

def run_deepspeed():
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    model = AutoModelForCausalLM.from_pretrained(language_model)

    # Initialize DeepSpeed inference engine
    ds_model = deepspeed.init_inference(
        model=model,         # Transformers model
        mp_size=1,           # Number of GPUs
        dtype=torch.half,    # dtype of the weights (fp16)
        replace_method="auto", # Let DeepSpeed automatically identify the layer to replace
        replace_with_kernel_inject=True, # Replace the model with the kernel injector
        max_tokens=1024,
    )

    # Create a text-generation pipeline with DeepSpeed model
    ds_clf = pipeline("text-generation", model=ds_model.module, tokenizer=tokenizer, device=0)

    context = """Zorunlu stajlarını tamamladıkları halde tekrar staj yapmak isteyen öğrencilere,
    Dekanlığın akademik dönem içerisinde belirlemiş olduğu şartları karşıladıkları 
    takdirde, izin verilebilir. Mağduriyet yaşanmaması adına böyle bir talebi olacak öğrencinin öncelikle Dekanlık 
    staj sorumlusu ile görüşüp durumu netleştirmesi ve ondan sonra girişimlerde bulunması gerekmektedir."""

    question = "Zorunlu stajlarımı tamamladıktan sonra tekrar staj yapabilir miyim?"

    response = ask_question(ds_clf=ds_clf, context_text=context, query_text=question)

    print(f"\nYANIT: {response}\n")

def ask_question(ds_clf, context_text, query_text):
    messages = [
        {"role": "system", "content": f"Sen verilen bağlama göre soruları türkçe cevaplayan bir dil modelisin: {context_text}\n"},
        {"role": "user", "content": f"Verilen bağlama göre bu soruyu cevapla: {query_text}\n"},
    ]
    
    # Combine the messages into a single input text
    input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Generate the answer
    result = ds_clf(input_text, max_length=2000)
    
    # Extract the generated text from the result
    answer = result[0]['generated_text']
    
    return answer

if __name__ == '__main__':
    run_deepspeed()

