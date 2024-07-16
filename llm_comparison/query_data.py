import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langdetect import detect

from get_embedding_function import get_embedding_function

ollama_language_models = ["mistral", "llama3", "gemma2", "qwen2:7b", "llama3:27b", "gemma2:70b", "qwen2:72b"]
llama_language_models = ["curiositytech/MARS", "Eurdem/Defne_llama3_2x8B", "Metin/LLaMA-3-8B-Instruct-TR-DPO", "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct"]
mistral_language_models = ["Eurdem/megatron_1.1_MoE_2x7B"]
qwen2_language_models = ["Orbina/Orbita-v0.1"]

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE_EN = """
Answer the question based only on the following context: 
{context}
---
Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE_TR = """
Aşağıdaki bağlama dayanarak soruyu cevaplayın: 
{context}
---
Yukarıdaki bağlama dayanarak soruyu cevaplayın: {question}
"""

PROMPT_TEMPLATE_FR = """
Répondez à la question en vous basant uniquement sur le contexte suivant: 
{context}
---
Répondez à la question en vous basant sur le contexte ci-dessus: {question}
"""

PROMPT_TEMPLATE_ES = """
Responda a la pregunta basándose únicamente en el siguiente contexto: 
{context}
---
Responda a la pregunta basándose en el contexto anterior: {question}
"""

PROMPT_TEMPLATE_DE = """
Beantworten Sie die Frage nur anhand des folgenden Kontexts: 
{context}
---
Beantworten Sie die Frage anhand des obigen Kontexts: {question}
"""

PROMPT_TEMPLATE_IT = """
Rispondi alla domanda basandoti solo sul contesto seguente: 
{context}
---
Rispondi alla domanda basandoti sul contesto sopra: {question}
"""

def select_prompt_template(query_text: str) -> str:
    lang = detect(query_text)
    if lang == 'en':
        return PROMPT_TEMPLATE_EN
    elif lang == 'tr':
        return PROMPT_TEMPLATE_TR
    elif lang == 'fr':
        return PROMPT_TEMPLATE_FR
    elif lang == 'es':
        return PROMPT_TEMPLATE_ES
    elif lang == 'de':
        return PROMPT_TEMPLATE_DE
    elif lang == 'it':
        return PROMPT_TEMPLATE_IT
    else:
        return PROMPT_TEMPLATE_EN  # Default to English if language is not recognized
    
def select_message_template(query_text: str) -> str:
    lang_for_tr_llm = detect(query_text)
    if lang_for_tr_llm == 'tr':    
        return """
        Aşağıdaki bağlama dayanarak soruyu cevaplayın: 
        {context}
        ---
        Yukarıdaki bağlama dayanarak soruyu cevaplayın: {question}
        """
    else:
        return """
        Answer the question based only on the following context: 
        {context}
        ---
        Answer the question based on the above context: {question}
        """  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Sorgu metni.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str, language_model: str):
    start_time = time.time()

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    if language_model in ollama_language_models:
        prompt_template_str = select_prompt_template(query_text)
        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template.format(context=context_text, question=query_text)
        model = Ollama(model=language_model)
        response_text = model.invoke(prompt)

    elif language_model == "Eurdem/Defne_llama3_2x8B":
        prompt_template_str = select_message_template(query_text)
        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template.format(context=context_text, question=query_text)

        tokenizer = AutoTokenizer.from_pretrained(language_model)
        model = AutoModelForCausalLM.from_pretrained(language_model, torch_dtype=torch.bfloat16, device_map="auto")

        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.7, top_k=500,)
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))

    """inputs = tokenizer(messages, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, max_new_tokens=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)"""

    end_time = time.time()
    elapsed_time = end_time - start_time

    sources = [{"id": doc.metadata.get("id", None), "content": doc.page_content} for doc, _score in results]
    formatted_sources = "\n".join([f"{source['id']}: {source['content']}" for source in sources])
    
    formatted_response = f"Seçilen Model: {language_model}\nYanıt: {response_text}\nKaynaklar:\n{formatted_sources}\nSüre: {elapsed_time:.2f} saniye\n"
    print(formatted_response)
    return response_text


