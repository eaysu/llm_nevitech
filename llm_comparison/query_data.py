import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langdetect import detect
from huggingface_hub import login

from vllm_trial import LLM, SamplingParams

from get_embedding_function import get_embedding_function

ollama_language_models = ["mistral", "llama3", "gemma2", "qwen2:7b", "llama3:27b", "gemma2:70b", "qwen2:72b", "mixtral:8x7b", "mistral-nemo:12b"]

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
    prompt_template_str = select_prompt_template(query_text)
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    prompt = prompt_template.format(context=context_text, question=query_text)

    if language_model in ollama_language_models:
        model = Ollama(model=language_model)
        response_text = model.invoke(prompt)  

    elif language_model == "mistralai/Mistral-7B-v0.3":
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)

        llm = LLM(trust_remote_code=True, model="google/gemma-2-9b", dtype=torch.float16, tensor_parallel_size=4, max_model_len=20_000)

        context = """Zorunlu stajlarını tamamladıkları halde tekrar staj yapmak isteyen öğrencilere,
        Dekanlığın akademik dönem içerisinde belirlemiş olduğu şartları karşıladıkları 
        takdirde, izin verilebilir. Mağduriyet yaşanmaması adına böyle bir talebi olacak öğrencinin öncelikle Dekanlık 
        staj sorumlusu ile görüşüp durumu netleştirmesi ve ondan sonra girişimlerde bulunması gerekmektedir."""

        question = "Zorunlu stajlarımı tamamladıktan sonra tekrar staj yapabilir miyim?"

        chat_prompt = """
        [INST] 
        Verilen bağlam üzerinden soruları yanıtlamak üzere tasarlanmış bir yapay zeka asistanısınız. Göreviniz:
        1. Verilen bağlamı dikkatle okumak.
        2. Soruyu yalnızca bağlamdaki bilgileri kullanarak yanıtlamak.
        3. Eğer yanıt bağlamda bulunamıyorsa, "Bu soruyu verilen bağlama dayanarak yanıtlayamıyorum." şeklinde cevap vermek.
        4. Kısa ve doğru yanıtlar sağlamak. 
        [/INST]
        """

        prompt = chat_prompt + f"\n\nQuestion: {question}\n\nDocument: {context}"

        completion = llm.generate(prompt, sampling_params)
        # Extract and clean the output text
        response_text = completion[0].outputs[0].text.strip()

        # Post-process to remove unnecessary parts
        if "Answer:" in response_text:
            response_text = response_text.split("Answer:")[1].strip()

    else:
        tokenizer = AutoTokenizer.from_pretrained(language_model)
        model = AutoModelForCausalLM.from_pretrained(language_model)

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    sources = [{"id": doc.metadata.get("id", None), "content": doc.page_content} for doc, _score in results]
    formatted_sources = "\n".join([f"{source['id']}: {source['content']}" for source in sources])
    
    formatted_response = f"Seçilen Model: {language_model}\nYanıt: {response_text}\nSüre: {elapsed_time:.2f} saniye\n" #Kaynaklar:\n{formatted_sources}\n#
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
