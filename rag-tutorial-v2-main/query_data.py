import argparse
import time
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langdetect import detect

from get_embedding_function import get_embedding_function

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
Répondez à la question en vous basant uniquement sur le contexte suivant :

{context}

---

Répondez à la question en vous basant sur le contexte ci-dessus : {question}
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


def query_rag(query_text: str):
    start_time = time.time()

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template_str = select_prompt_template(query_text)
    print("lang: ", prompt_template_str, "\n")
    prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")  
    response_text = model.invoke(prompt)

    end_time = time.time()
    elapsed_time = end_time - start_time

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Yanıt: {response_text}\nKaynaklar: {sources}\nSüre: {elapsed_time:.2f} saniye"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
