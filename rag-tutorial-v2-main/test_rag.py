from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="Üretici öğrenme nedir?",
        expected_response="Üretken bir model, önce Bayes kuralını kullanarak P(y|x) değerini tahmin etmek için kullanabileceğimiz P(x|y) değerini tahmin ederek verilerin nasıl üretildiğini öğrenmeye çalışır"
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="Ağaç temelli ve topluluk yöntemleri nedir?",
        expected_response="Bu yöntemler hem regresyon hem de sınıflandırma problemleri için kullanılabilir. CART: Sınıflandırma ve Regresyon Ağaçları (Classification and Regression Trees (CART)), genellikle karar ağaçları olarak bilinir, ikili ağaçlar olarak temsil edilirler. Rastgele orman: Rastgele seçilen özelliklerden oluşan çok sayıda karar ağacı kullanan ağaç tabanlı bir tekniktir. Basit karar ağacının tersine, oldukça yorumlanamaz bir yapıdadır ancak genel olarak iyi performansı onu popüler bir algoritma yapar. Not: Rastgele ormanlar topluluk yöntemlerindendir. Artırım: Artırım yöntemlerinin temel fikri bazı zayıf öğrenicileri bir araya getirerek güçlü bir öğrenici oluşturmaktır. Temel yöntemler aşağıdaki tabloda özetlenmiştir:",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
