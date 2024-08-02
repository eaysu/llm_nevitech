from vllm import LLM, SamplingParams
import torch

torch.cuda.empty_cache()

def run_vllm():
    # Adjusted sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=500)

    llm = LLM(trust_remote_code=True, model="mistralai/Mistral-7B-v0.3", dtype=torch.float16, tensor_parallel_size=4, max_model_len=10_000)

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
    output_text = completion[0].outputs[0].text.strip()

    # Post-process to extract only the "Cevap" part
    if "Cevap:" in output_text:
        output_text = output_text.split("Cevap:")[1].strip()
    
    print("\nCevap:", output_text, "\n")

if __name__ == '__main__':
    run_vllm()
