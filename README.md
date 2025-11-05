
# 🧠 Medical AI Chatbot — LLaMA3 Fine-Tuned Model

This project presents an **AI-powered medical chatbot** built using the **LLaMA 3 model**, fine-tuned on **Arabic medical datasets** from Hugging Face.  
The model was trained and tested on **Google Colab**, and it is designed to **understand and respond to Arabic medical queries** with high accuracy and fluency.

---

## 🚀 Project Overview
The chatbot aims to assist patients by providing accurate medical information and guidance in Arabic, bridging the gap between users and healthcare centers.  
It forms part of a larger project — *Medical AI Voice & Chatbot System* — integrating AI into healthcare appointment management.

---

## 🧩 Model Details
- **Base Model:** Meta LLaMA 3 (8B)  
- **Fine-Tuning Framework:** Unsloth + LoRA  
- **Training Environment:** Google Colab (A100 GPU)  
- **Dataset:** 73 K Arabic doctor–patient dialogues (Hugging Face)  
- **Quantization:** 4-bit (bnb) for efficiency  
- **Evaluation Metrics:** Perplexity, accuracy on sample medical Q&A

---

## 🧠 Key Features
- Understands and answers Arabic medical questions  
- Supports contextual dialogue and follow-up questions  
- Fine-tuned for domain-specific vocabulary (diseases, symptoms, medications)  
- Lightweight model suitable for deployment on web or chatbot interfaces  

---

## 🧰 Technologies Used
| Category | Tools |
|-----------|-------|
| AI Model | LLaMA 3 8B – Unsloth + LoRA |
| Training | Python, Google Colab, Hugging Face |
| Frameworks | PyTorch, Transformers |
| Integration | Flask Backend / React Frontend |
| Version Control | Git + GitHub |

---

## 🧪 Example Usage
You can load the fine-tuned model in Colab using:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "jenanyousif/MedicalAIChatbot-llama3-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = השאלה בעברית: מהם התסמינים של אנמיה "#     ما هي أعراض فقر الدم ؟" 
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
