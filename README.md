# Developing a Fine-Tuned Solution for Solving Numerical Linear Algebra Problems Using Large Language Models (NLA Project)🔗https://ieeexplore.ieee.org/document/11453791/references#references

This repository presents a research project focused on evaluating, fine-tuning, and deploying Large Language Models (LLMs) to enhance mathematical reasoning in the domain of **Numerical Linear Algebra (NLA)**.

The project follows a systematic pipeline involving dataset preparation, multi-model evaluation, comparative analysis, domain-specific fine-tuning, and deployment as an application.

---
## 🤗 Hugging Face Model

The fine-tuned Numerical Linear Algebra (NLA) model is publicly available on Hugging Face:

🔗 **Model Repository:**  
https://huggingface.co/Dilaksan/NLA/tree/main

## Project Overview

The methodology adopted in this project consists of the following stages:

1. Preparation of an NLA question–answer dataset  
2. Evaluation of multiple baseline LLMs  
3. Comparative analysis using semantic similarity and rank-based metrics  
4. Selection of the best-performing model  
5. Fine-tuning using domain-specific NLA data  
6. Deployment of the fine-tuned model as an application for solving NLA problems  

---

## Project Structure

```
nla/
├── data/
│   ├── answers/
│   │   ├── book/answers.csv
│   │   ├── deepseek/deepseek-r1-0528-1.csv
│   │   ├── google/gemini-2.5-pro1.csv
│   │   ├── new_model/answers.csv
│   │   ├── openai/o4-mini1.csv
│   │   └── x-ai/grok-3-mini-beta1.csv
│   └── questions/questions.csv
│
├── finetune/
│   ├── code/finetune.ipynb
│   └── dataset/
│       ├── final_answers.csv
│       └── final_questions.csv
│
├── nla_tutor/
│   ├── app.py
│   └── requirements.txt
│
├── src/
│   └── data/
│       ├── config.py
│       ├── functions.py
│       ├── main.py
│       ├── main2.py
│       ├── visual_compare_llms.py
│       ├── visual_compare_llms_vs_new_model.py
│       └── __pycache__/
│
├── report/llm_analysis_results.csv
├── img/
├── .gitignore
└── README.md
```

---

## Models Compared

- OpenAI – **o4-mini**  
- Google – **Gemini 2.5 Pro**  
- DeepSeek – **DeepSeek-R1-0528**  
- xAI – **Grok-3 Mini (Beta)**  
- Custom Fine-Tuned Model – **new_model**  
- Baseline – **Textbook (ground-truth) answers**


---

## Evaluation Methodology

- Cosine similarity for semantic alignment between generated and ground-truth solutions  
- Rank-based evaluation to assess relative model accuracy per question  
- Visualization of performance comparisons across baseline and fine-tuned models  

---

## Outcome

Based on comparative evaluation, the best-performing baseline model is selected and fine-tuned using domain-specific NLA data.  
The fine-tuned model demonstrates improved semantic and structural alignment with expert-annotated solutions and is deployed as an application for solving Numerical Linear Algebra problems.

---

## License

This project is intended for **research and academic use**.  
Please cite this work appropriately if used in publications.

