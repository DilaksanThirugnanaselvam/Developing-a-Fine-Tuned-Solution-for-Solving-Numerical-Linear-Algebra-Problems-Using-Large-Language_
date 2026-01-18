import csv
import os

import torch
from transformers import BitsAndBytesConfig, pipeline
from unsloth import FastLanguageModel

# === Suppress Torch Dynamo Errors (optional for speed) ===
torch._dynamo.config.suppress_errors = True

# === Load Your Fine-Tuned Model (with 4-bit and GPU) ===
model_path = "Dilaksan/NLA"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=8192,
    dtype=None,
    quantization_config=bnb_config,
    device_map="auto",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === LaTeX-Aware System Prompt ===
system_prompt = r"""
You are an expert in numerical linear algebra with deep knowledge.

Respond to each user question with a direct, well-structured answer using LaTeX for all mathematical notation.

Use LaTeX to represent:
- matrices (e.g., $A$),
- vectors (e.g., $\vec{x}$),
- norms (e.g., $\lVert x \rVert_2$),
- operators (e.g., $A^T A$, $\kappa(A)$),
- and complexity notations (e.g., $\mathcal{O}(n^3)$).

Do not include explanations about what you're doing.
Avoid meta-comments, apologies, or summaries.

Only provide the clean final answer using appropriate LaTeX syntax.
"""

# === File Paths ===
questions_path = "/content/questions2 - Copy.csv"
output_path = "/content/answers.csv"


# === Load CSV Questions ===
def load_csv_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return []


# === Save CSV Answers ===
def save_to_csv(data, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "answer"])
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        print(f"✅ Answers saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


# === Auto Save After Every Question ===
def auto_save_csv(data, output_path):
    try:
        save_to_csv(data, output_path)
        print(f"💾 Auto-saved after 1 question to {output_path}")
    except Exception as e:
        print(f"❌ Error auto-saving to CSV: {e}")


# === Main Processing ===
def process_questions():
    data = load_csv_data(questions_path)
    output_data = []
    question_counter = 0
    auto_save_interval = 1  # Auto-save after every question

    print("🚀 Starting Numerical Linear Algebra Answer Generation...\n")

    for entry in data:
        question_id = entry.get("id", "N/A")
        question = entry.get("questions", "")
        full_prompt = system_prompt.strip() + "\n\nUser Question: " + question.strip()

        print(f"🔍 Processing ID {question_id}: {question[:50]}...")

        try:
            response = pipe(full_prompt, max_new_tokens=2048)
            generated_text = response[0]["generated_text"]
            ai_answer = generated_text.replace(full_prompt, "").strip()
            print(f"📝 Answer Preview: {ai_answer[:50]}...\n")
        except Exception as e:
            print(f"❌ Error generating answer for ID {question_id}: {e}")
            ai_answer = "Error"

        output_data.append({"id": question_id, "answer": ai_answer})
        question_counter += 1

        # Auto-save after each question
        if question_counter % auto_save_interval == 0:
            auto_save_csv(output_data, output_path)

    # Final save
    auto_save_csv(output_data, output_path)

    print(f"\n🎯 All questions processed. Responses saved to:\n{output_path}")
    print("✅ Done.\n")


# === Run the Script ===
if __name__ == "__main__":
    process_questions()
