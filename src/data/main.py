from datetime import datetime

from config import BASE_PATH
from functions import auto_save_csv, build_prompt, get_ai_response, load_csv_data
from logger_answer import log_request

# File Paths
questions_path = f"{BASE_PATH}/data/questions/questions.csv"
output_path = f"{BASE_PATH}/data/answers/deepseek/ gemini-2.5-pro4.csv"  # gemini-2.5-pro.csv"  # grok-3-mini-beta1.csv" #o4-mini.csv

# Load Data
csv_data = load_csv_data(questions_path)

# AI Processing
output_data = []
auto_save_interval = 1
question_counter = 0
default_model = "deepseek/deepseek-r1-0528:free"  # "google/gemini-2.5-pro-preview"  # "x-ai/grok-3-mini-beta"  # "openai/o4-mini"  # Fallback model

print("Starting Numerical Linear Algebra Question Processing\n")

for question_entry in csv_data:
    question_id = question_entry.get("id", "Unknown ID")
    question = question_entry.get("questions", "Unknown Question")
    model = default_model  # Use default model
    # Reformat question to LaTeX-enhanced one-line version
    formatted_question = build_prompt(question).split("Output:")[-1].strip().strip('"')
    print(
        f"Processing question ID {question_id}: {formatted_question[:50]}... (Model: {model})"
    )

    # Get AI response
    ai_answer, status_code, api_error = get_ai_response(model, formatted_question)

    # Log API call
    log_request(
        timestamp=datetime.now().isoformat(),
        section="Numerical Linear Algebra",
        question=formatted_question[:50] + "...",
        model=model,
        status_code=status_code,
        error=api_error,
    )
    print(f"\tOutput: {ai_answer[:50]}...")

    # Save response
    output_data.append({"id": question_id, "answer": ai_answer})
    question_counter += 1

    # Auto-save
    if question_counter % auto_save_interval == 0:
        auto_save_csv(output_data, output_path)

# Final save
auto_save_csv(output_data, output_path)

print(f"\nAll questions processed. Responses saved to:\n{output_path}")
print("Done.\n")
