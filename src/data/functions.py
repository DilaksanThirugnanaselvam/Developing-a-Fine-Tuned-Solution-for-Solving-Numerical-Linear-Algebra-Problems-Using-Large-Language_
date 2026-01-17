import csv
import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError(
        "API key not found. Please set OPENROUTER_API_KEY in your .env file."
    )


def build_prompt(input_text):
    clean_text = " ".join(input_text.split())
    return f"""
You are a helpful assistant that reformats poorly written math problems into a clean, one-line, LaTeX-enhanced version.

Example:
Output: "Give the proof that the matrix $A^T A$ is positive definite if and only if the columns of $A$ are linearly independent."

Input: "{input_text}"
Output: "{clean_text}"
"""


def load_csv_data(file_path):
    """Loads CSV data from the given file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return []
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return []


def get_ai_response(model, question) -> tuple[str, int | None, Exception | None]:
    """Sends a request to OpenRouter API and retrieves AI response.

    Args:
        model (str): The OpenRouter model ID.
        question (str): The LaTeX-enhanced one-line question to fetch the answer for.

    Returns:
        tuple[str, int | None, Exception | None]: The response, status code, and exception if any.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    system_prompt = r"""You are an expert in numerical linear algebra with deep knowledge.

Respond to each user question generate the answer using LaTeX for all mathematical notation.

Use LaTeX to represent:
- matrices (e.g., $A$),
- vectors (e.g., $x$),
- norms (e.g., $||x||_2$),
- operators (e.g., $A^TA$, $\kappa(A)$),
- and complexity terms (e.g., $\mathcal{O}(n^3)$).

provide meta-instructions or summaries. Provide well-structured, notation-rich,step by step full answers ."""

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if response.status_code == 200:
            return (
                result["choices"][0]["message"]["content"].strip(),
                response.status_code,
                None,
            )
        return "Null", response.status_code, None
    except requests.RequestException as e:
        return "Null", getattr(response, "status_code", None), e
    except Exception as e:
        return "Null", getattr(response, "status_code", None), e


def save_to_csv(data, output_path):
    """Save data to a CSV file with id and answer columns."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "answer"])
            writer.writeheader()
            for item in data:
                writer.writerow({"id": item["id"], "answer": item["answer"].strip()})
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to CSV file: {e}")


def auto_save_csv(data, output_path):
    """Auto-saves the data into a CSV file at intervals."""
    try:
        save_to_csv(data, output_path)
        print(f"Auto-saved progress to {output_path}")
    except Exception as e:
        print(f"Error auto-saving to CSV: {e}")
