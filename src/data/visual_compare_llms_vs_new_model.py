import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths to ground truth and LLM answers
answer_path = r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\book\answers.csv"
llm_paths = {
    "deepseek-r1-0528": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\deepseek\deepseek-r1-0528-1.csv",
    "gemini-2.5-pro": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\google\gemini-2.5-pro1.csv",
    "grok-3-mini": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\x-ai\grok-3-mini-beta1.csv",
    "04-mini-high": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\openai\o4-mini1.csv",
    "Dilaksan/nla": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\new_model\answers.csv",
}

# Load ground truth answers
ground_truth_df = pd.read_csv(answer_path)
ground_truth_answers = ground_truth_df["answer"].tolist()
ground_truth_embeddings = model.encode(ground_truth_answers, convert_to_tensor=False)

# Dictionary to store results
llm_results = {}

# Process each LLM
for llm_name, llm_path in llm_paths.items():
    llm_df = pd.read_csv(llm_path)
    if "answer" not in llm_df.columns:
        print(f"Skipping {llm_name} — missing 'answer' column")
        continue

    llm_answers = llm_df["answer"].tolist()
    llm_embeddings = model.encode(llm_answers, convert_to_tensor=False)

    # Adjust number of questions based on available data
    num_questions = min(100, len(llm_embeddings), len(ground_truth_embeddings))

    # Scenario 1: Cosine similarity for aligned Q&A
    similarity_scores = [
        cosine_similarity([ground_truth_embeddings[i]], [llm_embeddings[i]])[0][0]
        for i in range(num_questions)
    ]

    # Scenario 2: Rank correct answer among all
    ranks = []
    scenario2_similarities = []
    for i in range(num_questions):
        question_embedding = llm_embeddings[i]
        all_similarities = cosine_similarity(
            [question_embedding], ground_truth_embeddings
        )[0]
        rank = np.where(np.argsort(-all_similarities) == i)[0][0] + 1
        ranks.append(min(rank, 11))  # Cap at 11
        scenario2_similarities.append(all_similarities[np.argmax(all_similarities)])

    llm_results[llm_name] = {
        "similarity_scores": similarity_scores,
        "ranks": ranks,
        "scenario2_similarities": scenario2_similarities,
        "mean_similarity_s1": np.mean(similarity_scores),
        "mean_similarity_s2": np.mean(scenario2_similarities),
        "num_questions": num_questions,
    }

# Print summary
print("\nMean Similarity Scores:")
for llm_name, results in llm_results.items():
    print(
        f"{llm_name}: Scenario 1 = {results['mean_similarity_s1']:.3f}, "
        f"Scenario 2 = {results['mean_similarity_s2']:.3f}, "
        f"Questions Evaluated = {results['num_questions']}"
    )

# Export to CSV
results_df = []
for llm_name, results in llm_results.items():
    for qid in range(results["num_questions"]):
        results_df.append(
            {
                "LLM": llm_name,
                "Question_ID": qid + 1,
                "Scenario_1_Similarity": results["similarity_scores"][qid],
                "Scenario_2_Similarity": results["scenario2_similarities"][qid],
                "Rank": results["ranks"][qid],
            }
        )
results_df = pd.DataFrame(results_df)
results_df.to_csv("llm_analysis_results.csv", index=False)
print("\nResults exported to 'llm_analysis_results.csv'")

# Visualization 1: Scenario 1 Distribution (Histograms)
plt.figure(figsize=(12, 6))
colors = ["firebrick", "forestgreen", "navy", "purple", "orange"]
for idx, (llm_name, results) in enumerate(llm_results.items()):
    plt.hist(
        results["similarity_scores"],
        bins=10,
        alpha=0.5,
        color=colors[idx % len(colors)],
        label=llm_name,
        edgecolor="black",
    )
plt.title("Distribution of Similarity Scores (Scenario 1)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.xlim(0.4, 1.0)
plt.legend()
plt.savefig("scenario1_distribution_updated.png")
plt.show()

# Visualization 2: Rank per Question (Bar Chart)
plt.figure(figsize=(14, 6))
bar_width = 0.15
x = np.arange(1, 101)
for idx, (llm_name, results) in enumerate(llm_results.items()):
    plt.bar(
        x[: results["num_questions"]] + idx * bar_width,
        results["ranks"],
        bar_width,
        alpha=0.8,
        color=colors[idx % len(colors)],
        label=llm_name,
    )
plt.title("Rank of Correct Answer (Scenario 2)")
plt.xlabel("Question ID")
plt.ylabel("Rank")
plt.grid(True, alpha=0.3, axis="y")
plt.ylim(0, 12)
plt.xticks(
    x[: max(results["num_questions"] for results in llm_results.values())],
    rotation=90,
    fontsize=8,
)
plt.legend()
plt.tight_layout()
plt.savefig("scenario2_ranks.png")
plt.show()

# Visualization 3: Scenario 1 vs. Scenario 2 Scatter
plt.figure(figsize=(10, 6))
for idx, (llm_name, results) in enumerate(llm_results.items()):
    plt.scatter(
        results["similarity_scores"],
        results["scenario2_similarities"],
        s=100,
        alpha=0.5,
        color=colors[idx % len(colors)],
        label=llm_name,
    )
plt.plot([0.4, 1], [0.4, 1], "k--", label="Perfect Correlation")
plt.title("Similarity Scores: Scenario 1 vs. Scenario 2")
plt.xlabel("Scenario 1 Similarity")
plt.ylabel("Scenario 2 Similarity")
plt.grid(True, alpha=0.3)
plt.xlim(0.4, 1.0)
plt.ylim(0.4, 1.0)
plt.legend()
plt.text(0.75, 0.8, "Perfect Correlation", rotation=45)
plt.savefig("scenario1_vs_scenario2.png")
plt.show()

# ----------- Box Plot Comparing "Dilaksan/nla" vs Others -----------

plt.figure(figsize=(12, 7))

# Prepare data and labels
boxplot_data = []
labels = []

# If "Dilaksan/nla" is present, put it first for emphasis
if "Dilaksan/nla" in llm_results:
    boxplot_data.append(llm_results["Dilaksan/nla"]["similarity_scores"])
    labels.append("Dilaksan/nla")

# Append all other models except "Dilaksan/nla"
for llm_name, res in llm_results.items():
    if llm_name != "Dilaksan/nla":
        boxplot_data.append(res["similarity_scores"])
        labels.append(llm_name)

# Plot boxplot (updated parameter name)
box = plt.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, showfliers=True)

plt.title("Scenario 1 Cosine Similarity Comparison: Dilaksan/nla vs Other Models")
plt.ylabel("Cosine Similarity Score")
plt.xlabel("Model")
plt.ylim(0.4, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Color the "Dilaksan/nla" box differently for highlight
colors_box = ["lightcoral"] + ["lightgray"] * (len(labels) - 1)
for patch, color in zip(box["boxes"], colors_box):
    patch.set_facecolor(color)

plt.tight_layout()
plt.savefig("dilaksan_vs_others_boxplot.png")
plt.show()
