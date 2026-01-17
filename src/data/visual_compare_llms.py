import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sns.set(style="whitegrid")  # Better plot style

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths for ground truth and LLM outputs
answer_path = r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\book\answers.csv"
llm_paths = {
    "deepseek-r1-0528": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\deepseek\deepseek-r1-0528-1.csv",
    "gemini-2.5-pro": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\google\gemini-2.5-pro1.csv",
    "grok-3-mini": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\x-ai\grok-3-mini-beta1.csv",
    "o4-mini-high": r"C:\Users\MY PC\Desktop\research0\Investigate-LLMs-and-Develop-a-Model-for-Numerical-Linear-Algebra\data\answers\openai\o4-mini1.csv",
}

# Load ground truth
ground_truth_df = pd.read_csv(answer_path)
ground_truth_answers = ground_truth_df["answer"].tolist()
ground_truth_embeddings = model.encode(ground_truth_answers, convert_to_tensor=False)

llm_results = {}

for llm_name, llm_path in llm_paths.items():
    llm_df = pd.read_csv(llm_path)
    if "answer" not in llm_df.columns:
        print(f"Skipping {llm_name}: missing 'answer' column")
        continue

    llm_answers = llm_df["answer"].tolist()
    llm_embeddings = model.encode(llm_answers, convert_to_tensor=False)

    num_questions = min(100, len(llm_embeddings), len(ground_truth_embeddings))

    # Scenario 1: Cosine similarity between matching pairs
    similarity_scores = [
        cosine_similarity([ground_truth_embeddings[i]], [llm_embeddings[i]])[0][0]
        for i in range(num_questions)
    ]

    # Scenario 2: Rank of correct GT answer among all
    ranks = []
    scenario2_similarities = []
    for i in range(num_questions):
        sims = cosine_similarity([llm_embeddings[i]], ground_truth_embeddings)[0]
        rank = np.where(np.argsort(-sims) == i)[0][0] + 1
        ranks.append(min(rank, 11))
        scenario2_similarities.append(sims[i])

    llm_results[llm_name] = {
        "similarity_scores": similarity_scores,
        "scenario2_similarities": scenario2_similarities,
        "ranks": ranks,
        "num_questions": num_questions,
        "mean_s1": np.mean(similarity_scores),
        "median_s1": np.median(similarity_scores),
        "std_s1": np.std(similarity_scores),
        "mean_s2": np.mean(scenario2_similarities),
        "median_s2": np.median(scenario2_similarities),
        "std_s2": np.std(scenario2_similarities),
        "mean_rank": np.mean(ranks),
        "median_rank": np.median(ranks),
        "std_rank": np.std(ranks),
    }

# Print detailed summary
print("\nDetailed LLM Similarity & Rank Summary:\n")
for name, res in llm_results.items():
    print(f"{name}:")
    print(
        f"  Scenario 1 Similarity - Mean: {res['mean_s1']:.4f}, Median: {res['median_s1']:.4f}, Std: {res['std_s1']:.4f}"
    )
    print(
        f"  Scenario 2 Similarity - Mean: {res['mean_s2']:.4f}, Median: {res['median_s2']:.4f}, Std: {res['std_s2']:.4f}"
    )
    print(
        f"  Rank (lower better) - Mean: {res['mean_rank']:.2f}, Median: {res['median_rank']:.2f}, Std: {res['std_rank']:.2f}\n"
    )

# Perform pairwise t-tests between LLMs for Scenario 1 similarity
llm_names = list(llm_results.keys())
print("Pairwise t-tests for Scenario 1 Similarity:\n")
for i in range(len(llm_names)):
    for j in range(i + 1, len(llm_names)):
        data1 = llm_results[llm_names[i]]["similarity_scores"]
        data2 = llm_results[llm_names[j]]["similarity_scores"]
        t_stat, p_val = ttest_rel(data1, data2)
        print(f"{llm_names[i]} vs {llm_names[j]}: t={t_stat:.3f}, p={p_val:.4f}")

# Export all results to CSV
rows = []
for llm_name, res in llm_results.items():
    for i in range(res["num_questions"]):
        rows.append(
            {
                "LLM": llm_name,
                "Question_ID": i + 1,
                "Scenario_1_Similarity": res["similarity_scores"][i],
                "Scenario_2_Similarity": res["scenario2_similarities"][i],
                "Rank": res["ranks"][i],
            }
        )
results_df = pd.DataFrame(rows)
results_df.to_csv("llm_analysis_results.csv", index=False)
print("\n✅ Results saved to 'llm_analysis_results.csv'")

# --- Visualization 1: Scenario 1 Histogram with KDE ---
plt.figure(figsize=(12, 6))
bins = np.linspace(0.5, 1.0, 21)  # finer bins
for idx, (llm, res) in enumerate(llm_results.items()):
    sns.histplot(
        res["similarity_scores"],
        bins=bins,
        kde=True,
        stat="density",
        label=llm,
        element="step",
        fill=False,
        linewidth=1.5,
        alpha=0.8,
    )
plt.title("Scenario 1: Cosine Similarity Distribution per LLM")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Density")
plt.xlim(0.5, 1.0)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("scenario1_similarity_kde.png")
plt.show()

# --- Visualization 2: Rank per Question Bar Chart ---
plt.figure(figsize=(14, 6))
bar_width = 0.18
x = np.arange(1, 101)
colors = ["tomato", "forestgreen", "steelblue", "purple"]
for idx, (llm, res) in enumerate(llm_results.items()):
    plt.bar(
        x[: res["num_questions"]] + idx * bar_width,
        res["ranks"],
        width=bar_width,
        label=llm,
        color=colors[idx % len(colors)],
        alpha=0.85,
    )
plt.title("Scenario 2: Rank of Correct Answer per Question")
plt.xlabel("Question ID")
plt.ylabel("Rank (Lower is Better)")
plt.ylim(0, 12)
plt.grid(axis="y", alpha=0.3)
plt.xticks(x[::5], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("scenario2_ranks_bar_chart.png")
plt.show()

# --- Visualization 3: Scatter Plot S1 vs S2 Similarity ---
plt.figure(figsize=(10, 6))
for idx, (llm, res) in enumerate(llm_results.items()):
    plt.scatter(
        res["similarity_scores"],
        res["scenario2_similarities"],
        label=llm,
        s=70,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
        color=colors[idx % len(colors)],
    )
plt.plot([0.5, 1], [0.5, 1], "r--", label="Perfect Correlation")
plt.xlabel("Scenario 1 Similarity")
plt.ylabel("Scenario 2 Similarity")
plt.title("Scenario 1 vs Scenario 2 Cosine Similarity")
plt.xlim(0.5, 1.0)
plt.ylim(0.5, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("scenario1_vs_scenario2_scatter.png")
plt.show()

# --- Visualization 4: 3D Scatter Plot (S1, S2, Rank) ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

for idx, (llm, res) in enumerate(llm_results.items()):
    ax.scatter(
        res["similarity_scores"],
        res["scenario2_similarities"],
        res["ranks"],
        label=llm,
        alpha=0.7,
        s=50,
        color=colors[idx % len(colors)],
        edgecolors="k",
        linewidth=0.5,
    )

ax.set_xlabel("Scenario 1 Similarity")
ax.set_ylabel("Scenario 2 Similarity")
ax.set_zlabel("Rank (Lower is Better)")
ax.set_title("3D Comparison: Similarities and Rank per Question")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.5, 1.0)
ax.set_zlim(0, 12)
ax.legend()
plt.tight_layout()
plt.savefig("3d_similarity_vs_rank.png")
plt.show()
