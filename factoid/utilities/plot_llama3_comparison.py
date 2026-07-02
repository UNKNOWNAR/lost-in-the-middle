import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(filepath, doc_count):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} does not exist.")
        return []
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Standardize field names
    standard_data = []
    for item in data:
        # Check different field name versions
        pos = item.get("Position of Answer (Gold Index)", item.get("position"))
        acc = item.get("Accuracy (%)", item.get("accuracy"))
        if pos is not None and acc is not None:
            standard_data.append({
                "Context Size (Documents)": f"{doc_count} Documents",
                "Position of Answer (Gold Index)": int(pos),
                "Accuracy (%)": float(acc)
            })
    return standard_data

def main():
    # Load all three json result files
    all_data = []
    all_data.extend(load_data("results/llama 3.1 8b/10documents_results/results_llama3.json", 10))
    all_data.extend(load_data("results/llama 3.1 8b/20documents_results/results_llama3_complete.json", 20))
    all_data.extend(load_data("results/llama 3.1 8b/30documents_results/results_llama3_30docs_final_run.json", 30))

    if not all_data:
        print("Error: No data loaded.")
        return

    df = pd.DataFrame(all_data)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    # Plot lines with markers
    sns.lineplot(
        data=df,
        x="Position of Answer (Gold Index)",
        y="Accuracy (%)",
        hue="Context Size (Documents)",
        marker="o",
        linewidth=2.5,
        markersize=8,
        palette="viridis"
    )

    # Add the closed-book baseline
    plt.axhline(y=23.8, color='r', linestyle='--', label='Closed-Book Baseline (23.8%)')

    plt.title("Llama 3.1 8B Evaluation Comparison\nAccuracy vs. Document Position (10, 20, and 30 Documents)", fontsize=14, pad=15)
    plt.ylim(0, 100)
    plt.xlim(-1, 30)
    plt.xticks(sorted(df["Position of Answer (Gold Index)"].unique()))
    plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()

    os.makedirs("results/llama 3.1 8b", exist_ok=True)
    output_image = "results/llama 3.1 8b/llama3_comparison_graph.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated comparison graph: {output_image}")

if __name__ == "__main__":
    main()
