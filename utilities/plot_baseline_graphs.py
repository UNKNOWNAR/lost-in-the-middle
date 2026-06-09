import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph_with_baseline(json_path, output_image_path, title, ylim=(20, 60), show_baseline=True):
    print(f"Loading data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    plot = sns.lineplot(
        data=df, 
        x="Position of Answer (Gold Index)", 
        y="Accuracy (%)", 
        hue="Context Size (Documents)", 
        marker="o",
        palette="deep"
    )
    
    if show_baseline:
        # Adding the baseline horizontally at 23.8%
        plt.axhline(y=23.8, color='r', linestyle='--', label='Closed-Book Baseline (23.8%)')

    plt.title(title, fontsize=14)
    plt.ylim(ylim)
    plt.xticks(df["Position of Answer (Gold Index)"].unique())
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Saved: {output_image_path}")
    plt.close()

# Recreate 10-docs graph (zoomed)
plot_graph_with_baseline(
    "results/llama 3.1 8b/10documents_results/results_llama3.json",
    "results/llama 3.1 8b/10documents_results/kaggle_graph_llama3_zoomed.png",
    "Lost in the Middle Evaluation (Llama 3 - 10 Docs) [Zoomed]"
)

# Recreate 20-docs graph (zoomed, no baseline, 50-60%)
plot_graph_with_baseline(
    "results/llama 3.1 8b/20documents_results/results_llama3_complete.json",
    "results/llama 3.1 8b/20documents_results/lost_in_the_middle_20_docs_zoomed.png",
    "Lost in the Middle Evaluation (Llama 3 - 20 Docs) [Zoomed]",
    ylim=(50, 60),
    show_baseline=False
)

# Recreate 30-docs graph (full range 0-100, no zoomed title)
plot_graph_with_baseline(
    "results/llama 3.1 8b/30documents_results/results_llama3_30docs_final_run.json",
    "results/llama 3.1 8b/30documents_results/lost_in_the_middle_30_docs_with_baseline.png",
    "Lost in the Middle Evaluation (Llama 3 - 30 Docs)",
    ylim=(0, 100)
)

print("Done generating new graphs with the baseline!")
