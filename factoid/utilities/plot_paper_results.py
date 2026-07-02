import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # Original data extracted from lost-in-the-middle/EXPERIMENTS.md
    # 20-document QA setting
    data = [
        # MPT-30B-Instruct
        {"Model": "MPT-30B-Instruct", "Position": 0, "Accuracy (%)": 53.67},
        {"Model": "MPT-30B-Instruct", "Position": 4, "Accuracy (%)": 51.75},
        {"Model": "MPT-30B-Instruct", "Position": 9, "Accuracy (%)": 52.17},
        {"Model": "MPT-30B-Instruct", "Position": 14, "Accuracy (%)": 52.66},
        {"Model": "MPT-30B-Instruct", "Position": 19, "Accuracy (%)": 56.23},

        # LongChat-13B-16k
        {"Model": "LongChat-13B-16k", "Position": 0, "Accuracy (%)": 68.59},
        {"Model": "LongChat-13B-16k", "Position": 4, "Accuracy (%)": 57.40},
        {"Model": "LongChat-13B-16k", "Position": 9, "Accuracy (%)": 55.33},
        {"Model": "LongChat-13B-16k", "Position": 14, "Accuracy (%)": 52.50},
        {"Model": "LongChat-13B-16k", "Position": 19, "Accuracy (%)": 55.03},

        # Llama-2-70b-chat-hf
        {"Model": "Llama-2-70b-chat", "Position": 0, "Accuracy (%)": 56.77},
        {"Model": "Llama-2-70b-chat", "Position": 4, "Accuracy (%)": 53.32},
        {"Model": "Llama-2-70b-chat", "Position": 9, "Accuracy (%)": 54.08},
        {"Model": "Llama-2-70b-chat", "Position": 14, "Accuracy (%)": 59.66},
        {"Model": "Llama-2-70b-chat", "Position": 19, "Accuracy (%)": 69.49},
    ]

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Plot lines with markers for each model
    sns.lineplot(
        data=df,
        x="Position",
        y="Accuracy (%)",
        hue="Model",
        marker="o",
        linewidth=2.5,
        markersize=8,
        palette="Set1"
    )

    plt.title("Original 'Lost in the Middle' Paper Results\n(20-Document QA Setting)", fontsize=14, pad=15)
    plt.ylim(40, 80)
    plt.xticks([0, 4, 9, 14, 19])
    plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(title="Model Evaluated")

    os.makedirs("results", exist_ok=True)
    output_image = "results/paper_original_results.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated paper results graph: {output_image}")

if __name__ == "__main__":
    main()
