import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    # Llama-2-70b-chat accuracy points on 10 and 30 documents QA from the paper
    data_10 = [
        {"Position": 0, "Accuracy (%)": 64.0},
        {"Position": 4, "Accuracy (%)": 52.0},
        {"Position": 9, "Accuracy (%)": 61.0}
    ]

    data_30 = [
        {"Position": 0, "Accuracy (%)": 53.0},
        {"Position": 14, "Accuracy (%)": 35.0},
        {"Position": 29, "Accuracy (%)": 44.0}
    ]

    df_10 = pd.DataFrame(data_10)
    df_30 = pd.DataFrame(data_30)

    # Plot 10 Documents
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_10, x="Position", y="Accuracy (%)", marker="o", linewidth=2.5, markersize=8, color="blue", label="Llama-2-70b-chat (10 Docs)")
    plt.title("Original Paper Results: Llama-2-70b-chat\n(10-Document QA Setting)", fontsize=14, pad=15)
    plt.ylim(0, 100)
    plt.xlim(-1, 10)
    plt.xticks([0, 4, 9])
    plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    output_image_10 = "results/paper_original_10docs.png"
    plt.tight_layout()
    plt.savefig(output_image_10, dpi=300)
    plt.close()
    print(f"Successfully generated 10-docs paper results graph: {output_image_10}")

    # Plot 30 Documents
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_30, x="Position", y="Accuracy (%)", marker="o", linewidth=2.5, markersize=8, color="red", label="Llama-2-70b-chat (30 Docs)")
    plt.title("Original Paper Results: Llama-2-70b-chat\n(30-Document QA Setting)", fontsize=14, pad=15)
    plt.ylim(0, 100)
    plt.xlim(-1, 30)
    plt.xticks([0, 14, 29])
    plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend()

    output_image_30 = "results/paper_original_30docs.png"
    plt.tight_layout()
    plt.savefig(output_image_30, dpi=300)
    plt.close()
    print(f"Successfully generated 30-docs paper results graph: {output_image_30}")

if __name__ == "__main__":
    main()
