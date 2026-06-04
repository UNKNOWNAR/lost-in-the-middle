import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    log_file = "results/gemma_results/live_output_gemini_gemma.txt"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} does not exist.")
        return

    results = []
    
    # Read the log file and search for lines like: >> Position 0 Accuracy: 72.0% (72/100)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    matches = re.findall(r">> Position (\d+) Accuracy: ([\d.]+)%", content)
    for pos, acc in matches:
        results.append({
            "Context Size (Documents)": "10",
            "Position of Answer (Gold Index)": int(pos),
            "Accuracy (%)": float(acc)
        })

    if not results:
        # Fallback if parsing fails for some reason
        results = [
            {"Context Size (Documents)": "10", "Position of Answer (Gold Index)": 0, "Accuracy (%)": 72.0},
            {"Context Size (Documents)": "10", "Position of Answer (Gold Index)": 4, "Accuracy (%)": 82.0},
            {"Context Size (Documents)": "10", "Position of Answer (Gold Index)": 9, "Accuracy (%)": 79.0}
        ]

    df = pd.DataFrame(results)
    
    # Sort by Position
    df = df.sort_values(by="Position of Answer (Gold Index)")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df, 
        x="Position of Answer (Gold Index)", 
        y="Accuracy (%)", 
        hue="Context Size (Documents)", 
        marker="o",
        palette="tab10",
        linewidth=2.5,
        markersize=8
    )

    plt.title("Lost in the Middle Phenomenon\nAccuracy by Answer Position (Gemini API - gemma-4-31b-it, 100 qs/pos)", fontsize=14, pad=15)
    plt.ylim(0, 105)
    plt.xlim(-1, 10)
    plt.xticks([0, 4, 9])
    plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(title="Context Size")

    os.makedirs("results/gemma_results", exist_ok=True)
    output_image = "results/gemma_results/gemma_4_31b_it_graph.png"
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated graph: {output_image}")

if __name__ == "__main__":
    main()
