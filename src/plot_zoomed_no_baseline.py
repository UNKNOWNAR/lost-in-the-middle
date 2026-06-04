import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
with open("results/10documents_results/results_llama3.json", "r") as f:
    data = json.load(f)
    
df = pd.DataFrame(data)

# Set up styling
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot
sns.lineplot(
    data=df, 
    x="Position of Answer (Gold Index)", 
    y="Accuracy (%)", 
    hue="Context Size (Documents)", 
    marker="o",
    palette="deep"
)

plt.title("Lost in the Middle Evaluation (Llama 3 - 10 Docs)", fontsize=14)
plt.ylim(40, 50)
plt.xticks(df["Position of Answer (Gold Index)"].unique())

plt.tight_layout()
plt.savefig("results/10documents_results/kaggle_graph_llama3_zoomed_no_baseline.png", dpi=300)
print("Zoomed graph generated successfully!")
