import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Artifact directory for saving the images
OUTPUT_DIR = r"C:\Users\amiar\.gemini\antigravity-ide\brain\187fbd4c-3682-48f3-9b86-a66cec767b8e"

def parse_results():
    path = r"e:\WorkSpace\lostinthemiddle\litm_pipeline\results\reports\RESULTS_k60.md"
    vital_scores = []
    okay_scores = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if "| **k60_" in line:
                # Example line: | **k60_1** | 1-3 (Primacy) | **51.45%** | 26.35% | **100/100 (100.0%) ✅** |
                parts = line.split('|')
                if len(parts) >= 5:
                    v_str = parts[3].replace('*', '').replace('%', '').strip()
                    o_str = parts[4].replace('%', '').strip()
                    try:
                        vital_scores.append(float(v_str))
                        okay_scores.append(float(o_str))
                    except ValueError:
                        pass
                        
    return vital_scores, okay_scores

def plot_graph(scores, title, filename, y_min, y_max, color, ylabel):
    plt.figure(figsize=(10, 6))
    
    # X-axis represents the 10 conditions (1 to 10)
    x_positions = range(1, 11)
    
    # Plotting the line
    plt.plot(x_positions, scores, marker='o', linestyle='-', linewidth=2, markersize=8, color=color)
    
    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Context Placement Condition (k60_1 to k60_10)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x_positions, [f"C{i}" for i in range(1, 11)])
    
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add value labels on the points
    for i, score in enumerate(scores):
        plt.text(x_positions[i], score + ((y_max - y_min) * 0.02), f"{score:.1f}%", 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

def main():
    vital, okay = parse_results()
    
    if len(vital) != 10 or len(okay) != 10:
        print(f"Error parsing results. Found {len(vital)} vital scores and {len(okay)} okay scores.")
        return
        
    print(f"Vital Scores: {vital}")
    print(f"Okay Scores: {okay}")
    
    # We want to save to the project's permanent figures directory
    project_figures = Path(r"e:\WorkSpace\lostinthemiddle\litm_pipeline\results\figures\k=60")
    project_figures.mkdir(parents=True, exist_ok=True)
    
    # 1. Vital Recall (Zoomed)
    plot_graph(vital, "Vital Recall across Conditions", str(project_figures / "vital_recall_zoomed.png"), 35, 55, '#1f77b4', "Vital Recall (%)")
    
    # 2. Vital Recall (Full 0-100)
    plot_graph(vital, "Vital Recall across Conditions", str(project_figures / "vital_recall_full.png"), 0, 100, '#1f77b4', "Vital Recall (%)")
    
    # 3. Okay Recall (Zoomed)
    plot_graph(okay, "Okay Recall across Conditions", str(project_figures / "okay_recall_zoomed.png"), 10, 30, '#ff7f0e', "Okay Recall (%)")
    
    # 4. Okay Recall (Full 0-100)
    plot_graph(okay, "Okay Recall across Conditions", str(project_figures / "okay_recall_full.png"), 0, 100, '#ff7f0e', "Okay Recall (%)")
    
    print("Graphs generated successfully.")

if __name__ == "__main__":
    main()
