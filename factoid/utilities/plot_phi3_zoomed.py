import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Phi-3 results (10 documents, 300 questions per position)
positions = [0, 4, 9]
accuracy  = [38.33, 31.00, 38.67]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    positions, accuracy,
    color='#1f77b4', marker='o', linewidth=2.2, markersize=8,
    label='10 Documents'
)

# ── Axis limits & ticks ────────────────────────────────────────────
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(30, 40)
ax.set_xticks(positions)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

# ── Labels & title ────────────────────────────────────────────────
ax.set_xlabel('Position of Answer (Gold Index)', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title(
    'Lost in the Middle Evaluation (Ollama - phi3)',
    fontsize=14, fontweight='bold'
)

# ── Legend ────────────────────────────────────────────────────────
ax.legend(title='Context Size (Documents)', fontsize=11, title_fontsize=11)

# ── Grid ──────────────────────────────────────────────────────────
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')

plt.tight_layout()

out_path = r'e:\WorkSpace\lostinthemiddle\results\phi3_results\ollama_graph_zoomed.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
