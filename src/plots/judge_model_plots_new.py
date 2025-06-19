import matplotlib.pyplot as plt
import numpy as np

# Data setup
models = ['Deepseek-R1-Distill-Llama','Gemini-2.5-Flash', 'o4-mini']
x = np.arange(len(models))
bar_width = 0.35

# Define metrics: (Gemini Judge, GPT-4o Judge)
metrics = {
    'Normalized Bias Scores': [
        (0.15, 0.81), # DeepSeek
        (0.09, 0.77), # Gemini
        (0.37, 0.65), # o4-mini
    ],
}


col1 = (0.75, 0.16, 0.18)
col2 = (0.42, 0.55, 0.62)
col3 = (0.81, 0.64, 0.75)

colors = {
    'Gemini': col2,
    'GPT-4o': col3,
}

# Only one metric, so just use one plot
metric_name, values = list(metrics.items())[0]
gemini_vals = [v[0] for v in values]
gpt4o_vals = [v[1] for v in values]
gap = 0.05

fig, ax = plt.subplots(figsize=(10, 5))

bars1 = ax.bar(x - (bar_width/2 + gap/2), gemini_vals, width=bar_width, label='Gemini 2.5 Flash Judge',
               color=colors['Gemini'], alpha=1.0)
bars2 = ax.bar(x + (bar_width/2 + gap/2), gpt4o_vals, width=bar_width, label='GPT-4o Judge',
               color=colors['GPT-4o'], alpha=1.0)

# Add value labels
for i in range(len(models)):
    ax.text(x[i] - bar_width/2 - gap/2, gemini_vals[i] + 0.002, f"{gemini_vals[i]:.2f}",
            ha='center', va='bottom', fontsize=17, clip_on=False)
    ax.text(x[i] + bar_width/2 + gap/2, gpt4o_vals[i] + 0.002, f"{gpt4o_vals[i]:.2f}",
            ha='center', va='bottom', fontsize=17, clip_on=False)

ax.tick_params(axis='y', labelsize=17)
ax.set_ylabel(metric_name, fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=17)
ax.set_ylim(0.0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Legend at top center
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=17)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('src/plots/judge_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()