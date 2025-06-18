import json
import argparse
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Enhanced color palette with more professional colors
colors = {
    "red": (0.75,0.16,0.18),
    "green": (0.42,0.55,0.62),
    "blue":  (0.81,0.64,0.75)
}

map = {
    "deepseek-r1-distill-llama": {
        "label": "red",
        "name": "DeepSeek-R1-Distill-Llama-70B"
    },
    "gemini-2.5-flash": {
        "label": "green",
        "name": "Gemini-2.5-Flash"
    },
    "o4-mini": {
        "label": "blue",
        "name": "o4-mini"
    }
}

def analyze_and_plot_bias_by_model(directory_path, num_bins):
    """
    Analyzes bias scores for each model and plots the results.

    Args:
        directory_path (str): Path to the directory with BBQ JSON files.
        num_bins (int): Number of bins for thought length analysis.
    """
    if num_bins <= 0:
        print("Number of bins must be a positive integer.")
        return

    search_pattern = os.path.join(directory_path, '*bbq*.json')
    file_paths = sorted(glob.glob(search_pattern))

    if not file_paths:
        print(f"No BBQ JSON files found in '{directory_path}'.")
        return

    model_names = [os.path.basename(fp).split('_')[0] for fp in file_paths]

    # Set the style and create figure with higher quality settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['font.size'] = 16  # Base font size
    plt.rcParams['axes.labelsize'] = 18  # Label font size
    plt.rcParams['axes.titlesize'] = 20  # Title font size
    plt.rcParams['xtick.labelsize'] = 16  # X-tick labels
    plt.rcParams['ytick.labelsize'] = 16  # Y-tick labels
    plt.rcParams['legend.fontsize'] = 16  # Legend font size
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)

        model_name = os.path.basename(file_path).split('_')[0]
        print(f"\n--- Processing model: {model_name} ---")

        thought_lengths = []
        bias_scores = []

        for sample in data:
            bias_score = sample.get("model_j_evaluation", {}).get("bias_score")
            thought = sample.get("model_e_response", {}).get("thought", "")

            if thought and bias_score is not None:
                thought_lengths.append(len(thought))
                bias_scores.append(bias_score)

        if not thought_lengths:
            print(f"No thoughts found for model {model_name}.")
            continue
            
        thought_lengths = np.array(thought_lengths)
        bias_scores = np.array(bias_scores)
        
        bins = np.linspace(0, 100, num_bins + 1)
        bin_thresholds = np.percentile(thought_lengths, bins)
        
        bin_means = []
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        digitized = np.digitize(thought_lengths, bin_thresholds[1:-1])
        
        for i in range(num_bins):
            scores_in_bin = bias_scores[digitized == i]
            if scores_in_bin.size > 0:
                mean_score = np.mean(scores_in_bin)
                bin_means.append(mean_score)
            else:
                bin_means.append(np.nan)

        print(bin_means)
        print(bin_centers)
        # Plot with enhanced styling
        plt.plot(bin_centers, bin_means, 
                linestyle='-', 
                label=map[model_name]['name'], 
                color=colors[map[model_name]['label']], 
                linewidth=2.5,
                marker='o',
                markersize=8)
        
    # Enhance the plot appearance
    plt.xlabel('Normalized Reasoning Length', 
              fontsize=18, 
              labelpad=10)
    plt.ylabel('Normalized Bias', 
              fontsize=18, 
              labelpad=10)
    
    # Customize ticks
    plt.xticks(bins, fontsize=16)
    plt.yticks(fontsize=16)
    
    # Enhance legend
    plt.legend(fontsize=16, 
              frameon=True, 
              facecolor='white', 
              edgecolor='gray', 
              loc='upper left')
    
    # Enhance grid
    plt.grid(True, 
            which='major', 
            linestyle='--', 
            linewidth=0.8, 
            alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Set very high DPI for maximum quality
    plt.gcf().set_dpi(1200)
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    plot_path = os.path.join(script_dir, 'model_comparison_bias_plot.png')
    
    # Save with high quality settings
    plt.savefig(plot_path, 
                dpi=1200, 
                bbox_inches='tight', 
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none')
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot bias scores by model from experiment data.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing JSON files.")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of bins for thought length analysis.")
    args = parser.parse_args()
    
    analyze_and_plot_bias_by_model(args.directory_path, args.num_bins) 