import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

def calculate_thought_length(thought):
    """Calculates the length of a thought string by word count."""
    if not thought or not isinstance(thought, str):
        return 0
    return len(thought)

def print_results_table(results):
    """Prints the results in a pretty table."""
    headers = ["Model", "Low CoT", "Low Max", "Low Min", "Low StdDev", 
              "Medium CoT", "Medium Max", "Medium Min", "Medium StdDev",
              "High CoT", "High Max", "High Min", "High StdDev"]
    
    model_names = sorted(results.keys())
    
    table_data = []
    for model in model_names:
        row_data = [model]
        for cot in ['low', 'medium', 'high']:
            stats = results.get(model, {}).get(cot, {})
            avg = stats.get('average')
            max_val = stats.get('max') 
            min_val = stats.get('min')
            std = stats.get('std')
            
            row_data.extend([
                f"{avg:.2f}" if avg is not None else "N/A",
                f"{max_val:.2f}" if max_val is not None else "N/A", 
                f"{min_val:.2f}" if min_val is not None else "N/A",
                f"{std:.2f}" if std is not None else "N/A"
            ])
        table_data.append(row_data)

    if not table_data:
        print("No data to display.")
        return

    col_widths = [len(h) for h in headers]
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    
    separator_line = "-+-".join("-" * w for w in col_widths)
    print(separator_line)

    for row in table_data:
        row_line = " | ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths))
        print(row_line)

    print("\n\n")


def analyze_directory(directory):
    """
    Analyzes all JSON files in a directory to calculate average thought length.
    """
    json_files = glob(os.path.join(directory, '**', '*.json'), recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    results = {}

    for file_path in tqdm(json_files, desc="Analyzing files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        
        file_name = os.path.basename(file_path)
        try:
            model_name = file_name.split('_gpt-4o')[0]
            cot_length_part = file_name.split('_disabled_')[1]
            cot_length = cot_length_part.split('_')[0]
        except IndexError:
            continue

        if cot_length not in ['low', 'medium', 'high']:
            continue

        file_thought_lengths = []
        if isinstance(data, list):
            for sample in data:
                if (isinstance(sample, dict) and 
                    'model_e_response' in sample and 
                    isinstance(sample['model_e_response'], dict) and
                    'thought' in sample['model_e_response']):
                    
                    thought = sample['model_e_response']['thought']
                    length = calculate_thought_length(thought)
                    
                    file_thought_lengths.append(length)
        
        if file_thought_lengths:
            if model_name not in results:
                results[model_name] = {}
            if cot_length not in results[model_name]:
                results[model_name][cot_length] = {}
                
            results[model_name][cot_length]['average'] = np.mean(file_thought_lengths)
            results[model_name][cot_length]['max'] = np.max(file_thought_lengths)
            results[model_name][cot_length]['min'] = np.min(file_thought_lengths)
            results[model_name][cot_length]['std'] = np.std(file_thought_lengths)

    print_results_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average thought length from dataset samples in JSON files.")
    parser.add_argument("directory", type=str, help="The directory containing the JSON files.")
    args = parser.parse_args()

    analyze_directory(args.directory) 