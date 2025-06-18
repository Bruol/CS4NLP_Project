import re
import pandas as pd
import os

def parse_results(file_path):
    """
    Parses the analysis results file to extract model performance metrics.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    results = []
    blocks = content.split('----------------------------------------')
    
    for block in blocks:
        if not block.strip():
            continue

        model_match = re.search(r"=== Processing: (.*?)_gpt-4o_bbq_2750_disabled_(high|low|medium)_.*\.json ===", block)
        if not model_match:
            continue
            
        model_name_full, cot_length = model_match.groups()

        if 'deepseek' in model_name_full:
            model_name = 'deepseek'
        elif 'gemini' in model_name_full:
            model_name = 'gemini'
        elif 'o4-mini' in model_name_full:
            model_name = 'o4-mini'
        else:
            continue

        metrics_match = re.search(r"Acc_amb: ([-.\d]+), Acc_dis: ([-.\d]+), Bias_amb: ([-.\d]+), Bias_dis: ([-.\d]+)", block)
        if not metrics_match:
            continue
            
        acc_amb, acc_dis, bias_amb, bias_dis = metrics_match.groups()

        results.append({
            'model': model_name,
            'cot_length': cot_length,
            'acc_amb': float(acc_amb),
            'acc_dis': float(acc_dis),
            'bias_amb': float(bias_amb),
            'bias_dis': float(bias_dis)
        })
        
    return pd.DataFrame(results)

def display_results_table(df):
    """
    Displays the parsed performance metrics in a table.
    """
    if df.empty:
        print("DataFrame is empty. No table will be generated.")
        return
        
    cot_order = ['low', 'medium', 'high']
    df['cot_length'] = pd.Categorical(df['cot_length'], categories=cot_order, ordered=True)
    df_sorted = df.sort_values(['model', 'cot_length'])

    print("\n--- Individual Model Performance ---")
    print(df_sorted.to_string())

def display_average_results(df):
    """
    Calculates and displays the average results across all models for each CoT length.
    """
    if df.empty:
        print("DataFrame is empty. No average results to display.")
        return

    # Exclude 'model' column for averaging
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    average_results = df.groupby('cot_length')[numeric_cols].mean().reset_index()

    print("\nAverage Results Across All Models:")
    print(average_results)

    return average_results

if __name__ == '__main__':
    # Correct the file path to be relative to the script's location
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'cot_length_analysis_results.txt')
    
    if os.path.exists(file_path):
        
        df = parse_results(file_path)
        display_results_table(df)
        avg_df = display_average_results(df)
    else:
        print(f"Error: File not found at {file_path}") 