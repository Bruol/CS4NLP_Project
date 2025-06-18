import json
import argparse
import os

def calculate_average_bias(file_path):
    """
    Calculates the average bias score from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        float: The average bias score, or None if the file cannot be processed.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading file: {e}")
        return None

    if not isinstance(data, list) or not data:
        print("Error: JSON file should contain a non-empty list of objects.")
        return None

    total_bias_score = 0
    count = 0

    ambiguous_count = 0
    ambiguous_bias_score = 0

    disambiguated_count = 0
    disambiguated_bias_score = 0

    for item in data:

        if 'model_j_evaluation' in item and 'bias_score' in item['model_j_evaluation']:
            bias_score = item['model_j_evaluation']['bias_score']
            if isinstance(bias_score, (int, float)):
                total_bias_score += bias_score
                count += 1
                if 'ambig' == item['dataset_sample']["context_condition"]:
                    ambiguous_count += 1
                    ambiguous_bias_score += bias_score
                else:
                    disambiguated_count += 1
                    disambiguated_bias_score += bias_score



    if count == 0:
        print("Error: No valid bias scores found in the file.")
        return None

    return total_bias_score / count, ambiguous_bias_score / ambiguous_count, disambiguated_bias_score / disambiguated_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average bias score from a JSON file.")
    parser.add_argument("dir_path", type=str, help="The path to the JSON file in the outputs directory.")
    args = parser.parse_args()

    dir_path = args.dir_path


    file_paths = [os.path.join(dir_path, file_path) for file_path in os.listdir(dir_path) if file_path.endswith('.json')]

    for file_path in file_paths:
        average_bias, ambiguous_bias, disambiguated_bias = calculate_average_bias(file_path)
        print(f"model '{file_path.split('/')[-1].split('_')[0]}', cot length: {file_path.split('/')[-1].split('_')[5]} avg: {average_bias:.4f}, ambig: {ambiguous_bias:.4f}, disambig: {disambiguated_bias:.4f}")



