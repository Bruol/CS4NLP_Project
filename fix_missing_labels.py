#!/usr/bin/env python3
"""
Script to add missing target labels to result files by matching question-context pairs
with the original BBQ dataset.
"""

import json
import os
import sys
from datasets import load_dataset, concatenate_datasets
from typing import Dict, Any, List
import argparse

def load_bbq_dataset():
    """Load the complete BBQ dataset for label lookup."""
    print("Loading BBQ dataset...")
    dataset_dict = load_dataset("Elfsong/BBQ")
    splits = ["age", "disability_status", "gender_identity", "nationality", 
              "physical_appearance", "race_ethnicity", "race_x_gender", 
              "race_x_ses", "religion", "ses", "sexual_orientation"]
    
    # Concatenate all splits
    full_dataset = dataset_dict[splits[0]]
    for split in splits[1:]:
        full_dataset = concatenate_datasets([full_dataset, dataset_dict[split]])
    
    print(f"Loaded {len(full_dataset)} samples from BBQ dataset")
    return full_dataset

def create_lookup_dict(dataset) -> Dict[str, Dict[str, Any]]:
    """
    Create a lookup dictionary using context + question as key.
    """
    print("Creating lookup dictionary...")
    lookup = {}
    
    for sample in dataset:
        # Create a unique key from context and question
        key = f"{sample['context']}|||{sample['question']}"
        lookup[key] = {
            'answer_label': sample['answer_label'],  # This is the target label from BBQ
            'target_label': sample['target_label'],  # Fallback to answer_label
            'context_condition': sample['context_condition'],
            'question_polarity': sample['question_polarity'],
            'category': sample['category'],
            'ans0': sample['ans0'],
            'ans1': sample['ans1'], 
            'ans2': sample['ans2']
        }
    
    print(f"Created lookup dictionary with {len(lookup)} entries")
    return lookup

def fix_result_file(file_path: str, lookup_dict: Dict[str, Dict[str, Any]]) -> bool:
    """
    Fix missing labels in a single result file.
    Returns True if any changes were made.
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    changes_made = False
    fixed_count = 0
    
    for i, result in enumerate(results):
        # Skip if this is just an empty entry or missing dataset_sample
        if not result or 'dataset_sample' not in result:
            continue
            
        dataset_sample = result['dataset_sample']
        
        # Check if answer_label or target_label is missing
        if ('answer_label' not in dataset_sample or dataset_sample.get('answer_label') is None or
            'target_label' not in dataset_sample or dataset_sample.get('target_label') is None):
            # Try to find it in the lookup
            context = dataset_sample.get('context', '')
            question = dataset_sample.get('question', '')
            
            if context and question:
                key = f"{context}|||{question}"
                
                if key in lookup_dict:
                    # Add the missing information
                    lookup_data = lookup_dict[key]
                    
                    if 'answer_label' not in dataset_sample or dataset_sample.get('answer_label') is None:
                        dataset_sample['answer_label'] = lookup_data['answer_label']
                    if 'target_label' not in dataset_sample or dataset_sample.get('target_label') is None:
                        dataset_sample['target_label'] = lookup_data['target_label']
                    
                    # Also add other missing fields if they don't exist
                    if 'context_condition' not in dataset_sample:
                        dataset_sample['context_condition'] = lookup_data['context_condition']
                    if 'question_polarity' not in dataset_sample:
                        dataset_sample['question_polarity'] = lookup_data['question_polarity']
                    if 'category' not in dataset_sample:
                        dataset_sample['category'] = lookup_data['category']
                    
                    changes_made = True
                    fixed_count += 1
                    print(f"  Fixed sample {i}: added answer_label = {lookup_data['answer_label']}, target_label = {lookup_data['target_label']}")
                else:
                    print(f"  Warning: Could not find match for sample {i}")
            else:
                print(f"  Warning: Sample {i} missing context or question")
    
    if changes_made:
        # Create backup
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"  Created backup: {backup_path}")
        
        # Write fixed file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"  Fixed {fixed_count} samples in {file_path}")
    else:
        print(f"  No changes needed for {file_path}")
    
    return changes_made

def main():
    parser = argparse.ArgumentParser(description='Fix missing target labels in result files')
    parser.add_argument('--output-dir', default='outputs', 
                       help='Directory containing result files to fix')
    parser.add_argument('--file-pattern', default='.json',
                       help='File pattern to match (default: .json)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    
    args = parser.parse_args()
    
    # Load the BBQ dataset for lookup
    bbq_dataset = load_bbq_dataset()
    lookup_dict = create_lookup_dict(bbq_dataset)
    
    # Find all result files
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} does not exist")
        return
    
    result_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(args.file_pattern) and not file.endswith('.backup'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print(f"No files found matching pattern '{args.file_pattern}' in {output_dir}")
        return
    
    print(f"Found {len(result_files)} files to process")
    
    total_fixed = 0
    for file_path in result_files:
        if args.dry_run:
            print(f"Would process: {file_path}")
        else:
            if fix_result_file(file_path, lookup_dict):
                total_fixed += 1
    
    if args.dry_run:
        print(f"Dry run complete. Would process {len(result_files)} files.")
    else:
        print(f"Processing complete. Fixed {total_fixed} files.")

if __name__ == "__main__":
    main()
