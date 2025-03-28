"""
Compare evaluation results across all ranks to verify consistency.
Usage: python compare_eval_results.py --input_dir /path/to/output_dir
"""

import os
import argparse
import numpy as np
from collections import defaultdict

def parse_eval_file(filepath):
    """Parse an evaluation results file and return metrics as a dictionary."""
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split(' = ')
                try:
                    # Try to convert to float
                    metrics[key] = float(value)
                except ValueError:
                    # If not a number, keep as string
                    metrics[key] = value
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across all ranks")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing rank-specific output directories")
    args = parser.parse_args()
    
    # Find all rank directories
    rank_dirs = []
    for item in os.listdir(args.input_dir):
        if item.startswith("rank_") and os.path.isdir(os.path.join(args.input_dir, item)):
            rank_dirs.append(os.path.join(args.input_dir, item))
    
    if not rank_dirs:
        print(f"No rank directories found in {args.input_dir}")
        return
    
    # Find all evaluation files
    eval_files = defaultdict(list)
    for rank_dir in rank_dirs:
        rank = os.path.basename(rank_dir).split("_")[1]
        for file in os.listdir(rank_dir):
            if file.startswith("eval_results_"):
                prefix = file.replace("eval_results_", "").replace(".txt", "")
                eval_files[prefix].append((rank, os.path.join(rank_dir, file)))
    
    if not eval_files:
        print(f"No evaluation result files found in rank directories")
        return
    
    # Compare metrics across ranks for each evaluation prefix
    for prefix, files in eval_files.items():
        print(f"\n=== Comparing results for {prefix} ===")
        
        # Parse all files
        rank_metrics = {}
        all_metric_keys = set()
        for rank, filepath in files:
            metrics = parse_eval_file(filepath)
            rank_metrics[rank] = metrics
            all_metric_keys.update(metrics.keys())
        
        # Compare metrics across ranks
        for metric in sorted(all_metric_keys):
            values = [rank_metrics[rank].get(metric, "N/A") for rank in sorted(rank_metrics.keys())]
            
            if all(isinstance(v, (int, float)) for v in values if v != "N/A"):
                # For numeric values, compute statistics
                numeric_values = [v for v in values if v != "N/A"]
                max_diff = max(numeric_values) - min(numeric_values) if numeric_values else 0
                mean = np.mean(numeric_values)
                std = np.std(numeric_values)
                
                consistency = "CONSISTENT" if max_diff < 1e-5 else "INCONSISTENT"
                
                print(f"Metric: {metric}")
                print(f"  Values across ranks: {values}")
                print(f"  Mean: {mean:.6f}, Std: {std:.6f}, Max Diff: {max_diff:.6f}")
                print(f"  Status: {consistency}")
            else:
                # For non-numeric values, just check if they're all the same
                unique_values = set(values)
                consistency = "CONSISTENT" if len(unique_values) == 1 else "INCONSISTENT"
                
                print(f"Metric: {metric}")
                print(f"  Values across ranks: {values}")
                print(f"  Status: {consistency}")

if __name__ == "__main__":
    main()