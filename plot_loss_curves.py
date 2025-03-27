#!/usr/bin/env python3
"""
Plot loss curves from the distributed training logs.
Usage: python plot_loss_curves.py --input_dir /path/to/output_dir
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from distributed training logs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing loss log files")
    args = parser.parse_args()
    
    # Find all loss log files
    log_files = []
    for file in os.listdir(args.input_dir):
        if file.startswith("loss_log_rank_") and file.endswith(".json"):
            log_files.append(os.path.join(args.input_dir, file))
    
    if not log_files:
        print(f"No loss log files found in {args.input_dir}")
        return
    
    # Sort log files by rank
    log_files.sort()
    
    plt.figure(figsize=(12, 8))
    
    # Load and plot each log file
    for log_file in log_files:
        rank = int(os.path.basename(log_file).split("_")[2].split(".")[0])
        
        with open(log_file, 'r') as f:
            loss_log = json.load(f)
        
        # Extract steps and losses
        steps = [entry['global_step'] for entry in loss_log if 'global_step' in entry]
        losses = [entry['loss'] for entry in loss_log]
        
        # Plot
        plt.plot(steps, losses, label=f"Rank {rank}")
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves for Different Ranks')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = os.path.join(args.input_dir, "loss_curves.png")
    plt.savefig(output_file)
    print(f"Loss curves saved to {output_file}")
    
    # Also create a zoomed-in version for better comparison
    plt.figure(figsize=(12, 8))
    
    # Set y-axis limits to zoom in on the loss curves
    all_losses = []
    for log_file in log_files:
        rank = int(os.path.basename(log_file).split("_")[2].split(".")[0])
        
        with open(log_file, 'r') as f:
            loss_log = json.load(f)
        
        # Skip the first few steps which might have high loss
        steps = [entry['global_step'] for entry in loss_log if 'global_step' in entry][5:]
        losses = [entry['loss'] for entry in loss_log][5:]
        all_losses.extend(losses)
        
        # Plot
        plt.plot(steps, losses, label=f"Rank {rank}")
    
    # Set y-axis limits to zoom in
    if all_losses:
        min_loss = np.min(all_losses)
        max_loss = np.max(all_losses)
        margin = (max_loss - min_loss) * 0.1  # Add 10% margin
        plt.ylim(max(0, min_loss - margin), max_loss + margin)
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves for Different Ranks (Zoomed)')
    plt.legend()
    plt.grid(True)
    
    # Save the zoomed plot
    zoomed_output_file = os.path.join(args.input_dir, "loss_curves_zoomed.png")
    plt.savefig(zoomed_output_file)
    print(f"Zoomed loss curves saved to {zoomed_output_file}")

if __name__ == "__main__":
    main()