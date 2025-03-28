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
        
        # Extract steps and total losses
        steps = [entry['global_step'] for entry in loss_log if 'global_step' in entry]
        total_losses = [entry['total_loss'] for entry in loss_log if 'total_loss' in entry]
        avg_losses = [entry['avg_loss'] for entry in loss_log if 'avg_loss' in entry]
        
        # If old format logs, use step_loss or loss
        if not total_losses and 'step_loss' in loss_log[0]:
            # Calculate cumulative loss
            total_losses = []
            running_total = 0
            for entry in loss_log:
                running_total += entry['step_loss']
                total_losses.append(running_total)
        elif not total_losses and 'loss' in loss_log[0]:
            # Calculate cumulative loss
            total_losses = []
            running_total = 0
            for entry in loss_log:
                running_total += entry['loss']
                total_losses.append(running_total)
        
        # Plot total loss
        plt.plot(steps, total_losses, label=f"Rank {rank} (Total Loss)")
    
    plt.xlabel('Global Step')
    plt.ylabel('Total Loss')
    plt.title('Cumulative Training Loss Curves for Different Ranks')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    output_file = os.path.join(args.input_dir, "total_loss_curves.png")
    plt.savefig(output_file)
    print(f"Total loss curves saved to {output_file}")
    
    # Now create a plot for average loss
    plt.figure(figsize=(12, 8))
    
    for log_file in log_files:
        rank = int(os.path.basename(log_file).split("_")[2].split(".")[0])
        
        with open(log_file, 'r') as f:
            loss_log = json.load(f)
        
        # Extract steps and average losses
        steps = [entry['global_step'] for entry in loss_log if 'global_step' in entry]
        
        if 'avg_loss' in loss_log[0]:
            avg_losses = [entry['avg_loss'] for entry in loss_log]
        else:
            # Calculate average losses manually
            avg_losses = []
            running_total = 0
            for i, entry in enumerate(loss_log):
                loss_key = 'step_loss' if 'step_loss' in entry else 'loss'
                running_total += entry[loss_key]
                avg_losses.append(running_total / (i + 1))
        
        # Plot average loss
        plt.plot(steps, avg_losses, label=f"Rank {rank} (Avg Loss)")
    
    plt.xlabel('Global Step')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss Curves for Different Ranks')
    plt.legend()
    plt.grid(True)
    
    # Save the average loss plot
    avg_output_file = os.path.join(args.input_dir, "avg_loss_curves.png")
    plt.savefig(avg_output_file)
    print(f"Average loss curves saved to {avg_output_file}")
    
    # Also create a zoomed-in version for better comparison
    plt.figure(figsize=(12, 8))
    
    # Set y-axis limits to zoom in on the loss curves
    all_avg_losses = []
    for log_file in log_files:
        rank = int(os.path.basename(log_file).split("_")[2].split(".")[0])
        
        with open(log_file, 'r') as f:
            loss_log = json.load(f)
        
        # Skip the first few steps which might have high loss
        steps = [entry['global_step'] for entry in loss_log if 'global_step' in entry][5:]
        
        if 'avg_loss' in loss_log[0]:
            avg_losses = [entry['avg_loss'] for entry in loss_log][5:]
        else:
            # Calculate average losses manually
            running_total = sum([entry['step_loss' if 'step_loss' in entry else 'loss'] for entry in loss_log[:5]])
            avg_losses = []
            for i, entry in enumerate(loss_log[5:], 5):
                loss_key = 'step_loss' if 'step_loss' in entry else 'loss'
                running_total += entry[loss_key]
                avg_losses.append(running_total / (i + 1))
                
        all_avg_losses.extend(avg_losses)
        
        # Plot
        plt.plot(steps, avg_losses, label=f"Rank {rank}")
    
    # Set y-axis limits to zoom in
    if all_avg_losses:
        min_loss = np.min(all_avg_losses)
        max_loss = np.max(all_avg_losses)
        margin = (max_loss - min_loss) * 0.1  # Add 10% margin
        plt.ylim(max(0, min_loss - margin), max_loss + margin)
    
    plt.xlabel('Global Step')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss Curves for Different Ranks (Zoomed)')
    plt.legend()
    plt.grid(True)
    
    # Save the zoomed plot
    zoomed_output_file = os.path.join(args.input_dir, "avg_loss_curves_zoomed.png")
    plt.savefig(zoomed_output_file)
    print(f"Zoomed average loss curves saved to {zoomed_output_file}")

if __name__ == "__main__":
    main()