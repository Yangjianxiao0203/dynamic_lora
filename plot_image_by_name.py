import json
import os

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

if not os.path.exists('./images/a.png'):
    os.makedirs('./images', exist_ok=True)

def plot_singular_values_and_loss(jsonl_file):
    # Read the data from the JSONL file
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)

    # Organize data by 'name'
    data_by_name = defaultdict(list)
    max_step = 0
    for obj in data:
        name = obj['name']
        epoch = obj.get('epoch', 0)
        step_num = obj['step_num']
        loss = obj.get('loss', 0)  # Extract loss value
        if step_num > max_step:
            max_step = step_num
        ranks = obj['ranks']
        data_by_name[name].append({'epoch': epoch, 'step_num': step_num, 'loss': loss, 'ranks': ranks})

    # Process and plot data for each 'name'
    for name, data_points in data_by_name.items():
        # Sort data_points by epoch and step_num
        data_points.sort(key=lambda x: (x['epoch'], x['step_num']))
        total_steps = []
        rank_values = [[] for _ in range(32)]  # Initialize list for each of the 32 ranks
        losses = []  # List to hold loss values

        for dp in data_points:
            epoch = dp['epoch']
            step_num = dp['step_num']
            loss = dp['loss']
            ranks = dp['ranks']

            total_step = (max_step + 1) * epoch + step_num
            total_steps.append(total_step)
            losses.append(loss)

            # Append each rank value to the corresponding list
            for i in range(32):
                if i < len(ranks):
                    rank_values[i].append(ranks[i])
                else:
                    rank_values[i].append(0)  # If ranks list is shorter than 32, append 0

        # Plot the rank values over total_steps
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot singular values
        for i in range(32):
            ax1.plot(total_steps, np.log(rank_values[i]), label=f'Rank {i + 1}')
        ax1.set_xlabel('Total Step')
        ax1.set_ylabel('Log Singular Value')
        ax1.set_title(f'Singular Values and Loss for {name}')

        # Add a second y-axis for loss
        ax2 = ax1.twinx()
        ax2.plot(total_steps, losses, 'r--', label='Loss', linewidth=2)  # Plot loss in red dashed line
        ax2.set_ylabel('Loss')

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.15, 1.05))

        plt.tight_layout()
        file_name = f'./images/{name}_with_loss.png'
        plt.savefig(file_name, transparent=False)
        print(f"{name} save successfully with loss plot")
        plt.close()


if __name__ == "__main__":
    plot_singular_values_and_loss('records/qwen-0_5.jsonl')
