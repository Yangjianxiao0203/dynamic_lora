import json
import os

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

if not os.path.exists('./images/a.png'):
    os.makedirs('./images', exist_ok=True)

def plot_singular_values(jsonl_file):
    # Read the data from the JSONL file
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)

    # Organize data by 'name'
    data_by_name = defaultdict(list)
    for obj in data:
        name = obj['name']
        epoch = obj.get('epoch', 0)
        step_num = obj['step_num']
        loss = obj['loss']  # Extract the loss value
        ranks = obj['ranks']
        data_by_name[name].append({'epoch': epoch, 'step_num': step_num, 'loss': loss, 'ranks': ranks})

    # Process and plot data for each 'name' and 'epoch'
    for name, data_points in data_by_name.items():
        # Group data_points by 'epoch'
        data_by_epoch = defaultdict(list)
        for dp in data_points:
            epoch = dp['epoch']
            data_by_epoch[epoch].append(dp)

        # For each epoch, process and plot data
        for epoch, epoch_data in data_by_epoch.items():
            # Sort epoch_data by step_num
            epoch_data.sort(key=lambda x: x['step_num'])
            total_steps = []
            rank_values = [[] for _ in range(32)]  # Initialize list for each of the 32 ranks
            losses = []  # List to hold loss values

            for dp in epoch_data:
                step_num = dp['step_num']
                ranks = dp['ranks']
                loss = dp['loss']  # Get the loss value

                total_steps.append(step_num)
                losses.append(loss)

                # Append each rank value to the corresponding list
                for i in range(32):
                    if i < len(ranks):
                        rank_values[i].append(ranks[i])
                    else:
                        # If ranks list is shorter than 32, append 0
                        rank_values[i].append(0)

            # Plot the rank values and loss over total_steps
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()

            # Plot ranks
            for i in range(32):
                plt.plot(total_steps, np.log(rank_values[i]), label=f'Rank {i + 1}')

            # Create a second y-axis for the loss
            ax2 = ax1.twinx()
            ax2.plot(total_steps, losses, color='red', label='Loss', linestyle='--', linewidth=2)

            ax1.xaxis.set_major_locator(plt.MultipleLocator(1))  # X-axis major ticks at every 1 step
            ax1.yaxis.set_major_locator(plt.MultipleLocator(1))  # Y-axis major ticks at every 1 unit in log scale

            plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)
            plt.xlabel('Step Number')
            plt.ylabel('Log Singular Value')
            ax2.set_ylabel('Loss')  # Set label for the second y-axis
            plt.title(f'Singular Values and Loss for {name} - Epoch {epoch}')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            file_name = f"{name}_epoch_{epoch}.png"
            plt.savefig(f'./images/{file_name}', transparent=False)
            print(f"{file_name} save successfully")
            plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    plot_singular_values('records/qwen-0_5.jsonl')
