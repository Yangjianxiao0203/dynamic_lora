import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter


def plot_singular_values(jsonl_file):
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
        if step_num > max_step:
            max_step = step_num
        ranks = obj['ranks']
        data_by_name[name].append({'epoch': epoch, 'step_num': step_num, 'ranks': ranks})

    # Process and plot data for each 'name'
    for name, data_points in data_by_name.items():
        # Sort data_points by epoch and step_num
        data_points.sort(key=lambda x: (x['epoch'], x['step_num']))
        total_steps = []
        rank_values = [[] for _ in range(32)]  # Initialize list for each of the 32 ranks

        for dp in data_points:
            epoch = dp['epoch']
            step_num = dp['step_num']
            ranks = dp['ranks']

            total_step = (max_step + 1) * epoch + step_num
            total_steps.append(total_step)

            # Append each rank value to the corresponding list
            for i in range(32):
                if i < len(ranks):
                    rank_values[i].append(ranks[i])
                else:
                    # If ranks list is shorter than 32, append None or 0
                    rank_values[i].append(0)

        # Plot the rank values over total_steps
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))  # X-axis major ticks at every 1 step
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # Y-axis major ticks at every 1 unit in log scale
        plt.figure(figsize=(12, 8))

        plt.figure(figsize=(12, 8))
        for i in range(32):
            plt.plot(total_steps, np.log(rank_values[i]), label=f'Rank {i + 1}')

        plt.xlabel('Total Step')
        plt.ylabel('Log Singular Value')
        plt.title(f'Singular Values for {name}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
        plt.tight_layout()
        file_name = f'./images/{name}.png'
        plt.savefig(file_name, transparent=False)
        print(f"{name} save successfully")
        plt.close()


if __name__ == "__main__":
    plot_singular_values('records/qwen-0_5-exp-3')
