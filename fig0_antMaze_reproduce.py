import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define directories
dirs = {
    1: 'exp/fql/Debug/sd000_20250416_141502',
    2: 'exp/fql/Debug/sd000_20250416_160456',
    3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task3-v0/sd000_20250423_162300',
    4: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task4-v0/sd000_20250424_091357',
    5: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task5-v0/sd000_20250424_103236'
}

# Initialize storage
all_steps = []
all_accs = []

# Load data from each run
for i, path in dirs.items():
    eval_path = os.path.join(path, 'eval.csv')
    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        all_steps.append(df['step'].values)
        all_accs.append(df['evaluation/success'].values)
    else:
        print(f"Warning: {eval_path} not found")

# Convert to arrays and plot
if all_accs:
    steps = all_steps[0]
    acc_array = np.vstack(all_accs)
    mean_acc = np.mean(acc_array, axis=0)
    sem_acc = np.std(acc_array, axis=0, ddof=1) / np.sqrt(acc_array.shape[0])

    # Plot
    plt.figure(figsize=(6, 4))

    # Plot individual task curves in light gray
    for acc in acc_array:
        plt.plot(steps, acc, color='gray', alpha=0.5, linewidth=0.5)

    # Plot mean with SEM
    plt.plot(steps, mean_acc, label='Average Accuracy', color='C0')
    print(f"Final mean accuracy: {mean_acc[-1]}")
    plt.fill_between(steps, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.25, color='C0', edgecolor='none')

    plt.xlabel("Training Step")
    plt.ylabel("Eval Success Rate")
    plt.title("Eval Accuracy Across AntMaze Tasks")
    plt.legend()
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/fig_accuracy_antMaze_allTasks.pdf", dpi=1000)
else:
    print("No valid eval.csv files found.")
