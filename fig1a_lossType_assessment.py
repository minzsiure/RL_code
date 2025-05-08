import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define experiment directories for different loss schemes
loss_dir = {
    'MSE (default)': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_181144',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd001_20250423_223555',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd002_20250423_233237'},
    'L1 loss': {
        1: 'exp/fql/Debug/l1/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_190920',
        2: 'exp/fql/Debug/l1/constant/antmaze-large-navigate-singletask-v0/sd001_20250424_003041',
        3: 'exp/fql/Debug/l1/constant/antmaze-large-navigate-singletask-v0/sd002_20250424_012922'},
    'Cosine similarity': {
        1: 'exp/fql/Debug/cosine/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_200755',
        2: 'exp/fql/Debug/cosine/constant/antmaze-large-navigate-singletask-v0/sd001_20250424_022809',
        3: 'exp/fql/Debug/cosine/constant/antmaze-large-navigate-singletask-v0/sd002_20250424_032642'}
}

# Directory to save figures
dir_path = 'figs/fig1a_lossType_assessment'
os.makedirs(dir_path, exist_ok=True)

def aggregate_trials(data_key, file_name):
    """Aggregate data from multiple trials and return mean and SEM over steps."""
    all_steps = []
    all_values = []

    for trial_path in file_name.values():
        file_path = os.path.join(trial_path, data_key)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_steps.append(df['step'].values)
            all_values.append(df.iloc[:, 1:].values)  # Skip step column
        else:
            print(f"Warning: {file_path} not found")

    # Ensure alignment on steps
    steps = all_steps[0]
    all_values = np.array(all_values)  # shape: (n_trials, n_steps, n_metrics)

    mean_vals = np.mean(all_values, axis=0)
    sem_vals = np.std(all_values, axis=0, ddof=1) / np.sqrt(all_values.shape[0])

    return steps, mean_vals, sem_vals

# ------------------------
# 1. Plot: Training Loss Panels (Distill and Q Loss for each loss scheme)
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

for label, trials in loss_dir.items():
    steps, mean_vals, sem_vals = aggregate_trials('train.csv', trials)
    distill_loss = mean_vals[:, 0]
    distill_sem = sem_vals[:, 0]
    q_loss = mean_vals[:, 1]
    q_sem = sem_vals[:, 1]

    axes[0].plot(steps, distill_loss, label=label)
    axes[0].fill_between(steps, distill_loss - distill_sem, distill_loss + distill_sem, alpha=0.3)

    axes[1].plot(steps, q_loss, label=label)
    axes[1].fill_between(steps, q_loss - q_sem, q_loss + q_sem, alpha=0.3)

# Distillation Loss panel
axes[0].set_title("Distillation Loss")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Loss Magnitude")
axes[0].legend()

# Q Loss panel
axes[1].set_title("Q Loss")
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Loss Magnitude")
axes[1].legend()

# Use only first and last x-tick
xticks = steps
if len(xticks) > 1:
    axes[0].set_xticks([xticks[0], xticks[-1]])
    axes[1].set_xticks([xticks[0], xticks[-1]])

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_distillation_Q_loss_train_sidebyside.png'), dpi=1000)

# ------------------------
# 2. Plot: Evaluation Success Comparison
# ------------------------
plt.figure(figsize=(6, 4))

for label, trials in loss_dir.items():
    all_steps = []
    all_success = []
    for trial_path in trials.values():
        eval_path = os.path.join(trial_path, 'eval.csv')
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            all_steps.append(df['step'].values)
            all_success.append(df['evaluation/success'].values)
        else:
            print(f"Warning: {eval_path} not found")

    if all_success:
        steps = all_steps[0]
        success_array = np.vstack(all_success)
        mean_success = np.mean(success_array, axis=0)
        sem_success = np.std(success_array, axis=0, ddof=1) / np.sqrt(success_array.shape[0])

        plt.plot(steps, mean_success, label=label)
        plt.fill_between(steps, mean_success - sem_success, mean_success + sem_success, alpha=0.3)

plt.xlabel("Training Step")
plt.ylabel("Eval Success Rate")
plt.title("Eval Success Across Distillation Loss Schemes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_eval_success_overlay.png'), dpi=1000)
