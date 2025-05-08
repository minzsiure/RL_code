import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define experiment directories for different loss schemes
loss_dir = {
    r'Constant $\alpha$ (default)': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_181144',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd001_20250423_223555',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd002_20250423_233237'},
    'Distillation loss off halfway': {
        1: 'exp/fql/Debug/mse/off_after_half/antmaze-large-navigate-singletask-v0/sd000_20250420_210720',
        2: 'exp/fql/Debug/mse/off_after_half/antmaze-large-navigate-singletask-v0/sd001_20250423_223716',
        3: 'exp/fql/Debug/mse/off_after_half/antmaze-large-navigate-singletask-v0/sd002_20250423_233848'},
    r'Linear reduction (10 $\rightarrow$ 0)': {
        1: 'exp/fql/Debug/mse/linear_decay/antmaze-large-navigate-singletask-v0/sd000_20250420_221109',
        2: 'exp/fql/Debug/mse/linear_decay/antmaze-large-navigate-singletask-v0/sd001_20250424_004049',
        3: 'exp/fql/Debug/mse/linear_decay/antmaze-large-navigate-singletask-v0/sd002_20250424_013910'},
    r'Linear increase (0 $\rightarrow$ 10)': {
        1: 'exp/fql/Debug/mse/linear_increase/antmaze-large-navigate-singletask-v0/sd000_20250420_231027',
        2: 'exp/fql/Debug/mse/linear_increase/antmaze-large-navigate-singletask-v0/sd001_20250424_023840',
        3: 'exp/fql/Debug/mse/linear_increase/antmaze-large-navigate-singletask-v0/sd002_20250424_090826'}
}

# Directory to save figures
dir_path = 'figs/fig1b_alpha_scheme'
os.makedirs(dir_path, exist_ok=True)

def aggregate_trials(file_name, trial_paths, columns):
    all_steps = []
    all_values = []

    for path in trial_paths.values():
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            all_steps.append(df['step'].values)
            all_values.append(df[columns].values)
        else:
            print(f"Warning: {full_path} not found")

    if not all_values:
        return None, None, None

    steps = all_steps[0]
    values = np.array(all_values)  # shape: (n_trials, n_steps, n_metrics)
    mean_vals = np.mean(values, axis=0)
    sem_vals = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    return steps, mean_vals, sem_vals

# ------------------------
# 1. Plot: Training Loss Panels (Distill and Q Loss for each loss scheme)
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

for label, trials in loss_dir.items():
    steps, mean_vals, sem_vals = aggregate_trials('train.csv', trials,
                                                  ['training/actor/distill_loss', 'training/actor/q_loss'])
    if steps is None:
        continue

    distill_mean, q_mean = mean_vals[:, 0], mean_vals[:, 1]
    distill_sem, q_sem = sem_vals[:, 0], sem_vals[:, 1]

    axes[0].plot(steps, distill_mean, label=label)
    axes[0].fill_between(steps, distill_mean - distill_sem, distill_mean + distill_sem, alpha=0.3)

    axes[1].plot(steps, q_mean, label=label)
    axes[1].fill_between(steps, q_mean - q_sem, q_mean + q_sem, alpha=0.3)

axes[0].set_title("Distillation Loss")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Loss Magnitude")
axes[0].set_yscale("log")
axes[0].legend(fontsize=10)

axes[1].set_title("Q Loss")
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Loss Magnitude")
axes[1].legend(fontsize=10)

# Use only first and last x-tick
if steps is not None and len(steps) > 1:
    axes[0].set_xticks([steps[0], steps[-1]])
    axes[1].set_xticks([steps[0], steps[-1]])

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_distillation_Q_loss_train_sidebyside.png'), dpi=1000)

# ------------------------
# 2. Plot: Evaluation Success Comparison
# ------------------------
plt.figure(figsize=(6, 4))

for label, trials in loss_dir.items():
    all_steps = []
    all_success = []

    for path in trials.values():
        eval_path = os.path.join(path, 'eval.csv')
        if os.path.exists(eval_path):
            df = pd.read_csv(eval_path)
            all_steps.append(df['step'].values)
            all_success.append(df['evaluation/success'].values)
        else:
            print(f"Warning: {eval_path} not found")

    if all_success:
        steps = all_steps[0]
        success_arr = np.vstack(all_success)
        success_mean = np.mean(success_arr, axis=0)
        success_sem = np.std(success_arr, axis=0, ddof=1) / np.sqrt(success_arr.shape[0])

        plt.plot(steps, success_mean, label=label)
        plt.fill_between(steps, success_mean - success_sem, success_mean + success_sem, alpha=0.3)

plt.xlabel("Training Step")
plt.ylabel("Eval Success Rate")
plt.title(r"Eval Success Across $\alpha$ Schedules")
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_eval_success_overlay.png'), dpi=1000)
