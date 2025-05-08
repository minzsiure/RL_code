import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define experiment directories for different flow steps
loss_dir = {
    'flow step = 1': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250424_170337',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250424_174630',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250424_182929'},
    'flow step = 5': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250424_191257',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250424_200225',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250424_205217'},
    'flow step = 10 (default)': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_181144',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd001_20250423_223555',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd002_20250423_233237'},
    'flow step = 20':{
        1:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250424_214250',
        2:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250424_225714',
        3:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_001136',
    }
}

# Directory to save figures
dir_path = 'figs/fig2_flowStep'
os.makedirs(dir_path, exist_ok=True)

# ------------------------
# 1. Create figure with two panels
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, trials in loss_dir.items():
    train_steps_list = []
    total_time_list = []
    eval_steps_list = []
    success_list = []

    for trial_path in trials.values():
        train_path = os.path.join(trial_path, 'train.csv')
        eval_path = os.path.join(trial_path, 'eval.csv')

        if os.path.exists(train_path) and os.path.exists(eval_path):
            df_train = pd.read_csv(train_path)
            df_eval = pd.read_csv(eval_path)

            train_steps_list.append(df_train['step'].values)
            total_time_list.append(df_train['time/total_time'].values)

            eval_steps_list.append(df_eval['step'].values)
            success_list.append(df_eval['evaluation/success'].values)
        else:
            print(f"Warning: Missing file(s) in {trial_path}")

    if total_time_list and success_list:
        # Panel A: physical total time (use train.csv)
        train_steps = train_steps_list[0]
        total_time_array = np.vstack(total_time_list)
        mean_total_time = np.mean(total_time_array, axis=0)
        sem_total_time = np.std(total_time_array, axis=0, ddof=1) / np.sqrt(total_time_array.shape[0])

        axes[0].plot(train_steps, mean_total_time, label=label)
        axes[0].fill_between(train_steps, mean_total_time - sem_total_time, mean_total_time + sem_total_time, alpha=0.25, linewidth=0)

        # Panel B: eval success (use eval.csv)
        eval_steps = eval_steps_list[0]
        success_array = np.vstack(success_list)
        mean_success = np.mean(success_array, axis=0)
        sem_success = np.std(success_array, axis=0, ddof=1) / np.sqrt(success_array.shape[0])

        axes[1].plot(eval_steps, mean_success, label=label)
        axes[1].fill_between(eval_steps, mean_success - sem_success, mean_success + sem_success, alpha=0.25, linewidth=0)

# ------------------------
# Final touches
# ------------------------
# Panel A settings
axes[0].set_title("Physical Time Taken")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Total Time (seconds)")
axes[0].legend()

# Panel B settings
axes[1].set_title("Eval Success Rate")
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Eval Success Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_time_success_sidebyside.png'), dpi=1000)
