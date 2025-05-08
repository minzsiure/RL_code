import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Your provided directories
actor_dir = {
    '2-layer actor': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_012613',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_020943',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_025256'},
    '4-layer actor (default)': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_181144',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd001_20250423_223555',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd002_20250423_233237'},
    '6-layer actor':{
        1:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_033621',
        2:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_044935',
        3:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_060307',
    },
    '8-layer actor':{
        1:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_071627',
        2:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_084637',
        3:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_101539'
    }
}

value_dir = {
    '2-layer critic': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_114529',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_123357',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_132214'},
    '4-layer critic (default)': {
        1: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd000_20250420_181144',
        2: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd001_20250423_223555',
        3: 'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-v0/sd002_20250423_233237'},
    '6-layer critic':{
        1:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_141108',
        2:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_151952',
        3:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_162837',
    },
    '8-layer critic':{
        1:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd000_20250425_173753',
        2:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd001_20250425_185755',
        3:'exp/fql/Debug/mse/constant/antmaze-large-navigate-singletask-task1-v0/sd002_20250425_201821'
    }
}

# Save directory
dir_path = 'figs/fig3_actor_value_depth_comparison'
os.makedirs(dir_path, exist_ok=True)

# ------------------------
# Create figure with two panels
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Function to plot
def plot_eval_success(data_dict, ax, title):
    for label, trials in data_dict.items():
        eval_steps_list = []
        success_list = []

        for trial_path in trials.values():
            eval_path = os.path.join(trial_path, 'eval.csv')
            if os.path.exists(eval_path):
                df_eval = pd.read_csv(eval_path)
                eval_steps_list.append(df_eval['step'].values)
                success_list.append(df_eval['evaluation/success'].values)
            else:
                print(f"Warning: {eval_path} not found.")

        if success_list:
            eval_steps = eval_steps_list[0]
            success_array = np.vstack(success_list)
            mean_success = np.mean(success_array, axis=0)
            sem_success = np.std(success_array, axis=0, ddof=1) / np.sqrt(success_array.shape[0])

            ax.plot(eval_steps, mean_success, label=label)
            ax.fill_between(eval_steps, mean_success - sem_success, mean_success + sem_success, alpha=0.3)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Success Rate")
    ax.set_title(title)
    ax.legend()

# Left panel: actor depth comparison
plot_eval_success(actor_dir, axes[0], "Eval Success wrt Actor Network Depth (critic depth = 4)")

# Right panel: critic depth comparison
plot_eval_success(value_dir, axes[1], "Eval Success wrt Critic Network Depth (actor depth = 4)")

plt.tight_layout()
plt.savefig(os.path.join(dir_path, 'fig_actor_critic_depth_success_sidebyside.png'), dpi=1000)
