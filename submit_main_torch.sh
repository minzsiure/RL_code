#!/bin/bash
#SBATCH --job-name=fql_antmaze
#SBATCH --output=logs/torch_%x.out
#SBATCH --error=logs/torch_%x.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate fql

python main_torch.py \
  --env_name=antmaze-large-navigate-singletask-task1-v0 \
  --agent.q_agg=min \
  --agent.alpha=10
