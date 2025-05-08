Following **OGBench** code for training & evaluating agents.
To start, do
```
pip install -r requirements.txt
```

## Fig 1: examinate the role of different loss (Q loss vs distillation loss)
### (a) Experiment on different distillation loss type
The FQL paper uses l2 distillation loss, here we can also try `l1` or `cosine` similarity by passing in:
```
python main_jax.py --agent.distill_loss_type=mse     # default
python main_jax.py --agent.distill_loss_type=l1
python main_jax.py --agent.distill_loss_type=cosine
```

### (b) Experiment on weighting (alpha) of distillation loss
The FQL paper employs a constant `alpha` to modulate distillation loss.
We ask, what the performance and stability look if we 
1. turn off the distillation loss halfway (`off_after_half`) or, 
2. reduce it linearly (`linear_decay`) or,
3. increase it lienarly (`linear_increase`) or,
4. stay constant (`constant`), which is the default matching the paper.

```
# Linearly decay from alpha=10 to alpha=0
python main_jax.py --agent.alpha=10 --agent.alpha_final=0 --agent.distill_loss_schedule=linear_decay

# Turn off distillation halfway through
python main_jax.py --agent.distill_loss_schedule=off_after_half

# Gradually increase distillation
python main_jax.py --agent.alpha=0 --agent.alpha_final=10 --agent.distill_loss_schedule=linear_increase
```