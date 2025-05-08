# evaluation_torch.py

from collections import defaultdict
import torch
import numpy as np
from tqdm import trange
import random

def supply_rng(f, seed=None):
    """
    Helper function to create a torch.Generator seeded by an integer and supply it 
    as the 'seed' keyword argument to function f.
    
    Args:
        f: A function that accepts a keyword argument 'seed' (of type torch.Generator).
        seed: An optional integer seed. If not provided, a random seed is generated.
    
    Returns:
        A wrapped version of f that always passes the same torch.Generator.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    def wrapped(*args, **kwargs):
        return f(*args, seed=gen, **kwargs)
    return wrapped

def flatten(d, parent_key='', sep='.'):
    """Flatten a nested dictionary.
    
    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Prefix for keys.
        sep (str): Separator between keys.
    
    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    """
    Append each value in single_dict to the corresponding list in dict_of_lists.
    
    Args:
        dict_of_lists (dict): A dictionary whose values are lists.
        single_dict (dict): A dictionary with scalar values.
    """
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """
    Evaluate the agent in the environment using PyTorch RNG.
    
    Args:
        agent: The agent, which should have a sample_actions method.
        env: The environment.
        config: Configuration dictionary (not used directly here).
        num_eval_episodes: Number of evaluation episodes.
        num_video_episodes: Number of episodes to render (these are not included in statistics).
        video_frame_skip: How many frames to skip between video renders.
        eval_temperature: Sampling temperature for actions.
    
    Returns:
        A tuple of (stats, trajectories, renders):
          - stats: A dictionary of averaged evaluation metrics.
          - trajectories: A list of trajectories (each a dictionary of lists).
          - renders: A list of numpy arrays representing rendered video frames.
    """
    # Wrap the agent's sample_actions function with a torch RNG generator
    actor_fn = supply_rng(agent.sample_actions)
    
    trajs = []
    stats = defaultdict(list)
    renders = []
    
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        
        observation, info = env.reset()
        done = False
        step = 0
        render = []
        
        while not done:
            # Call the actor_fn with observations and temperature;
            # note that actor_fn will automatically supply a torch.Generator as seed.
            action = actor_fn(observations=observation, temperature=eval_temperature)
            # Convert the action from a tensor to a numpy array.
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1, 1)
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)
            
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
    
    # Average evaluation statistics.
    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    return stats, trajs, renders
