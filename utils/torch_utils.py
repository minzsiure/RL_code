import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.optim as optim


class ModuleDict(nn.Module):
    """
    A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.
    """
    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__()
        # Use PyTorch's native ModuleDict
        self.modules_dict = nn.ModuleDict(modules)

    def forward(self, *args, name=None, **kwargs):
        """
        Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if set(kwargs.keys()) != set(self.modules_dict.keys()):
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {list(kwargs.keys())} but module keys {list(self.modules_dict.keys())}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules_dict[key](**value)
                elif isinstance(value, Sequence) and not isinstance(value, str):
                    out[key] = self.modules_dict[key](*value)
                else:
                    out[key] = self.modules_dict[key](value)
            return out
        else:
            return self.modules_dict[name](*args, **kwargs)


class TrainState:
    """
    Custom train state for models.

    Attributes:
        step: Counter to keep track of training steps.
        model: The PyTorch model (nn.Module).
        optimizer: Optimizer for the model.
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, step: int = 1):
        self.step = step
        self.model = model
        self.optimizer = optimizer

    @classmethod
    def create(cls, model: nn.Module, optimizer: optim.Optimizer, **kwargs):
        """Create a new train state."""
        return cls(model=model, optimizer=optimizer, step=1)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def apply_gradients(self, loss):
        """
        Compute gradients from the loss, update the model parameters, and return gradient statistics.

        Args:
            loss: A scalar loss tensor computed from the model's output.

        Returns:
            A dictionary containing gradient statistics.
        """
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_max = -float('inf')
        grad_min = float('inf')
        grad_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                grad_max = max(grad_max, param.grad.max().item())
                grad_min = min(grad_min, param.grad.min().item())
                grad_norm += param.grad.data.norm(1).item()

        self.optimizer.step()
        self.step += 1
        info = {
            'grad/max': grad_max,
            'grad/min': grad_min,
            'grad/norm': grad_norm,
        }
        return info

    def apply_loss_fn(self, loss_fn, *args, **kwargs):
        """
        Compute the loss using a provided loss function, backpropagate gradients, update the state, and return
        the loss value along with gradient statistics.

        The loss_fn should take the model as its first argument and return a scalar loss tensor.

        Returns:
            A tuple (loss_value, info) where loss_value is a float and info is a dict of gradient statistics.
        """
        loss = loss_fn(self.model, *args, **kwargs)
        info = self.apply_gradients(loss)
        return loss.item(), info


def save_agent(agent: TrainState, save_dir: str, epoch: int):
    """
    Save the agent's state to a file.

    Args:
        agent: The TrainState to save.
        save_dir: Directory to save the agent.
        epoch: Epoch number to include in the filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'params_{epoch}.pt')
    torch.save({
        'step': agent.step,
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, save_path)
    print(f'Saved to {save_path}')


def restore_agent(agent: TrainState, restore_path: str, restore_epoch: int) -> TrainState:
    """
    Restore the agent from a file.

    Args:
        agent: The TrainState to restore.
        restore_path: Path pattern to the directory containing the saved agent.
        restore_epoch: Epoch number corresponding to the saved checkpoint.

    Returns:
        The restored TrainState.
    """
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    restore_dir = candidates[0]
    file_path = os.path.join(restore_dir, f'params_{restore_epoch}.pt')
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.step = checkpoint['step']
    print(f'Restored from {file_path}')
    return agent
