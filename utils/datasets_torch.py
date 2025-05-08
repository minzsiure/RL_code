# datasets_torch.py

import numpy as np
import torch
import torch.nn.functional as F


def get_size(data):
    """Return the size of the dataset given a dict of numpy arrays."""
    sizes = [len(arr) for arr in data.values()]
    return max(sizes)


def random_crop(img, crop_from, padding):
    """
    Randomly crop an image using padding.
    
    Args:
        img (torch.Tensor): Image tensor of shape (H, W, C).
        crop_from (array-like or torch.Tensor): Coordinates to crop from; expected shape (3,). 
            The first two entries are the (y, x) starting indices.
        padding (int): Padding size.
        
    Returns:
        torch.Tensor: Cropped image of shape (H, W, C).
    """
    # Ensure img is a torch.Tensor.
    if not torch.is_tensor(img):
        img = torch.tensor(img)
    # PyTorch’s padding for images assumes channel-first format. Permute to (C, H, W).
    img_perm = img.permute(2, 0, 1)
    padded = F.pad(img_perm, pad=(padding, padding, padding, padding), mode='replicate')
    # Permute back to original shape (H, W, C)
    padded = padded.permute(1, 2, 0)
    # Get the cropping indices (assume crop_from is (y, x, 0) where the third value is a dummy).
    if torch.is_tensor(crop_from):
        crop_from = crop_from.cpu().numpy()
    y, x = int(crop_from[0]), int(crop_from[1])
    H, W, _ = img.shape
    cropped = padded[y:y+H, x:x+W, :]
    return cropped


def batched_random_crop(imgs, crop_froms, padding):
    """
    Batched version of random_crop.
    
    Args:
        imgs (torch.Tensor): Batch of images of shape (B, H, W, C).
        crop_froms (torch.Tensor): Tensor of shape (B, 3) with cropping coordinates.
        padding (int): Padding size.
        
    Returns:
        torch.Tensor: Batch of cropped images with shape (B, H, W, C).
    """
    cropped_list = []
    for i in range(imgs.shape[0]):
        cropped = random_crop(imgs[i], crop_froms[i], padding)
        cropped_list.append(cropped)
    return torch.stack(cropped_list, dim=0)


class Dataset:
    """
    Dataset class rewritten for PyTorch/NumPy.
    
    Attributes:
        _dict (dict): Underlying dictionary of numpy arrays.
        size (int): The dataset size (number of transitions).
        frame_stack (int or None): Number of frames to stack.
        p_aug (float or None): Image augmentation probability.
        return_next_actions (bool): Whether to also return next actions.
        terminal_locs (np.ndarray): Indices of terminals.
        initial_locs (np.ndarray): Indices of episode starts.
    """
    def __init__(self, data, freeze=True):
        """
        Args:
            data (dict): Dictionary of numpy arrays.
            freeze (bool): If True, set the arrays read-only.
        """
        self._dict = data
        if freeze:
            for arr in self._dict.values():
                arr.setflags(write=False)
        self.size = get_size(self._dict)
        self.frame_stack = None  # To be set externally if needed.
        self.p_aug = None        # Augmentation probability (set externally).
        self.return_next_actions = False
        self.terminal_locs = np.nonzero(self._dict['terminals'] > 0)[0]
        self.initial_locs = np.concatenate(([0], self.terminal_locs[:-1] + 1))
        
    def __getitem__(self, key):
        return self._dict[key]
    
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)
    
    def items(self):
        return self._dict.items()
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()

    @classmethod
    def create(cls, freeze=True, **fields):
        """
        Create a dataset from fields.

        Args:
            freeze (bool): Whether to freeze the arrays.
            **fields: Must include at least 'observations'.
        """
        data = fields
        assert 'observations' in data, "Dataset must have key 'observations'."
        return cls(data, freeze=freeze)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(0, self.size, size=num_idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset for the given indices."""
        result = {k: v[idxs] for k, v in self._dict.items()}
        if self.return_next_actions:
            # WARNING: May be incorrect near episode boundaries.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply random-crop image augmentation to the specified keys in the batch."""
        padding = 3
        batch_size = len(batch[keys[0]])
        # Generate random (y, x) starting positions. Append a dummy 0 to match expected shape.
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        zeros = np.zeros((batch_size, 1), dtype=np.int64)
        crop_froms = np.concatenate([crop_froms, zeros], axis=1)  # Shape: (B, 3)
        crop_froms = torch.from_numpy(crop_froms)
        for key in keys:
            imgs = batch[key]
            # Convert to torch tensor if necessary.
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs)
            # Assume imgs is of shape (B, H, W, C) and type float.
            batch[key] = batched_random_crop(imgs, crop_froms, padding).numpy()

    def sample(self, batch_size: int, idxs=None):
        """
        Sample a batch of transitions.

        Args:
            batch_size (int): Number of samples.
            idxs (np.ndarray or None): Optional indices; if None they are randomly selected.
            
        Returns:
            dict: Batch of transitions with keys such as 'observations' and 'next_observations'.
                  Converted to torch tensors.
        """
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames for observations and next_observations.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs_list = []      # To store stacked observation frames.
            next_obs_list = [] # To store stacked next observation frames.
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs_list.append(self._dict['observations'][cur_idxs])
                if i != self.frame_stack - 1:
                    next_obs_list.append(self._dict['observations'][cur_idxs])
            next_obs_list.append(self._dict['next_observations'][idxs])
            batch['observations'] = np.concatenate(obs_list, axis=-1)
            batch['next_observations'] = np.concatenate(next_obs_list, axis=-1)
        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        # Convert each value in the batch to a torch.Tensor.
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


class ReplayBuffer(Dataset):
    """
    ReplayBuffer extends Dataset and supports adding transitions.
    """
    @classmethod
    def create(cls, transition, size):
        """
        Create a replay buffer from an example transition.
        
        Args:
            transition (dict): Example transition.
            size (int): Replay buffer size.
        """
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        buffer_dict = {k: create_buffer(v) for k, v in transition.items()}
        return cls(buffer_dict, freeze=False)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """
        Create a replay buffer pre-filled with an initial dataset.
        
        Args:
            init_dataset (dict): Initial dataset.
            size (int): Buffer size.
        """
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[:len(init_buffer)] = init_buffer
            return buffer
        buffer_dict = {k: create_buffer(v) for k, v in init_dataset.items()}
        dataset = cls(buffer_dict, freeze=False)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, data, freeze=False):
        super().__init__(data, freeze=freeze)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """
        Add a transition to the replay buffer.

        Args:
            transition (dict): Transition to add.
        """
        for k in self._dict.keys():
            self._dict[k][self.pointer] = transition[k]
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = 0
        self.pointer = 0
