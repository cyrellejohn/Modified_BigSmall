import os
import random
import numpy as np
import torch

def seed_everything_custom(seed=100, workers=True):
    """
    Custom function to set random seeds for reproducibility across:
    - Python's built-in RNG
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN
    - PyTorch DataLoader workers (if workers=True)

    Args:
        seed (int): The seed value to apply globally.
        workers (bool): Whether to create separate generators and worker_init_fn for DataLoader.

    Returns:
        dict or None: If workers=True, returns:
            {'train_generator': torch.Generator,
             'general_generator': torch.Generator,
             'seed_worker': callable}
            Otherwise, returns None.
    """

    # Set global seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally handle DataLoader worker and Generator seeding
    if workers:
        train_generator = torch.Generator().manual_seed(seed)
        general_generator = torch.Generator().manual_seed(seed)

        def seed_worker(worker_id):
            """
            Initialize each worker with a unique, deterministic seed.
            """
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return {'train_generator': train_generator,
                'general_generator': general_generator,
                'seed_worker': seed_worker}

    return None