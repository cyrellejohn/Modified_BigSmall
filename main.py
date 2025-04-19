""" The main function of Modified BigSmall Model pipeline. """

import argparse
import random
import time
import multiprocessing

import numpy as np
import torch
from neural_methods import trainer
from dataset import data_loader
from config import get_config
'''
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
'''
from torch.utils.data import DataLoader

# Set a fixed random seed for reproducibility
RANDOM_SEED = 100

# Set the random seed for PyTorch on CPU
torch.manual_seed(RANDOM_SEED)

# Set the random seed for PyTorch on GPU
torch.cuda.manual_seed(RANDOM_SEED)

# Set the random seed for NumPy
np.random.seed(RANDOM_SEED)

# Set the random seed for Python's built-in random module
random.seed(RANDOM_SEED)

# Configure PyTorch to use deterministic algorithms for reproducibility
torch.backends.cudnn.deterministic = True

# Disable cuDNN benchmarking to ensure deterministic behavior
torch.backends.cudnn.benchmark = False

# Create a general random number generator for validation, test, and unsupervised data loaders
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)

# Create a separate random number generator for the training data loader
# This helps in isolating the training data loader from others and controlling non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/MDMER_BigSmall.yaml", 
                        type=str, help="The name of the model.")

    return parser

def seed_worker(worker_id):
    """
    Initializes the random seed for a data loader worker to ensure reproducibility.

    This function sets the seed for the random number generators used by PyTorch, NumPy,
    and Python's built-in random module. It ensures that each worker in a data loading
    process has a unique and deterministic seed, which is crucial for reproducibility
    when using multiple workers in PyTorch's DataLoader.

    Parameters:
    worker_id (int): The ID of the worker. This is automatically provided by PyTorch's DataLoader.
    """
    # Generate a seed for the worker by taking the initial seed of PyTorch's RNG and applying a modulo operation to fit within a 32-bit integer range.
    worker_seed = torch.initial_seed() % 2 ** 32

    # Set the seed for NumPy's random number generator to ensure deterministic behavior.
    np.random.seed(worker_seed)

    # Set the seed for Python's built-in random module to ensure deterministic behavior.
    random.seed(worker_seed)

def run_model(config, data_loader_dict, train=True, test=True):
    """
    Runs training, testing, or both based on the specified boolean flags.

    Parameters:
    - config: Configuration object containing model settings and parameters.
    - data_loader_dict: Dictionary containing data loaders for different datasets.
    - train: Boolean indicating whether to run training.
    - test: Boolean indicating whether to run testing.

    Raises:
    - ValueError: If the model name specified in the configuration is not supported.
    """
    if config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported Yet!')

    if train:
        model_trainer.train(data_loader_dict)
    if test:
        model_trainer.test(data_loader_dict)

def unsupervised_method_inference(config, data_loader):
    # TODO: Implement this function
    print("NOT IMPLEMENTED YET")

# MAIN EXECUTION SCRIPT START
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Configurations
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict() # Dictionary of data loaders 
    if config.TOOLBOX_MODE == "train_and_test":
        # -----------------------------------------------------------------------------
        # Train Loader
        # -----------------------------------------------------------------------------
        if config.TRAIN.DATA.DATASET == "UBFC-rPPG":
            train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "UBFCrPPG_BigSmall":
            train_loader = data_loader.BigSmallLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "MDMER":
            train_loader = data_loader.BigSmallLoader.MDMERLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG dataset.")

        # Check if both the dataset name and path are specified in the configuration.
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            # Initialize the training data loader with the correct toolbox mode, specified dataset name, path, and configuration.
            train_data_loader = train_loader(name="train", # Label indicating this is the training data.
                                             data_path=config.TRAIN.DATA.DATA_PATH, # Path to the training data.
                                             config_data=config.TRAIN.DATA) # Additional configuration for loading the dataset.
            
            # Create a PyTorch DataLoader for the training data.
            data_loader_dict['train'] = DataLoader(dataset=train_data_loader, # The dataset to load.
                                                   num_workers=16, # Default 16, 4 for now | Number of subprocesses to use for data loading.
                                                   batch_size=config.TRAIN.BATCH_SIZE, # Number of samples per batch to load.
                                                   shuffle=True, # Shuffle the data at every epoch.
                                                   worker_init_fn=seed_worker, # Function to initialize the random seed for each worker.
                                                   generator=train_generator) # Random number generator for controlling randomness.
        else:
            # If dataset name or path is not specified, set the training data loader to None.
            data_loader_dict['train'] = None

        # -----------------------------------------------------------------------------
        # Valid Loader
        # -----------------------------------------------------------------------------
        if config.VALID.DATA.DATASET == "UBFC-rPPG":
            valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "UBFCrPPG_BigSmall":
            valid_loader = data_loader.BigSmallLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET is None and config.VALID.RUN_VALIDATION:
            raise ValueError("Validation dataset not specified despite RUN_VALIDATION is set to True!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG dataset")

        # Check if a validation dataset and path are specified and if running validation is True
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and config.VALID.RUN_VALIDATION):
            # Initialize the validation data loader with the specified dataset and configuration
            valid_data_loader = valid_loader(name="valid", # Name the data loader as 'valid'
                                             data_path=config.VALID.DATA.DATA_PATH, # Path to the validation data
                                             config_data=config.VALID.DATA) # Additional configuration for validation
            
            # Create a DataLoader for the validation dataset
            data_loader_dict["valid"] = DataLoader(dataset=valid_data_loader, # The dataset to load
                                                   num_workers=8, # Number of subprocesses for data loading
                                                   batch_size=config.TRAIN.BATCH_SIZE, # Batch size, same as training
                                                   shuffle=False, # Do not shuffle validation data
                                                   worker_init_fn=seed_worker, # Initialize each worker with a seed
                                                   generator=general_generator) # Use a specific random generator for consistency
        else:
            # If conditions are not met, set the validation data loader to None
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # -----------------------------------------------------------------------------
        # Test Loader
        # -----------------------------------------------------------------------------
        if config.TEST.DATA.DATASET == "UBFC-rPPG":
            test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "UBFCrPPG_BigSmall":
            test_loader = data_loader.BigSmallLoader.UBFCrPPGLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG dataset.")

        if config.TOOLBOX_MODE == "train_and_test" and not config.VALID.RUN_VALIDATION:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')

        # Check if both the test dataset name and path are provided in the configuration.
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            # Initialize the test dataset using the specified loader, dataset name, and path.
            test_data_loader = test_loader(name="test",
                                           data_path=config.TEST.DATA.DATA_PATH,
                                           config_data=config.TEST.DATA)

            # Create a DataLoader for the test dataset. This DataLoader will handle batching, shuffling, and parallel data loading.
            data_loader_dict["test"] = DataLoader(dataset=test_data_loader, # The dataset to load
                                                  num_workers=8, # Number of subprocesses for data loading.
                                                  batch_size=config.INFERENCE.BATCH_SIZE, # Number of samples per batch.
                                                  shuffle=False, # Do not shuffle data (order matters for testing).
                                                  worker_init_fn=seed_worker, # Function to initialize each worker process.
                                                  generator=general_generator) # Random number generator for deterministic behavior.
        else:
            # If dataset name or path is missing, set the test data loader to None.
            data_loader_dict['test'] = None

    # elif config.TOOLBOX_MODE == "unsupervised_method": # TODO: NOT IMPLEMENTED AND NOT FINAL

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test, only_test or unsupervised_method")
    
    if config.TOOLBOX_MODE == "train_and_test":
        pass # run_model(config, data_loader_dict, train=True, test=True)

    elif config.TOOLBOX_MODE == "only_test":
        run_model(config, data_loader_dict, train=False, test=True)

    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)

    else:
        print("Unsupported toolbox_mode! Currently support train_and_test, only_test or unsupervised_method", end='\n\n')