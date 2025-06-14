""" Main script for training and testing the Modified BigSmall model pipeline. """

import argparse
from torch.utils.data import DataLoader
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from utils.seed_utils import seed_everything_custom
from utils.label_helper import LabelHelper

def add_args(parser):
    """Adds CLI arguments."""
    parser.add_argument('--config_file', default="configs/MDMER_BigSmall.yaml", type=str)
    return parser


def get_data_loader_class(dataset_name):
    """Maps dataset name to corresponding loader class."""
    loader_map = {
        "UBFC-rPPG": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
        "UBFCrPPG_BigSmall": data_loader.BigSmallLoader.UBFCrPPGLoader,
        "MDMER": data_loader.BigSmallLoader.MDMERLoader
    }
    if dataset_name not in loader_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loader_map[dataset_name]


def build_dataloader(mode, data_config, batch_size, num_workers, shuffle, seed_worker, generator):
    """Initializes and returns a PyTorch DataLoader."""
    if not data_config.DATASET or not data_config.DATA_PATH:
        return None
    loader_class = get_data_loader_class(data_config.DATASET)
    dataset = loader_class(mode, data_config.DATA_PATH, data_config)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=seed_worker,
                      generator=generator,
                      persistent_workers=True,
                      pin_memory=True)

def run_model(config, dataloaders, train=True, test=True):
    """Instantiates and runs training and/or testing."""

    if not train and not test:
        print("[Warning] Neither training nor testing is enabled. Exiting run_model.")
        return

    use_amp = False
    calculate_weights = False

    trainer_map = {
        "BigSmall": trainer.BigSmallTrainer.BigSmallTrainer,
        "MDMER_BigSmall": trainer.MDMERTrainer.BigSmallTrainer,
        "DeepPhys": trainer.DeepPhysTrainer.DeepPhysTrainer
    }

    if config.MODEL.NAME not in trainer_map:
        raise ValueError(f"Unsupported model: {config.MODEL.NAME}")

    trainer_class = trainer_map[config.MODEL.NAME]

    # Prepare weights only once if needed
    if train and calculate_weights:
        label_helper = LabelHelper(config, num_emotion_classes=6)
        label_helper.compute_weights_and_save()

    # Instantiate trainer only once
    model_trainer = trainer_class(config, dataloaders, use_amp=use_amp) if train else trainer_class(config, dataloaders)

    # Execute train/test
    if train:
        model_trainer.train(dataloaders)
    if test:
        model_trainer.test(dataloaders)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Load configuration and seed
    config = get_config(args)
    print('Configuration:', config, end='\n\n')

    seed = seed_everything_custom(seed=17, workers=True)
    train_gen, general_gen, seed_worker = seed["train_generator"], seed["general_generator"], seed["seed_worker"]

    dataloaders = {}

    if config.TOOLBOX_MODE in ["train_and_test", "only_test"]:
        if config.TOOLBOX_MODE == "train_and_test":
            dataloaders["train"] = build_dataloader(mode="train", 
                                                    data_config=config.TRAIN.DATA,
                                                    batch_size=config.TRAIN.BATCH_SIZE,
                                                    num_workers=28,
                                                    shuffle=True,
                                                    generator=train_gen,
                                                    seed_worker=seed_worker)

            if config.VALID.RUN_VALIDATION:
                dataloaders["valid"] = build_dataloader(mode="valid", 
                                                        data_config=config.VALID.DATA,
                                                        batch_size=config.TRAIN.BATCH_SIZE,
                                                        num_workers=28,
                                                        shuffle=False,
                                                        generator=general_gen,
                                                        seed_worker=seed_worker)

        dataloaders["test"] = build_dataloader(mode="test", 
                                               data_config=config.TEST.DATA,
                                               batch_size=config.INFERENCE.BATCH_SIZE,
                                               num_workers=28,
                                               shuffle=False,
                                               generator=general_gen,
                                               seed_worker=seed_worker)

     # Execute training or testing
    if config.TOOLBOX_MODE == "train_and_test":
        run_model(config, dataloaders, train=True, test=False)
    elif config.TOOLBOX_MODE == "only_test":
        run_model(config, dataloaders, train=False, test=True)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        print("Unsupervised method is not implemented yet.")
    else:
        raise ValueError(f"Unsupported toolbox mode: {config.TOOLBOX_MODE}")

if __name__ == "__main__":
    main()