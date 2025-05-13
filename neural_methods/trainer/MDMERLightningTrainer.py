"""Trainer for BigSmall Multitask Models"""

# Training / Eval Imports 
import torch
import torch.optim as optim
import re
import torchmetrics

from neural_methods.model.BigSmall import BigSmall
from neural_methods.trainer.BaseTrainer import BaseTrainer

# from evaluation.bigsmall_multitask_metrics import (calculate_ppg_metrics, calculate_openface_au_metrics)

'''
from neural_methods import loss
'''

# Other Imports
from collections import OrderedDict
import numpy as np
import os
from tqdm import tqdm
from lightning import LightningModule
from torch import nn
import pandas as pd
from torch import tensor as torch_tensor
from torch import optim


class BigSmallLightningTrainer(LightningModule):
    def __init__(self, config, train_dataloader):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataloader'])
        self.config = config
        self.num_train_batches = len(train_dataloader)

        # ---------------------------
        # Task enable flags (ablation)
        # ---------------------------
        self.enable_au_eval = True
        self.enable_ppg_eval = True
        self.enable_emotion_eval = True
        self.set_label_weights = True

        # ---------------------------
        # Load label names and extract indices
        # ---------------------------
        label_list_path = os.path.dirname(config.TRAIN.DATA.DATA_PATH)
        label_names = self.load_label_names(label_list_path)

        # ---------------------------
        # Define column names
        # ---------------------------
        au_names = [name for name in label_names if "AU" in name and "int" not in name]
        discrete_emotion_colname = "emotion_lf"
        train_ppg_colname = "pos_ppg"
        test_ppg_colname = "ppg" # Option: ppg or filtered_ppg

        self.au_indices = [i for i, name in enumerate(label_names) if "AU" in name and "int" not in name]
        self.ppg_train_index = label_names.index(train_ppg_colname)
        self.ppg_test_index = label_names.index(test_ppg_colname)

        self.discrete_emotion_exist = discrete_emotion_colname in label_names   
        if self.discrete_emotion_exist:
            self.emotion_index = label_names.index(discrete_emotion_colname)
        else:
            raise NotImplementedError("Continuous emotion labels (arousal, valence, dominance) not implemented.")

        # ---------------------------
        # Compute class weights (if enabled)
        # ---------------------------
        if self.set_label_weights:
            au_weights, emotion_weights = self.compute_weights(au_names, discrete_emotion_colname)
        else:
            au_weights, emotion_weights = None, None

        # ---------------------------
        # Define metrics and losses
        # ---------------------------
        if self.enable_au_eval:
            self.criterionAU = nn.BCEWithLogitsLoss(pos_weight=au_weights)
        if self.enable_ppg_eval:
            self.criterionPPG = nn.MSELoss()
        if self.enable_emotion_eval:
            self.emotion_class = 8
            self.criterionEmotion = nn.CrossEntropyLoss(weight=emotion_weights)
            self.train_accuracy_emotion = torchmetrics.Accuracy(task="multiclass", 
                                                                num_classes=self.emotion_class)

        # ---------------------------
        # GPU + TSM setup
        # ---------------------------
        self.num_of_gpu = torch.cuda.device_count()
        self.using_TSM = self.num_of_gpu > 0
        if self.using_TSM:
            self.frame_depth = config.MODEL.BIGSMALL.FRAME_DEPTH
            self.base_len = self.num_of_gpu * self.frame_depth

        # ---------------------------
        # Initialize model
        # ---------------------------
        self.model = self.define_model(config)

        # ---------------------------
        # Initialize training loss
        # ---------------------------
        self.train_loss = []

    def configure_optimizers(self):
        LR = self.config.TRAIN.LR
        max_epochs = self.config.TRAIN.EPOCHS

        optimizer = optim.AdamW(self.model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=max_epochs,
                                                  steps_per_epoch=self.num_train_batches)
        return [optimizer], [scheduler]

    def define_model(self, config):
        if self.discrete_emotion_exist:
            return BigSmall(out_size_au=12 if self.enable_au_eval else 0,
                            out_size_ppg=1 if self.enable_ppg_eval else 0,
                            out_size_emotion=self.emotion_class if self.enable_emotion_eval else 0)

    @staticmethod
    def load_label_names(label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")
        with open(full_path, "r") as f:
            return [line.strip() for line in f]

    def get_label_paths(self):
        try:
            df = pd.read_csv(self.config.TRAIN.DATA.FILE_LIST_PATH, usecols=["label"])
        except ValueError as e:
            raise ValueError("CSV missing required column 'label'.") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV not found at {self.config.TRAIN.DATA.FILE_LIST_PATH}")

        if df.empty or df['label'].isnull().any():
            raise ValueError("CSV 'label' column is empty or contains NaNs.")

        return [os.path.join(self.config.TRAIN.DATA.CACHED_PATH, os.path.basename(p))
                for p in df["label"].values]

    def compute_weights(self, au_colname, emotion_colname, num_emotion_classes=8):
        """
        Computes AU and emotion class weights from training labels.
        Returns torch tensors or None if task is disabled.
        """
        
        label_paths = self.get_label_paths()

        au_pos = np.zeros(len(au_colname), dtype=np.float64)
        au_neg = np.zeros(len(au_colname), dtype=np.float64)
        emotion_counts = np.zeros(num_emotion_classes, dtype=np.float64)
        
        for i, path in enumerate(label_paths):
            labels = np.load(path)

            if self.enable_emotion_eval and self.discrete_emotion_exist:
                emotion = labels[emotion_colname].astype(np.int32)
                emotion_counts += np.bincount(emotion, minlength=num_emotion_classes)

            if self.enable_au_eval:
                # Extract AU fields into a regular float array
                au = np.stack([labels[field] for field in au_colname], axis=-1).astype(np.float32)
                au_pos += np.sum(au == 1, axis=0)
                au_neg += np.sum(au == 0, axis=0)
        
        au_weights = au_neg / (au_pos + 1e-5) if self.enable_au_eval else None

        emotion_weights = emotion_counts.sum() / (emotion_counts + 1e-5)
        emotion_weights /= emotion_weights.sum()

        return (torch.tensor(au_weights, dtype=torch.float32) if au_weights is not None else None,
                torch.tensor(emotion_weights, dtype=torch.float32))

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data, labels = batch

        # Original batch size before flattening
        B = labels.shape[0]  # (B, T, num_labels)

        data, labels = self.format_data_shape(data, labels)
        outputs = self(data)

        loss = self.compute_loss(outputs, labels, mode="Train")
        self.train_loss.append(loss)

        # Use original B when logging
        self.log("Training Loss: ", loss, prog_bar=True, logger=True,
                on_step=True, on_epoch=True, enable_graph=True, batch_size=B)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch

        B = labels.shape[0]  # (B, T, num_labels)

        data, labels = self.format_data_shape(data, labels)
        outputs = self(data)

        loss = self.compute_loss(outputs, labels, mode="Validation")
        self.val_loss.append(loss)

        self.log("Validation Loss: ", loss, prog_bar=True, logger=True,
                on_step=True, on_epoch=True, enable_graph=True, batch_size=B)
        
        return loss

    def on_train_epoch_end(self):
        mean_training_loss = np.mean(self.train_loss)
        self.log("Mean Training Loss: ", mean_training_loss)

        self.train_loss.clear()

    def compute_loss(self, outputs, labels, mode="Train"):
        au_output, ppg_output, emotion_output = outputs
        total_loss = 0.0
        au_loss = 0.0
        ppg_loss = 0.0
        emotion_loss = 0.0

        # Estimate B from reshaped labels (B*T, ...)
        batch_size = labels.shape[0] // self.frame_depth if self.using_TSM else labels.shape[0]

        if self.enable_au_eval:
            au_target = labels[:, self.au_indices]
            au_loss = self.criterionAU(au_output, au_target)
            total_loss += au_loss

            self.log(f"{mode} AU Loss", au_loss, batch_size=batch_size)

        if self.enable_ppg_eval:
            ppg_target = labels[:, self.ppg_train_index].unsqueeze(1)
            ppg_loss = self.criterionPPG(ppg_output, ppg_target)
            total_loss += ppg_loss

            self.log(f"{mode} PPG Loss", ppg_loss, batch_size=batch_size)

        if self.enable_emotion_eval:
            emotion_target = labels[:, self.emotion_index]
            self.train_accuracy_emotion.update(emotion_output, emotion_target)
            emotion_loss = self.criterionEmotion(emotion_output, emotion_target)
            total_loss += emotion_loss

            self.log(f"{mode} Accuracy Emotion", self.train_accuracy_emotion, batch_size=batch_size)
            self.log(f"{mode} Emotion Loss", emotion_loss, batch_size=batch_size)

        return total_loss, au_loss, ppg_loss, emotion_loss

    def format_data_shape(self, data, labels):
        """
        Efficiently reshapes input tensors for model training or inference.
        
        - Flattens temporal dimension (B, T, ...) → (B*T, ...)
        - Truncates to a multiple of `base_len` if `using_TSM` is enabled.
        """
        # Extract big and small data
        big, small = data

        # Flatten temporal dimension
        B, T, C, H, W = big.shape
        big = big.view(B * T, C, H, W)

        B, T, C, H, W = small.shape
        small = small.view(B * T, C, H, W)

        # Flatten labels
        labels = labels.view(B * T, -1)

        # TSM truncation
        if self.using_TSM:
            max_len = (B * T) // self.base_len * self.base_len
            big = big[:max_len]
            small = small[:max_len]
            labels = labels[:max_len]

        # Return the formatted data and labels
        data[0] = big
        data[1] = small

        return data, labels