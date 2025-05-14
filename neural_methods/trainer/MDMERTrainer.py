"""Trainer for BigSmall Multitask Models"""

import numpy as np
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from neural_methods.model.BigSmall import BigSmall
from neural_methods.trainer.BaseTrainer import BaseTrainer
from evaluation.bigsmall_metrics import (calculate_ppg_metrics, calculate_au_metrics, calculate_emotion_metrics)
from collections import Counter
from torch.cuda.amp import GradScaler, autocast


class BigSmallTrainer(BaseTrainer):
    def __init__(self, config, data_loader, use_amp=False):
        print('\nInitializing BigSmall Multitask Trainer\n')

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

        # Config parameters
        train_cfg = config.TRAIN
        model_cfg = config.MODEL.BIGSMALL
        data_cfg = train_cfg.DATA

        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = train_cfg.MODEL_FILE_NAME
        self.max_epoch_num = train_cfg.EPOCHS
        self.LR = train_cfg.LR
        self.chunk_len = data_cfg.PREPROCESS.CHUNK_LENGTH

        # GPU & TSM setup
        self.num_of_gpu = torch.cuda.device_count()
        self.using_TSM = self.num_of_gpu > 0
        self.frame_depth = model_cfg.FRAME_DEPTH if self.using_TSM else 1
        self.base_len = self.num_of_gpu * self.frame_depth

        # Training state
        self.best_epoch = 0
        self.min_valid_loss = float("inf")

        # Task toggles
        self.train_au = self.train_ppg = self.train_emotion = True
        self.enable_au_eval = self.enable_ppg_eval = self.enable_emotion_eval = True

        # Label config
        self.train_ppg_colname = "pos_ppg"
        self.test_ppg_colname = "ppg"
        self.emotion_colname = "emotion_lf"
        self.set_label_weights = True
        self.emotion_class = 8

        # Load and set label names
        label_path = os.path.dirname(train_cfg.DATA.DATA_PATH)
        self.label_names = self.load_label_names(label_path)
        self.set_label_indices(self.label_names)

        # Initialize model
        self.model = self.define_model().to(self.device)

        # Training setup
        if "train" in data_loader:
            self.num_train_batches = len(data_loader["train"])
            
            if self.set_label_weights:
                au_weights, emotion_weights = self.load_loss_weights()
                self.configure_loss(au_weights, emotion_weights)
            else:
                self.configure_loss()
            
            self.configure_optimizers()

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                       max_lr=self.LR, 
                                                       epochs=self.max_epoch_num, 
                                                       steps_per_epoch=self.num_train_batches)

    def set_label_indices(self, label_names):
        self.au_indices = [i for i, name in enumerate(label_names) if "AU" in name and "int" not in name]
        try:
            self.ppg_train_index = label_names.index(self.train_ppg_colname)
            self.ppg_test_index = label_names.index(self.test_ppg_colname)
        except ValueError as e:
            raise ValueError("Missing required PPG labels") from e
        
        if self.emotion_colname in label_names:
            self.emotion_index = label_names.index(self.emotion_colname)
            self.discrete_emotion_exist = True
        else:
            raise NotImplementedError("Only discrete emotion label is supported.")

    def configure_loss(self, au_weights=None, emotion_weights=None):
        if self.train_au:
            self.criterionAU = torch.nn.BCEWithLogitsLoss(pos_weight=au_weights).to(self.device)
        if self.train_ppg:
            self.criterionPPG = torch.nn.MSELoss().to(self.device)
        if self.train_emotion:
            self.criterionEmotion = torch.nn.CrossEntropyLoss(weight=emotion_weights).to(self.device)

    def load_loss_weights(self, weights_dir=None):
        """
        Load precomputed AU and emotion weights from disk.
        """
        weights_dir = weights_dir

        au_weights_path = os.path.join(weights_dir, "au_weights.pt")
        emotion_weights_path = os.path.join(weights_dir, "emotion_weights.pt")

        au_weights = torch.load(au_weights_path).to(self.device) if os.path.exists(au_weights_path) else None
        emotion_weights = torch.load(emotion_weights_path).to(self.device) if os.path.exists(emotion_weights_path) else None

        if au_weights is not None:
            print(f"[INFO] Loaded AU weights from {au_weights_path}")
        if emotion_weights is not None:
            print(f"[INFO] Loaded emotion weights from {emotion_weights_path}")

        return au_weights, emotion_weights

    @staticmethod
    def load_label_names(label_list_path):
        path = os.path.join(label_list_path, "label_list.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"label_list.txt not found in: {label_list_path}")
        with open(path, "r") as f:
            return [line.strip() for line in f]

    def define_model(self):
        if not self.discrete_emotion_exist:
            raise ValueError("Model definition requires discrete emotion labels.")
        return BigSmall(out_size_au=len(self.au_indices), out_size_emotion=self.emotion_class)

    def format_data_shape(self, data, labels):
        """
        Reshapes and formats the input data and labels for model training or evaluation.

        Args:
            data (list): A list containing two tensors, big_data and small_data, representing different scales of input data.
            labels (torch.Tensor): A tensor containing the labels corresponding to the input data.

        Returns:
            tuple: A tuple containing the reshaped data list ([4D_big_data, 4D_small_data]) and 2D labels tensor.
        """

        # Extract and reshape Big data
        big_data = data[0]
        B, T, C, H, W = big_data.shape
        big_data = big_data.view(B * T, C, H, W)

        # Extract and reshape Small data
        small_data = data[1]
        B, T, C, H, W = small_data.shape
        small_data = small_data.view(B * T, C, H, W)

        # Reshape labels
        B, T, class_num = labels.shape 
        labels = labels.view(B * T, class_num) 

        # TSM truncation if enabled
        if self.using_TSM:
            max_len = (B * T) // self.base_len * self.base_len
            big_data = big_data[:max_len]
            small_data = small_data[:max_len]
            labels = labels[:max_len]

        # Return the formatted data and labels
        return [big_data, small_data], labels

    def send_data_to_device(self, data, labels):
        data = tuple(d.to(self.device, non_blocking=True) for d in data)
        labels = labels.to(self.device, non_blocking=True)
        return data, labels

    def compute_loss(self, outputs, labels, mode="Train"):
        au_output, ppg_output, emotion_output = outputs
        total_loss = 0.0
        losses = {"AU": None, "PPG": None, "Emotion": None}

        if self.train_au:
            au_target = labels[:, self.au_indices]
            losses["AU"] = self.criterionAU(au_output, au_target)
            total_loss += losses["AU"]

        if self.train_ppg:
            ppg_target = labels[:, self.ppg_train_index].unsqueeze(-1)
            losses["PPG"] = self.criterionPPG(ppg_output, ppg_target)
            total_loss += losses["PPG"]

        if self.train_emotion:
            emotion_target = labels[:, self.emotion_index].long()
            losses["Emotion"] = self.criterionEmotion(emotion_output, emotion_target)
            total_loss += losses["Emotion"]
        
        print()
        # Optional: concise logging loop
        for task, loss_val in losses.items():
            if loss_val is not None:
                print(f"{mode} {task} Loss: {loss_val}")

        return total_loss, losses["AU"], losses["PPG"], losses["Emotion"]

    def train(self, data_loader):
        train_loader = data_loader.get("train")
        if train_loader is None:
            raise ValueError("No training data provided.")

        print('\nStarting Training Routine\n')
        mean_train_losses, mean_valid_losses, lrs = [], [], []

        for epoch in range(self.max_epoch_num):
            print(f"==== Training Epoch: {epoch} ====")
            self.model.train()
            epoch_losses = {"Loss": [], "AU": [], "PPG": [], "Emotion": []}
            running_loss = 0.0

            tbar = tqdm(train_loader, ncols=80, desc=f"\nTrain Epoch: {epoch}")
            for idx, (data, labels, _, _) in enumerate(tbar):
                data, labels = self.send_data_to_device(*self.format_data_shape(data, labels))
                
                total_loss, au_loss, ppg_loss, emotion_loss = self.backward_step(data, labels)
                self.scheduler.step()

                lrs.append(self.scheduler.get_last_lr()[0])
                epoch_losses["Loss"].append(total_loss.item())
                epoch_losses["AU"].append(au_loss.item())
                epoch_losses["PPG"].append(ppg_loss.item())
                epoch_losses["Emotion"].append(emotion_loss.item())
                running_loss += total_loss.item()

                if idx % 100 == 99:
                    print(f"Epoch [{epoch}] | Batch [{idx + 1}] | Avg Loss (Last 100): {running_loss / 100}")
                    running_loss = 0.0

                tbar.set_postfix(loss=total_loss.item(), lr=self.optimizer.param_groups[0]["lr"])
                

            mean_train_losses.append(np.mean(epoch_losses["Loss"]))
            self.save_model(epoch)

            if self.config.VALID.RUN_VALIDATION:
                val_loss = self.valid(data_loader, epoch, save_best_model=True)
                mean_valid_losses.append(val_loss)

        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_train_losses, mean_valid_losses, lrs, self.config)

    def backward_step(self, data, labels):
        """
        Performs the forward, backward, and optimizer step with AMP support.
        
        Returns:
            tuple: (total_loss, au_loss, ppg_loss, emotion_loss)
        """
        self.optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                outputs = self.model(data)
                total_loss, au_loss, ppg_loss, emotion_loss = self.compute_loss(outputs, labels)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(data)
            total_loss, au_loss, ppg_loss, emotion_loss = self.compute_loss(outputs, labels)
            total_loss.backward()
            self.optimizer.step()

        return total_loss, au_loss, ppg_loss, emotion_loss

    def save_model(self, index=None, best_model=False):
        """
        Saves the model state to a file.

        Args:
            index (int, optional): Epoch index for naming the checkpoint.
            best (bool): If True, saves as 'best_model.pth' regardless of index.
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if best_model:
            best_model_filename = f"{self.model_file_name}_Epoch{index}_BM.pth"
            model_path = os.path.join(self.model_dir, "best_model", best_model_filename)
        elif index is not None:
            model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        else:
            raise ValueError("Either index must be provided or best must be True.")

        # Ensure the full directory path exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        torch.save(self.model.state_dict(), model_path)
        print(f"Saved Model Path: {model_path}\n")

    def valid(self, data_loader, epoch=None, save_best_model=False):
        """
        Evaluates the model on the validation set, logs metrics, and optionally saves the best model.

        Args:
            data_loader (dict): Contains the validation DataLoader with key "valid".
            epoch (int): Current epoch index.
            save_best_model (bool): Whether to save the best model.

        Returns:
            tuple: (mean_total_loss, evaluation_metrics_dict)
        """
        valid_loader = data_loader.get("valid")
        if valid_loader is None:
            raise ValueError("No Validation Data Provided.")

        print("\n=== Validating ===\n")
        self.model.eval()

        losses = {"total": [], "au": [], "ppg": [], "emotion": []}
        preds_dict = {"au": {}, "ppg": {}, "emotion": {}}
        labels_dict = {"au": {}, "ppg": {}, "emotion": {}}

        with torch.no_grad():
            for data, labels, subject_ids, chunk_ids in tqdm(valid_loader, ncols=80, desc="\nValidation"):
                batch_size = labels.shape[0]
                data, labels = self.send_data_to_device(*self.format_data_shape(data, labels))

                outputs = self.model(data)
                loss, au_loss, ppg_loss, emotion_loss = self.compute_loss(outputs, labels, mode="Valid")

                losses["total"].append(loss.item())
                losses["au"].append(au_loss.item())
                losses["ppg"].append(ppg_loss.item())
                losses["emotion"].append(emotion_loss.item())

                preds_batch, labels_batch = self.process_predictions(outputs, labels, subject_ids, chunk_ids, batch_size)
                self.update_predictions(preds_dict, labels_dict, preds_batch, labels_batch)

        mean_losses = {k: np.mean(v) for k, v in losses.items()}
        self.evaluate_predictions(preds_dict, labels_dict)

        if epoch is not None:
            print(f"Validation Loss @ Epoch {epoch}: {mean_losses['total']:.4f}")
            if save_best_model:
                if mean_losses["total"] < self.min_valid_loss:
                    self.best_epoch = epoch
                    self.min_valid_loss = mean_losses["total"]
                    self.save_model(epoch, best_model=True)
                    print(f"New Best Model @ Epoch {epoch} | Loss: {mean_losses['total']:.4f}")
                else:
                    print(f"No Update | Best: Epoch {self.best_epoch}, Loss: {self.min_valid_loss:.4f}")

        return mean_losses["total"]

    def test(self, data_loader):
        """Runs model evaluation on the testing dataset with metrics and loss tracking."""
        test_loader = data_loader.get("test")
        if test_loader is None:
            raise ValueError("No Testing Data Provided.")

        print("\n=== Testing ===\n")

        # Load model
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.INFERENCE.MODEL_PATH
            print("Using Pretrained Model for Testing.")
        else:
            model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{self.used_epoch}.pth")
            print("Using Trained Model Checkpoint for Testing.")

        print(f"Model Path: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model Not Found At: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        preds_dict = {"au": {}, "ppg": {}, "emotion": {}}
        labels_dict = {"au": {}, "ppg": {}, "emotion": {}}

        with torch.no_grad():
            for data, labels, subject_ids, chunk_ids in tqdm(test_loader, ncols=80, desc="\nTesting"):
                batch_size = labels.shape[0]
                if batch_size == 0:
                    continue

                data, labels = self.send_data_to_device(*self.format_data_shape(data, labels))
                outputs = self.model(data)

                preds_batch, labels_batch = self.process_predictions(outputs, labels, subject_ids, chunk_ids, batch_size)
                self.update_predictions(preds_dict, labels_dict, preds_batch, labels_batch)

        self.evaluate_predictions(preds_dict, labels_dict)

    def update_predictions(self, preds_dict, labels_dict, preds_batch, labels_batch):
        """Efficiently updates the full prediction dictionaries with current batch."""
        for task in preds_dict:
            preds_dict[task].update(preds_batch[task])
            labels_dict[task].update(labels_batch[task])

    def process_predictions(self, outputs, labels, subject_ids, chunk_ids, batch_size):
        prediction = {"au": {}, "ppg": {}, "emotion": {}}
        ground_truth = {"au": {}, "ppg": {}, "emotion": {}}

        au_output = torch.sigmoid(outputs[0]) if self.train_au else None
        ppg_output = outputs[1] if self.train_ppg else None
        emotion_output = torch.softmax(outputs[2], dim=1) if self.train_emotion else None

        for idx in range(batch_size):
            if idx * self.chunk_len >= batch_size and self.using_TSM:
                continue

            sid = subject_ids[idx]
            cid = int(chunk_ids[idx])
            start, end = idx * self.chunk_len, (idx + 1) * self.chunk_len

            if self.train_au:
                prediction["au"].setdefault(sid, {})[cid] = au_output[start:end].cpu()
                ground_truth["au"].setdefault(sid, {})[cid] = labels[start:end, self.au_indices].cpu()

            if self.train_ppg:
                prediction["ppg"].setdefault(sid, {})[cid] = ppg_output[start:end].cpu()
                ground_truth["ppg"].setdefault(sid, {})[cid] = labels[start:end, self.ppg_test_index].unsqueeze(-1).cpu()

            if self.train_emotion:
                prediction["emotion"].setdefault(sid, {})[cid] = emotion_output[start:end].cpu()
                ground_truth["emotion"].setdefault(sid, {})[cid] = labels[start:end, self.emotion_index].cpu()

        return prediction, ground_truth

    def evaluate_predictions(self, preds_dict, labels_dict):
        class_emotions = ["Neutral", "Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust"]
        eval_tasks = {
            "ppg": {
                "enabled": self.enable_ppg_eval,
                "function": calculate_ppg_metrics,
                "extra_args": []
            },
            "au": {
                "enabled": self.enable_au_eval,
                "function": calculate_au_metrics,
                "extra_args": [[name for name in self.label_names if "AU" in name and "int" not in name]]
            },
            "emotion": {
                "enabled": self.enable_emotion_eval,
                "function": calculate_emotion_metrics,
                "extra_args": [class_emotions]
            }
        }

        for task, config in eval_tasks.items():
            if not config["enabled"] or task not in preds_dict or task not in labels_dict:
                continue
            try:
                config["function"](preds_dict[task], labels_dict[task], self.config, *config["extra_args"])
            except Exception as e:
                print(f"[Error] Failed to compute {task.upper()} metrics: {e}")

    def _reform_data_from_dict(self, data, flatten=True):
        """
        Reformats nested dictionary {subject_id: {chunk_id: tensor}} into a single NumPy array.

        Args:
            data (dict or tensor): Input data.
            flatten (bool): Whether to flatten the result into 1D.

        Returns:
            np.ndarray: Combined data array.
        """
        tensors = []

        if isinstance(data, dict):
            for subject_dict in data.values():
                if isinstance(subject_dict, dict):
                    for chunk_tensor in subject_dict.values():
                        if isinstance(chunk_tensor, torch.Tensor):
                            tensors.append(chunk_tensor)
                elif isinstance(subject_dict, torch.Tensor):  # Fallback
                    tensors.append(subject_dict)

            if not tensors:
                return np.array([])

            try:
                stacked = torch.cat(tensors, dim=0)
            except Exception as e:
                print(f"[Error] Failed to concatenate tensors: {e}")
                return np.array([])
        elif isinstance(data, torch.Tensor):
            stacked = data
        else:
            return np.array([])

        arr = stacked.cpu().numpy()
        return arr.reshape(-1) if flatten else arr

    def debug_emotion_distribution(self, preds_dict, labels_dict, expected_num_classes=7):
        """
        Debugs the class distribution in predictions and labels for emotion recognition.

        Args:
            preds_dict (dict): Nested dict of predictions {subject: {chunk: tensor}}.
            labels_dict (dict): Nested dict of ground-truth labels.
            expected_num_classes (int): Number of emotion classes. Default: 7
        """
        print("\n========================")
        print("🔍 Emotion Debug Utility")
        print("========================")

        # Reform data
        y_pred_logits = self._reform_data_from_dict(preds_dict, flatten=False)
        y_true = self._reform_data_from_dict(labels_dict, flatten=True)

        # Check logits shape
        print(f"\n🧠 Logits shape: {y_pred_logits.shape}")
        if len(y_pred_logits.shape) == 2 and y_pred_logits.shape[1] > 1:
            pred_classes = np.argmax(y_pred_logits, axis=1)
        else:
            pred_classes = y_pred_logits.astype(int)

        # Label distribution
        label_counter = Counter(y_true.tolist())
        print("\n🧾 Ground Truth Label Distribution:")
        for i in range(expected_num_classes):
            count = label_counter.get(i, 0)
            print(f"  Class {i}: {count} samples")

        # Prediction distribution
        pred_counter = Counter(pred_classes.tolist())
        print("\n📊 Prediction Class Distribution:")
        for i in range(expected_num_classes):
            count = pred_counter.get(i, 0)
            print(f"  Class {i}: {count} predictions")

        # Sanity check
        if set(label_counter.keys()) != set(range(expected_num_classes)):
            print("\n⚠️ Warning: Some classes are missing in the labels.")
        if set(pred_counter.keys()) != set(range(expected_num_classes)):
            print("⚠️ Warning: Some classes are missing in the predictions.")

        print("========================\n")