import numpy as np
import torch
import pandas as pd
import os

class LabelHelper:
    def __init__(self, config, discrete_emotion=True, num_emotion_classes=8):
        self.config = config
        self.discrete_emotion_exist = discrete_emotion
        self.discrete_emotion_class = num_emotion_classes if discrete_emotion else None
        self.au_indices = []
        self.emotion_index = None

    def compute_weights_and_save(self, save_dir=None):
        """
        Computes and saves AU and emotion weights instead of returning them.

        Args:
            save_dir (str): Directory to save the weights. If None, uses config.MODEL.MODEL_DIR.
        """
        if save_dir is None:
            save_dir = self.config.MODEL.MODEL_DIR
        os.makedirs(save_dir, exist_ok=True)

        label_list_path = os.path.dirname(self.config.TRAIN.DATA.DATA_PATH)
        label_names = self._load_label_names(label_list_path)
        self._set_label_indices(label_names)

        au_weights, emotion_weights = self._compute_weights()

        if au_weights is not None:
            torch.save(au_weights, os.path.join(save_dir, "au_weights.pt"))
            print(f"[INFO] AU weights saved to {os.path.join(save_dir, 'au_weights.pt')}")

        if emotion_weights is not None:
            torch.save(emotion_weights, os.path.join(save_dir, "emotion_weights.pt"))
            print(f"[INFO] Emotion weights saved to {os.path.join(save_dir, 'emotion_weights.pt')}")

    def _load_label_names(self, label_list_path):
        with open(os.path.join(label_list_path, "label_list.txt"), "r") as f:
            return [line.strip() for line in f]

    def _set_label_indices(self, label_names):
        self.au_indices = [i for i, name in enumerate(label_names) if "AU" in name and "int" not in name]

        if "emotion_lf" in label_names or "emotion_quadrant" in label_names:
            self.emotion_index = label_names.index("emotion_lf") if "emotion_lf" in label_names else label_names.index("emotion_quadrant")
            if not self.discrete_emotion_exist:
                raise ValueError("`discrete_emotion_exist` is False, but emotion label found in label names.")
        else:
            if self.discrete_emotion_exist:
                raise NotImplementedError("Discrete emotion label not found in label names.")

    def _get_label_paths(self):
        df = pd.read_csv(self.config.TRAIN.DATA.FILE_LIST_PATH, usecols=["label"])
        return [os.path.join(self.config.TRAIN.DATA.CACHED_PATH, os.path.basename(p)) for p in df["label"].values]

    def _compute_weights(self):
        """
        Computes class-balancing weights for AU detection and emotion classification.

        Returns:
            au_weights (torch.Tensor): Weights for AU loss (shape: [num_AUs]).
            emotion_weights (torch.Tensor or None): Weights for emotion classification (shape: [discrete_emotion_class]).
        """
        label_paths = self._get_label_paths()

        au_pos = np.zeros(len(self.au_indices), dtype=np.float64)
        au_neg = np.zeros(len(self.au_indices), dtype=np.float64)
        emotion_counts = np.zeros(self.discrete_emotion_class, dtype=np.float64) if self.discrete_emotion_exist else None

        for i, path in enumerate(label_paths):
            print(i)
            labels = np.load(path)

            if self.discrete_emotion_exist:
                emotion = np.asarray(labels[:, self.emotion_index]).astype(int).flatten()
                if np.any(emotion < 0):
                    raise ValueError(f"Negative values detected in emotion labels at {path}")
                emotion_counts += np.bincount(emotion, minlength=self.discrete_emotion_class)

            try:
                au_data = np.stack([np.asarray(labels[:, i]) for i in self.au_indices], axis=-1)
            except Exception as e:
                raise ValueError(f"Error stacking AU labels at {path}: {e}")

            au_pos += np.sum(au_data == 1, axis=0)
            au_neg += np.sum(au_data == 0, axis=0)

        au_weights = au_neg / (au_pos + 1e-5)
        au_weights = torch.tensor(au_weights, dtype=torch.float32)

        emotion_weights = None
        if self.discrete_emotion_exist:
            raw_weights = emotion_counts.sum() / (emotion_counts + 1e-5)
            normalized_weights = raw_weights / raw_weights.sum()
            emotion_weights = torch.tensor(normalized_weights, dtype=torch.float32)

        return au_weights, emotion_weights