from sklearn.model_selection import StratifiedShuffleSplit
import os
import pandas as pd
import numpy as np
import glob
import re
from collections import Counter

def stratified_subject_split(data_dirs, valid_size=0.10, test_size=0.10, seed=70):
    emotion_mapping = {
        "Happiness": 0,
        "Anger": 1,
        "Fear": 2,
        "Sadness": 3,
        "Calm": 4,
        "Neutral": 5
    }

    labels = []
    for d in data_dirs:
        path = os.path.join(d['path'], "raw_modified_vad.csv")
        modified_vad = pd.read_csv(path, skiprows=1, header=None)

        # Extract last column (text emotions)
        emotion_label = modified_vad.iloc[:, -1].astype(str).str.strip()

        # Map string labels to integers
        mapped = emotion_label.map(emotion_mapping).dropna().astype(int).values

        if mapped.size == 0:
            raise ValueError(f"No valid emotion labels in: {path}")

        dominant = np.bincount(mapped).argmax()
        labels.append(dominant)

    labels = np.array(labels)
    indices = np.arange(len(data_dirs))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size + test_size, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    temp_labels = labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (valid_size + test_size), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, temp_labels))

    train_dirs = [data_dirs[i] for i in train_idx]
    val_dirs = [data_dirs[i] for i in temp_idx[val_idx]]
    test_dirs = [data_dirs[i] for i in temp_idx[test_idx]]

    return train_dirs, val_dirs, test_dirs

def stratified_subject_split_with_print(data_dirs, valid_size=0.10, test_size=0.10, seed=70):
    emotion_mapping = {
        "Happiness": 0,
        "Anger": 1,
        "Fear": 2,
        "Sadness": 3,
        "Calm": 4,
        "Neutral": 5
    }

    reverse_mapping = {v: k for k, v in emotion_mapping.items()}

    labels = []
    dominant_labels = []

    for d in data_dirs:
        path = os.path.join(d['path'], "raw_modified_vad.csv")
        modified_vad = pd.read_csv(path, skiprows=1, header=None)

        # Extract and map last column
        emotion_label = modified_vad.iloc[:, -1].astype(str).str.strip()
        mapped = emotion_label.map(emotion_mapping).dropna().astype(int).values

        if mapped.size == 0:
            raise ValueError(f"No valid emotion labels in: {path}")

        dominant = np.bincount(mapped).argmax()
        labels.append(dominant)
        dominant_labels.append((d['index'], dominant))  # For tracing

    labels = np.array(labels)
    indices = np.arange(len(data_dirs))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size + test_size, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    temp_labels = labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (valid_size + test_size), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, temp_labels))

    # Build split directories
    train_dirs = [data_dirs[i] for i in train_idx]
    val_dirs = [data_dirs[i] for i in temp_idx[val_idx]]
    test_dirs = [data_dirs[i] for i in temp_idx[test_idx]]

    # Analyze dominant emotion distributions
    def get_dominants(dir_list):
        return [dict(dominant_labels)[d['index']] for d in dir_list]

    train_counts = Counter(get_dominants(train_dirs))
    val_counts = Counter(get_dominants(val_dirs))
    test_counts = Counter(get_dominants(test_dirs))

    # Print distributions
    def print_distribution(name, counter):
        print(f"\n{name} Emotion Distribution:")
        for k in sorted(counter):
            print(f"  {reverse_mapping[k]} ({k}): {counter[k]}")

    print_distribution("Train", train_counts)
    print_distribution("Validation", val_counts)
    print_distribution("Test", test_counts)

    return train_dirs, val_dirs, test_dirs

def balanced_stratified_split(data_dirs, valid_size=0.10, test_size=0.10, seed=70):
    emotion_mapping = {"Happiness": 0, 
                       "Anger": 1, 
                       "Fear": 2,
                       "Sadness": 3, 
                       "Calm": 4, 
                       "Neutral": 5}

    def get_dominant_label(path):
        df = pd.read_csv(path, skiprows=1, header=None)
        labels = df.iloc[:, -1].astype(str).str.strip().map(emotion_mapping).dropna().astype(int)
        if labels.empty:
            raise ValueError(f"No valid emotion labels in: {path}")
        return np.bincount(labels).argmax()

    # Step 1: Get dominant emotion label per subject
    dominant_labels = np.array([
        get_dominant_label(os.path.join(d['path'], "raw_modified_vad.csv"))
        for d in data_dirs
    ])
    indices = np.arange(len(data_dirs))

    # Step 2: Split into train and temp (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size + test_size, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, dominant_labels))

    # Step 3: Split temp into val and test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (valid_size + test_size), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, dominant_labels[temp_idx]))

    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    def ensure_label_coverage(idx, others):
        idx_labels = set(dominant_labels[idx])
        missing = set(emotion_mapping.values()) - idx_labels
        if not missing:
            return idx, others

        for label in missing:
            for other in others:
                for i in other:
                    if dominant_labels[i] == label:
                        idx.append(i)
                        other.remove(i)
                        break
                else:
                    continue
                break
        return idx, others

    # Ensure label coverage in all splits
    train_idx, [val_idx, test_idx] = ensure_label_coverage(list(train_idx), [list(val_idx), list(test_idx)])
    val_idx, [train_idx, test_idx] = ensure_label_coverage(list(val_idx), [list(train_idx), list(test_idx)])
    test_idx, [train_idx, val_idx] = ensure_label_coverage(list(test_idx), [list(train_idx), list(val_idx)])

    return (
        [data_dirs[i] for i in sorted(train_idx)],
        [data_dirs[i] for i in sorted(val_idx)],
        [data_dirs[i] for i in sorted(test_idx)]
    )

def balanced_stratified_split_with_print(data_dirs, valid_size=0.10, test_size=0.10, seed=70):
    emotion_mapping = {"Happiness": 0, 
                       "Anger": 1, 
                       "Fear": 2,
                       "Sadness": 3, 
                       "Calm": 4, 
                       "Neutral": 5}

    def get_dominant_label(path):
        df = pd.read_csv(path, skiprows=1, header=None)
        labels = df.iloc[:, -1].astype(str).str.strip().map(emotion_mapping).dropna().astype(int)
        if labels.empty:
            raise ValueError(f"No valid emotion labels in: {path}")
        return np.bincount(labels).argmax()

    # Precompute dominant labels per subject
    dominant_labels = []
    for d in data_dirs:
        csv_path = os.path.join(d['path'], "raw_modified_vad.csv")
        dominant_labels.append(get_dominant_label(csv_path))
    dominant_labels = np.array(dominant_labels)
    indices = np.arange(len(data_dirs))

    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=valid_size + test_size, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, dominant_labels))

    # Second split: val vs test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (valid_size + test_size), random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, dominant_labels[temp_idx]))

    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    def ensure_label_coverage(name, idx, others):
        label_set = set(dominant_labels[idx])
        missing = set(emotion_mapping.values()) - label_set
        if not missing:
            return idx, others
        print(f"[Fix] {name} missing labels: {missing}")
        for label in missing:
            for other in others:
                for i in other:
                    if dominant_labels[i] == label:
                        idx.append(i)
                        other.remove(i)
                        print(f"Moved subject with label {label} to {name}")
                        break
                else:
                    continue
                break
        return idx, others

    # Ensure each split has all labels
    train_idx, [val_idx, test_idx] = ensure_label_coverage("Train", list(train_idx), [list(val_idx), list(test_idx)])
    val_idx, [train_idx, test_idx] = ensure_label_coverage("Valid", list(val_idx), [list(train_idx), list(test_idx)])
    test_idx, [train_idx, val_idx] = ensure_label_coverage("Test", list(test_idx), [list(train_idx), list(val_idx)])

    def print_distribution(name, idx):
        counts = Counter(dominant_labels[idx])
        print(f"\n{name} Emotion Distribution:")
        for k in sorted(emotion_mapping.values()):
            label = next(key for key, val in emotion_mapping.items() if val == k)
            print(f"  {label:<9} ({k}): {counts.get(k, 0)}")

    # Print label distributions
    print_distribution("Train", train_idx)
    print_distribution("Valid", val_idx)
    print_distribution("Test", test_idx)

    return (
        [data_dirs[i] for i in sorted(train_idx)],
        [data_dirs[i] for i in sorted(val_idx)],
        [data_dirs[i] for i in sorted(test_idx)]
    )

def get_raw_data(data_path, dataset_name):
    """
    Returns a list of dictionaries for each subject in the dataset directory.

    Args:
        data_path (str): Path to the dataset root directory.
        dataset_name (str): Name of the dataset (for error message context).

    Returns:
        List[Dict[str, str]]: List of {'index': subject_id, 'path': full_path}.
    """
    pattern = os.path.join(data_path, "subject*")
    data_dirs = glob.glob(pattern)

    if not data_dirs:
        raise ValueError(f"{dataset_name} data paths empty!")

    subject_regex = re.compile(r'subject\d+')

    dirs = [
        {"index": match.group(0), "path": path}
        for path in data_dirs
        if (match := subject_regex.search(path))
    ]

    return dirs