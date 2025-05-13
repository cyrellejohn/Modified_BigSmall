import numpy as np
import torch
from evaluation.metrics import calculate_metrics, reform_data_to_numpy
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, matthews_corrcoef)


def calculate_ppg_metrics(predictions, labels, config):
    print('=====================')
    print('==== PPG Metrics ====')
    print('=====================')
    calculate_metrics(predictions, labels, config)
    print()

def calculate_au_metrics(predictions, labels, config, au_colname):
    """
    Calculate AU Metrics (AU F1, Precision, Mean F1, Mean Acc, Mean Precision).

    Parameters:
    -----------
    predictions : dict
        Dictionary of predicted AU values per trial.
    labels : dict
        Dictionary of ground truth AU values per trial.
    config : object
        Config object containing TEST.METRICS list.
    au_colname : list
        List of AU names.
    debug : bool
        If True, prints detailed debug information.

    Returns:
    --------
    None
        Prints metrics and their averages to console.
    """
    if 'AU_METRICS' not in config.TEST.METRICS:
        print("AU_METRICS not in TEST.METRICS, skipping AU evaluation.")
        return

    au_colname_len = len(au_colname)

    # Convert predictions and labels to numpy arrays
    predictions = {k: reform_data_to_numpy(v, flatten=False) for k, v in predictions.items()}
    labels = {k: reform_data_to_numpy(v, flatten=False) for k, v in labels.items()}

    # Concatenate all trials
    all_predictions = np.concatenate(list(predictions.values()), axis=0)
    all_labels = np.concatenate(list(labels.values()), axis=0)

    print('\n==================================')
    print('========== AU METRICS ============')
    print('==================================')
    print('AU Precision, Recall, F1, Accuracy')

    metric_dict = {}
    avg_prec = avg_recall = avg_f1 = avg_acc = 0

    for i, au in enumerate(au_colname):
        y_pred = (all_predictions[:, i] >= 0.5).astype(int)
        y_true = all_labels[:, i].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, beta=1.0, zero_division=0)

        # Index 1 corresponds to the positive class
        precision = precision[1] * 100 if len(precision) > 1 else 0.0
        recall = recall[1] * 100 if len(recall) > 1 else 0.0
        f1 = f1[1] * 100 if len(f1) > 1 else 0.0

        acc = (y_pred == y_true).mean() * 100

        metric_dict[au] = (precision, recall, f1, acc)
        avg_prec += precision
        avg_recall += recall
        avg_f1 += f1
        avg_acc += acc

        print(f"AU: {au}")
        print(f"  Metrics:")
        print(f"    - Precision: {precision:.2f}%")
        print(f"    - Recall: {recall:.2f}%")
        print(f"    - F1 Score: {f1:.2f}%")
        print(f"    - Accuracy: {acc:.2f}%")
        print(f"  Prediction Details: ")
        print(f"  0: Not Activated | 1: Activated")
        print(f"    - Unique Predictions: {np.unique(y_pred)}")
        print(f"    - Unique Labels: {np.unique(y_true)}")
        print(f"    - Predictions where 0: {np.sum(y_pred == 0)}")
        print(f"    - Predictions where 1: {np.sum(y_pred == 1)}")
        print(f"    - Labels where 0: {np.sum(y_true == 0)}")
        print(f"    - Labels where 1: {np.sum(y_true == 1)}")
        print("")

    avg_prec /= au_colname_len
    avg_recall /= au_colname_len
    avg_f1 /= au_colname_len
    avg_acc /= au_colname_len

    metric_dict[f'{au_colname_len}AU_AvgPrec'] = avg_prec
    metric_dict[f'{au_colname_len}AU_AvgRecall'] = avg_recall
    metric_dict[f'{au_colname_len}AU_AvgF1'] = avg_f1
    metric_dict[f'{au_colname_len}AU_AvgAcc'] = avg_acc

    print(f'\nMean {au_colname_len} AU Precision: {avg_prec:.2f}')
    print(f'Mean {au_colname_len} AU Recall: {avg_recall:.2f}')
    print(f'Mean {au_colname_len} AU F1: {avg_f1:.2f}')
    print(f'Mean {au_colname_len} AU Accuracy: {avg_acc:.2f}\n')

def flatten_nested_dict_to_tensor(data, flatten=True):
    """
    Flatten a nested dictionary {subject: {chunk: tensor}} into a numpy array.

    Args:
        data (dict or torch.Tensor): Input data.
        flatten (bool): Whether to flatten the final numpy array.

    Returns:
        np.ndarray: Flattened or shaped numpy array.
    """
    if isinstance(data, torch.Tensor):
        data_tensor = data
    elif isinstance(data, dict):
        values = [
            chunk for subject_chunks in data.values()
            if isinstance(subject_chunks, dict)
            for chunk in subject_chunks.values()
        ]
        if not values:
            return np.empty(0)
        data_tensor = torch.cat(values, dim=0)
    else:
        return np.empty(0)

    array = data_tensor.cpu().numpy()
    return array.reshape(-1) if flatten else array

def print_per_class_report(y_true, y_pred, labels_range, class_names, class_precision, 
                           class_recall, class_f1, support):
    """
    Print per-class metrics with OvR MCC, recall, precision, F1, and true binary accuracy.
    """
    print("\n==== Detailed Per-Class Metrics ====\n")
    for i, label in enumerate(class_names):
        y_true_bin = (y_true == labels_range[i]).astype(int)
        y_pred_bin = (y_pred == labels_range[i]).astype(int)

        TP = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
        FN = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
        TN = np.sum((y_pred_bin == 0) & (y_true_bin == 0))

        per_class_acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
        per_class_mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

        print(f"Class: {label}")
        print(f"  Metrics:")
        print(f"    - Accuracy (OvR): {per_class_acc * 100:.2f}%")
        print(f"    - Matthews Correlation Coefficient (MCC): {per_class_mcc:.4f}")
        print(f"    - Precision: {class_precision[i] * 100:.2f}%")
        print(f"    - Recall: {class_recall[i] * 100:.2f}%")
        print(f"    - F1 Score: {class_f1[i] * 100:.2f}%")
        print(f"    - Support: {support[i]}")
        print(f"  Confusion Components (OvR):")
        print(f"    - True Positives (TP): {TP}")
        print(f"    - False Positives (FP): {FP}")
        print(f"    - False Negatives (FN): {FN}")
        print(f"    - True Negatives (TN): {TN}")
        print("")

def calculate_emotion_metrics(predictions, labels, config=None, class_emotions=None):
    """
    Compute and print emotion classification metrics.

    Args:
        predictions (dict): {subject_id: {chunk_id: logits or predictions}}
        labels (dict): {subject_id: {chunk_id: class indices}}
        config (object): Optional config object
        class_emotions (list): List of emotion class names
    """
    print("==========================")
    print("==== EMOTION METRICS =====")
    print("==========================")

    y_pred_logits = flatten_nested_dict_to_tensor(predictions, flatten=False)
    y_true = flatten_nested_dict_to_tensor(labels, flatten=True).astype(int)

    if y_pred_logits.shape[0] != y_true.shape[0]:
        raise ValueError(f"Shape mismatch: predictions ({y_pred_logits.shape[0]}) vs labels ({y_true.shape[0]})")

    y_pred = (np.argmax(y_pred_logits, axis=1)
              if y_pred_logits.ndim == 2 and y_pred_logits.shape[1] > 1
              else y_pred_logits.reshape(-1).astype(int))

    if class_emotions:
        labels_range = list(range(len(class_emotions)))
    else:
        labels_range = sorted(list(set(y_true) | set(y_pred)))
        class_emotions = [str(i) for i in labels_range]

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=labels_range
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, labels=labels_range
    )

    class_p, class_r, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=labels_range
    )

    print("\nClassification Report (Detailed):")
    print(classification_report(y_true, y_pred,
                                labels=labels_range,
                                target_names=class_emotions,
                                digits=4,
                                zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels_range))

    print("\n==== Summary Metrics ====")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"Macro Precision:    {macro_p:.4f}")
    print(f"Macro Recall:       {macro_r:.4f}")
    print(f"Macro F1-score:     {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f}")
    print(f"Weighted Recall:    {weighted_r:.4f}")
    print(f"Weighted F1-score:  {weighted_f1:.4f}")

    print_per_class_report(
        y_true, y_pred, labels_range, class_emotions,
        class_p, class_r, class_f1, support
    )