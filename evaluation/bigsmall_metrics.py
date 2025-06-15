import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, matthews_corrcoef
)
from evaluation.metrics import calculate_metrics, reform_data_to_numpy


def calculate_ppg_metrics(predictions, labels, config):
    print('=====================')
    print('==== PPG Metrics ====')
    print('=====================')
    return calculate_metrics(predictions, labels, config)


def calculate_au_metrics(predictions, labels, config, au_colname):
    if 'AU_METRICS' not in config.TEST.METRICS:
        print("AU_METRICS not in TEST.METRICS, skipping AU evaluation.")
        return

    predictions = {k: reform_data_to_numpy(v, flatten=False) for k, v in predictions.items()}
    labels = {k: reform_data_to_numpy(v, flatten=False) for k, v in labels.items()}

    all_predictions = np.concatenate(list(predictions.values()), axis=0)
    all_labels = np.concatenate(list(labels.values()), axis=0)

    print('\n==================================')
    print('========== AU METRICS ============')
    print('==================================')
    print('AU Precision, Recall, F1, Accuracy')

    avg_prec = avg_recall = avg_f1 = avg_acc = 0
    metric_dict = {}

    for i, au in enumerate(au_colname):
        y_pred = (all_predictions[:, i] >= 0.5).astype(int)
        y_true = all_labels[:, i].astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, beta=1.0, zero_division=0, average=None
        )

        precision = precision[1] * 100 if len(precision) > 1 else 0.0
        recall = recall[1] * 100 if len(recall) > 1 else 0.0
        f1 = f1[1] * 100 if len(f1) > 1 else 0.0
        acc = (y_pred == y_true).mean() * 100

        avg_prec += precision
        avg_recall += recall
        avg_f1 += f1
        avg_acc += acc

        metric_dict[au] = (precision, recall, f1, acc)

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

    au_len = len(au_colname)
    print(f'\nMean {au_len} AU Precision: {avg_prec / au_len:.2f}')
    print(f'Mean {au_len} AU Recall: {avg_recall / au_len:.2f}')
    print(f'Mean {au_len} AU F1: {avg_f1 / au_len:.2f}')
    print(f'Mean {au_len} AU Accuracy: {avg_acc / au_len:.2f}\n')

    return {
        au: {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc
        }
        for au, (precision, recall, f1, acc) in metric_dict.items()
    }


def flatten_nested_dict_to_tensor(data, flatten=True):
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        values = [v for subj in data.values() for v in (subj.values() if isinstance(subj, dict) else [])]
        if not values:
            return np.empty(0)
        tensor = torch.cat(values, dim=0)
    else:
        return np.empty(0)
    arr = tensor.cpu().numpy()
    return arr.reshape(-1) if flatten else arr


def calculate_emotion_metrics(predictions, labels, config=None, class_emotions=None):
    print("==========================")
    print("==== EMOTION METRICS =====")
    print("==========================")

    y_pred_logits = flatten_nested_dict_to_tensor(predictions, flatten=False)
    y_true = flatten_nested_dict_to_tensor(labels, flatten=True).astype(int)

    if y_pred_logits.shape[0] != y_true.shape[0]:
        raise ValueError(f"Shape mismatch: predictions ({y_pred_logits.shape[0]}) vs labels ({y_true.shape[0]})")

    y_pred = np.argmax(y_pred_logits, axis=1) if y_pred_logits.ndim == 2 else y_pred_logits.reshape(-1).astype(int)

    labels_range = list(range(len(class_emotions))) if class_emotions else sorted(set(y_true) | set(y_pred))
    class_emotions = class_emotions or [str(i) for i in labels_range]

    print("\nClassification Report (Detailed):")
    print(classification_report(y_true, y_pred, labels=labels_range, target_names=class_emotions, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels_range))

    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, labels=labels_range)

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0, labels=labels_range)

    print("\n==== Summary Metrics ====")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"Macro Precision:    {macro_p:.4f}")
    print(f"Macro Recall:       {macro_r:.4f}")
    print(f"Macro F1-score:     {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f}")
    print(f"Weighted Recall:    {weighted_r:.4f}")
    print(f"Weighted F1-score:  {weighted_f1:.4f}\n")

    print("==== Detailed Per-Class Metrics ====")
    for i, cls in enumerate(class_emotions):
        y_true_bin = (y_true == labels_range[i]).astype(int)
        y_pred_bin = (y_pred == labels_range[i]).astype(int)

        TP = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
        FN = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
        TN = np.sum((y_pred_bin == 0) & (y_true_bin == 0))

        acc_ovr = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
        mcc_ovr = matthews_corrcoef(y_true_bin, y_pred_bin)

        print(f"\nClass: {cls}")
        print(f"  Metrics:")
        print(f"    - Accuracy (OvR): {acc_ovr * 100:.2f}%")
        print(f"    - Matthews Correlation Coefficient (MCC): {mcc_ovr:.4f}")
        print(f"    - Precision: {macro_p * 100:.2f}%")
        print(f"    - Recall: {macro_r * 100:.2f}%")
        print(f"    - F1 Score: {macro_f1 * 100:.2f}%")
        print(f"    - Support: {np.sum(y_true == labels_range[i])}")
        print(f"  Confusion Components (OvR):")
        print(f"    - True Positives (TP): {TP}")
        print(f"    - False Positives (FP): {FP}")
        print(f"    - False Negatives (FN): {FN}")
        print(f"    - True Negatives (TN): {TN}")

    return {
        "accuracy": accuracy,
        "mcc": mcc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_range).tolist(),
        "per_class_f1": {
            cls: precision_recall_fscore_support(
                y_true == labels_range[i],
                y_pred == labels_range[i],
                average="binary", zero_division=0
            )[2]
            for i, cls in enumerate(class_emotions)
        }
    }