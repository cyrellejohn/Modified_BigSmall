import numpy as np
import torch
from evaluation.postProcess import *
from tqdm import tqdm
from evaluation.blandAltman import BlandAltman
import os


def reform_data_to_numpy(data, flatten=True):
    """Converts dict of tensors or a tensor to a numpy array."""
    if isinstance(data, dict):
        if not data:
            return np.empty(0)
        try:
            concatenated = torch.cat([tensor for _, tensor in sorted(data.items())], dim=0)
        except Exception as e:
            print(f"[Error] Concatenation failed: {e}")
            return np.empty(0)
    elif isinstance(data, torch.Tensor):
        concatenated = data
    else:
        return np.empty(0)

    array = concatenated.cpu().numpy()
    return array.reshape(-1) if flatten else array

def compute_metrics(pred, gt, snr, macc, config, filename_id):
    """Compute and print metrics."""

    summary = {
        "MAE": None, "RMSE": None, "MAPE": None,
        "Pearson": None, "SNR": None, "MACC": None
    }

    n = len(pred)
    if n == 0:
        print("[Warning] No valid samples. Metrics skipped.")
        return

    for metric in config.TEST.METRICS:
        if metric == "MAE":
            err = np.abs(pred - gt)
            summary["MAE"] = (err.mean(), err.std(ddof=1) / np.sqrt(n))
            print(f"\nMAE : {err.mean():.4f} +/- {err.std(ddof=1) / np.sqrt(n):.4f}")

        elif metric == "RMSE":
            rmse = np.sqrt(np.mean(np.square(pred - gt)))
            se = np.std(np.square(pred - gt), ddof=1) / np.sqrt(n)
            summary["RMSE"] = (rmse, np.sqrt(se))
            print(f"RMSE : {rmse:.4f} +/- {np.sqrt(se):.4f}")

        elif metric == "MAPE":
            mask = gt != 0
            if mask.any():
                ape = np.abs((pred[mask] - gt[mask]) / gt[mask])
                summary["MAPE"] = (ape.mean() * 100, ape.std(ddof=1) / np.sqrt(mask.sum()) * 100)
                print(f"MAPE : {ape.mean() * 100:.4f} +/- {ape.std(ddof=1) / np.sqrt(mask.sum()) * 100:.4f}")
            else:
                print("MAPE : Skipped (all ground truth values are zero)")

        elif metric == "Pearson":
            corr = np.corrcoef(pred, gt)[0, 1]
            stderr = np.sqrt((1 - corr ** 2) / (n - 2)) if n > 2 else np.nan
            summary["Pearson"] = (corr, stderr)
            print(f"Pearson : {corr:.4f} +/- {stderr:.4f}")

        elif metric == "SNR":
            summary["SNR"] = (snr.mean(), snr.std(ddof=1) / np.sqrt(n))
            print(f"SNR (dB): {snr.mean():.4f} +/- {snr.std(ddof=1) / np.sqrt(n):.4f}")

        elif metric == "MACC":
            summary["MACC"] = (macc.mean(), macc.std(ddof=1) / np.sqrt(n))
            print(f"MACC: {macc.mean():.4f} +/- {macc.std(ddof=1) / np.sqrt(n):.4f}")

        elif metric == "BA":
            print("\n")
            plot_bland_altman(gt, pred, config, filename_id)

        elif metric == "AU_METRICS":
            continue  # Handled elsewhere

        else:
            raise ValueError(f"Unsupported metric type: {metric}")

    return {k: v for k, v in summary.items() if v is not None}

def plot_bland_altman(ground_truth, prediction, config, filename_id):
    """Generates Bland-Altman plots."""
    ba = BlandAltman(ground_truth, prediction, config, averaged=True)

    ba.scatter_plot(x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_BA_ScatterPlot',
                    file_name=f'{filename_id}_BA_ScatterPlot.pdf')

    ba.difference_plot(x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                       y_label='Average of rPPG HR and GT PPG HR [bpm]',
                       show_legend=True, figure_size=(5, 5),
                       the_title=f'{filename_id}_BA_DifferencePlot',
                       file_name=f'{filename_id}_BA_DifferencePlot.pdf')

def calculate_metrics(predictions, labels, config):
    """Calculate and print metrics for rPPG evaluation."""
    if not predictions or not labels:
        print("[Warning] Empty predictions or labels. Skipping.")
        return

    print("Calculating Metrics!")

    # Setup
    fps = config.TEST.DATA.FS
    label_type = config.TEST.DATA.PREPROCESS.LABEL_TYPE
    is_diff_norm = label_type == "DiffNormalized"
    method = config.INFERENCE.EVALUATION_METHOD
    use_small_win = config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW
    win_len_sec = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE

    pred_list, gt_list, snr_list, macc_list = [], [], [], []

    for subject_id in tqdm(predictions.keys(), ncols=100):
        pred = reform_data_to_numpy(predictions[subject_id])
        gt = reform_data_to_numpy(labels[subject_id])

        if pred.size == 0 or gt.size == 0:
            print(f"[Warning] Skipping {subject_id}: empty prediction or label.")
            continue

        video_len = len(pred)
        win_size = min(win_len_sec * fps, video_len) if use_small_win else video_len

        for i in range(0, video_len, win_size):
            pred_win = pred[i:i + win_size]
            gt_win = gt[i:i + win_size]

            if len(pred_win) < 9:
                continue

            gt_hr, pred_hr, snr, macc = calculate_metric_per_video(pred_win, gt_win,
                                                                   diff_normalized=is_diff_norm,
                                                                   apply_bandpass=True,
                                                                   fps=fps, hr_method=method)

            pred_list.append(pred_hr)
            gt_list.append(gt_hr)
            snr_list.append(snr)
            macc_list.append(macc)

    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_name = os.path.splitext(os.path.basename(config.INFERENCE.MODEL_PATH))[0]
        filename_id = f"{model_name}_{config.TEST.DATA.DATASET}"
    else:
        raise ValueError(f"Unsupported TOOLBOX_MODE: {config.TOOLBOX_MODE}")

    return compute_metrics(np.array(pred_list), np.array(gt_list), np.array(snr_list), 
                           np.array(macc_list), config, filename_id)