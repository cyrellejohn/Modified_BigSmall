import numpy as np
import scipy
import scipy.io
import os

from scipy.signal import butter
from sklearn.metrics import precision_recall_fscore_support
from evaluation.metrics import calculate_metrics, _reform_data_from_dict
from evaluation.postProcess import _detrend, _next_power_of_2, _calculate_SNR
from tqdm import tqdm
from evaluation.blandAltman import BlandAltman

def load_label_names(label_list_path):
    full_path = os.path.join(label_list_path, "label_list.txt")

    with open(full_path, "r") as file:
        label_list = [line.strip() for line in file]  # Read and strip spaces

    return label_list

# PPG Metrics
def calculate_ppg_metrics(predictions, labels, config):
    """
    Calculate PPG Metrics (MAE, RMSE, MAPE, Pearson Coefficient, SNR).

    This function prints a header indicating the start of PPG metrics calculation
    and then calls the `calculate_metrics` function to compute various metrics
    between the predicted and actual labels. The metrics calculated include:

    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error)
    - MAPE (Mean Absolute Percentage Error)
    - Pearson Coefficient
    - SNR (Signal-to-Noise Ratio)

    Parameters:
    - predictions: array-like
        The predicted values from the model.
    - labels: array-like
        The actual ground truth values.
    - config: dict or object
        Configuration settings that may influence the metric calculations.

    Returns:
    None
    """

    # Print a header to indicate the start of PPG metrics calculation
    print('=====================')
    print('==== PPG Metrics ====')
    print('=====================')

    # Call the calculate_metrics function to compute the metrics
    calculate_metrics(predictions, labels, config)

    # Print an empty line for formatting purposes
    print('')

# AU Metrics
def calculate_openface_au_metrics(preds, labels, config, named_AU):
    """
    Calculate AU Metrics (AU F1, Precision, Mean F1, Mean Acc, Mean Precision).

    This function computes various metrics for 12 Action Units (AUs) including:
    - Individual AU F1 scores
    - Individual AU Precision scores
    - Mean F1 score across all AUs
    - Mean Precision across all AUs
    - Mean Accuracy across all AUs
    
    Parameters:
    -----------
    preds : dict
        Dictionary containing predicted AU values for each trial
    labels : dict
        Dictionary containing ground truth AU values for each trial
    config : object
        Configuration object containing TEST.METRICS list
    
    Returns:
    --------
    None
        Prints metrics to console and stores results in metric_dict
    """
    named_AU_len = len(named_AU)

    # Reformat prediction and label data from dictionary format
    for index in preds.keys():
        preds[index] = _reform_data_from_dict(preds[index], flatten=False)
        labels[index] = _reform_data_from_dict(labels[index], flatten=False)

    # Initialize storage for metrics and concatenated data
    all_trial_preds = []
    all_trial_labels = []

    # Combine predictions and labels from all trials into single arrays
    for T in labels.keys():
        all_trial_preds.append(preds[T])
        all_trial_labels.append(labels[T])

    # Concatenate all trial data along first axis
    all_trial_preds = np.concatenate(all_trial_preds, axis=0)
    all_trial_labels = np.concatenate(all_trial_labels, axis=0)

    # Iterate through metrics specified in config
    for metric in config.TEST.METRICS:
        if metric == 'AU_METRICS':
            AU_data = dict()
            AU_data['labels'] = dict()
            AU_data['preds'] = dict()

            # Extract data for each AU from concatenated arrays
            for i in range(named_AU_len):
                AU_data['labels'][named_AU[i]] = all_trial_labels[:, i, 0]
                AU_data['preds'][named_AU[i]] = all_trial_preds[:, i]

            # Initialize storage for metrics and running averages
            metric_dict = dict()  
            avg_f1 = 0
            avg_prec = 0 
            avg_acc = 0   

            # Print section header for AU metrics
            print('')
            print('======================')
            print('===== AU METRICS =====')
            print('======================')
            print('AU / F1 / Precision')

            # Calculate metrics for each individual AU
            for au in named_AU:
                # Convert continuous predictions to binary using 0.5 threshold
                preds = np.array(AU_data['preds'][au])
                preds[preds < 0.5] = 0
                preds[preds >= 0.5] = 1
                labels = np.array(AU_data['labels'][au])

                # Calculate precision, recall, F1 score, and support using sklearn
                precision, recall, f1, support = precision_recall_fscore_support(labels, preds, beta=1.0)

                # Add debugging information
                print(f"\nDebug info for {au}:")
                print(f"F1 scores: {f1}")
                print(f"Precision scores: {precision}")
                print(f"Recall scores: {recall}")
                
                print(f"Unique values in predictions: {np.unique(preds)}")
                print(f"Unique values in labels: {np.unique(labels)}")
                print(f"Number of 0s in predictions: {np.sum(preds == 0)}")
                print(f"Number of 1s in predictions: {np.sum(preds == 1)}")
                print(f"Number of 0s in labels: {np.sum(labels == 0)}")
                print(f"Number of 1s in labels: {np.sum(labels == 1)}")

                '''
                # TODO: Alternate this code with the other code
                # TODO: Fix this error: https://tinyurl.com/2s4dmbfe and https://tinyurl.com/85s7rh3r
                # TODO: Possible solution on the error 1: Try Another Way of OpenFace prediction
                # TODO: Possible solution on the error 2: Try PyFeat AU prediction
                '''

                # Extract metrics for positive class (index 1)
                f1 = f1[1]
                precision = precision[1]
                recall = recall[1]

                '''
                # Alternative
                # Use conditional expressions to select the correct index
                f1 = f1[1] if len(f1) > 1 else f1[0]
                precision = precision[1] if len(precision) > 1 else precision[0]
                recall = recall[1] if len(recall) > 1 else recall[0]
                '''

                # Convert all metrics to percentage format
                f1 = f1 * 100
                precision = precision * 100
                recall = recall * 100

                # Calculate accuracy as percentage of correct predictions
                acc = sum(1 for x, y in zip(preds, labels) if x == y) / len(labels) * 100

                # Store metrics for current AU
                metric_dict[au] = (f1, precision, recall, acc)

                # Add current AU metrics to running averages
                avg_f1 += f1
                avg_prec += precision
                avg_acc += acc
                
                # Print results for current AU
                print(au, f1, precision)
                print('')

            # Calculate final averages across all AUs
            avg_f1 = avg_f1 / named_AU_len
            avg_acc = avg_acc / named_AU_len
            avg_prec = avg_prec / named_AU_len

            # Store aggregate metrics in dictionary
            metric_dict[f'{named_AU_len}AU_AvgF1'] = avg_f1
            metric_dict[f'{named_AU_len}AU_AvgPrec'] = avg_prec
            metric_dict[f'{named_AU_len}AU_AvgAcc'] = avg_acc

            # Print final aggregate results
            print('')
            print(f'Mean {named_AU_len} AU F1:', avg_f1)
            print(f'Mean {named_AU_len} AU Prec:', avg_prec)
            print(f'Mean {named_AU_len} AU Acc:', avg_acc)
            print('')

        else:
            print('{} metric is not for evaluating AUs'.format(metric))
            pass
            