import numpy as np
import pandas as pd
import torch
from evaluation.postProcess import *
from tqdm import tqdm
from evaluation.blandAltman import BlandAltman

def read_label(dataset):
    """
    Read manually corrected labels from a CSV file and return them as a dictionary.

    Parameters:
    dataset (str): The name of the dataset to read labels for. This is used to construct
                   the filename of the CSV file in the format '{dataset}_Comparison.csv'.

    Returns:
    dict: A dictionary where each key is a 'VideoID' (as a string) and each value is a 
          dictionary containing the corresponding row data from the CSV file.
    """
    # Read the CSV file into a Pandas DataFrame. The file is located in the 'label' directory
    # and is named using the format '{dataset}_Comparison.csv'.
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))

    # Convert the DataFrame to a dictionary with the index as keys and rows as dictionaries.
    out_dict = df.to_dict(orient='index')

    # Reformat the dictionary to use 'VideoID' as keys and the entire row data as values.
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}

    # Return the reformatted dictionary.
    return out_dict

def read_hr_label(feed_dict, index):
    """
    Extracts the heart rate (HR) label for a given index from the UBFC dataset.
    
    Parameters:
    - feed_dict (dict): A dictionary containing video data with heart rate information.
    - index (str): The key used to access specific video data in the dictionary. 
                   It may start with 'subject', which will be stripped if present.
    
    Returns:
    - tuple: A tuple containing the adjusted index and the heart rate value.
    """
    
    # Check if the index starts with 'subject' and remove it for accessing the dictionary
    if index[:7] == 'subject':
        index = index[7:]

    # Retrieve the dictionary for the specific video or subject
    video_dict = feed_dict[index]

    # Determine the preferred method for heart rate measurement and extract the HR value
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        # Default to 'Peak Detection' if no preferred method is specified
        hr = video_dict['Peak Detection']

    # Return the adjusted index and the selected heart rate value
    return index, hr

def _reform_data_from_dict(data, flatten=True):
    """
    Helper function to reformat predictions and labels from dictionaries into a consistent format for metric calculations.

    Parameters:
    - data (dict): A dictionary where keys are identifiers (e.g., video IDs) and values are tensors containing data (e.g., predictions or labels).
    - flatten (bool): A flag indicating whether the output should be a flattened 1D array. Defaults to True.

    Returns:
    - np.ndarray: A NumPy array containing the concatenated data, either flattened or in its original shape.
    """
    
    # Sort the dictionary items by key to ensure consistent processing order
    sort_data = sorted(data.items(), key=lambda x: x[0])

    # Extract the tensor values from the sorted items
    sort_data = [i[1] for i in sort_data]

    # Concatenate the list of tensors along the first dimension
    sort_data = torch.cat(sort_data, dim=0)

    # Convert the concatenated tensor to a NumPy array. If flatten is True, reshape the array to 1D
    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    # Return the processed data as a NumPy array
    return sort_data

def calculate_metrics(predictions, labels, config):
    """
    Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef., SNR, MACC).

    This function evaluates the performance of rPPG predictions against ground truth HR labels
    using specified metrics. It supports two evaluation methods: "Peak Detection" and "FFT".

    Parameters:
    - predictions: dict
        A dictionary containing predicted HR data for each video.
    - labels: dict
        A dictionary containing ground truth HR data for each video.
    - config: object
        Configuration object containing settings for evaluation, including:
        - INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW: bool
            Whether to use a smaller evaluation window.
        - INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE: int
            Size of the evaluation window.
        - TEST.DATA.FS: int
            Sampling frequency of the data.
        - TEST.DATA.PREPROCESS.LABEL_TYPE: str
            Type of label preprocessing ("Standardized", "Raw", "DiffNormalized").
        - INFERENCE.EVALUATION_METHOD: str
            Method of evaluation ("Peak Detection" or "FFT").
        - TOOLBOX_MODE: str
            Mode of the toolbox ("train_and_test" or "only_test").
        - TEST.METRICS: list
            List of metrics to calculate (e.g., "MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC").

    Returns:
    - None
        Prints the calculated metrics and generates Bland-Altman plots if specified.

    Raises:
    - ValueError
        If an unsupported label type, evaluation method, or toolbox mode is specified.

    Process:
    1. Initialize lists to store predictions, ground truth values, SNR, and MACC.
    2. Iterate over each prediction, reformatting data and determining window size.
    3. For each window, calculate metrics using the specified method.
    4. Generate a filename ID for saving results based on the toolbox mode.
    5. Convert lists to numpy arrays and calculate specified metrics, printing results.
    6. Generate Bland-Altman plots if specified in the metrics.
    """

    # Initialize lists for storing results
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("Calculating metrics!")

    # Iterate over predictions
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # Determine window size
        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        # Process each window
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i + window_frame_size]
            label_window = label[i:i + window_frame_size]
            
            # Check minimum window size
            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue
            
            # Determine label type
            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            # Calculate metrics based on evaluation method
            if config.INFERENCE.EVALUATION_METHOD == "Peak Detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(pred_window,
                                                                                 label_window, 
                                                                                 diff_flag=diff_flag_test, 
                                                                                 fs=config.TEST.DATA.FS, 
                                                                                 hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)

            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(pred_window, 
                                                                               label_window, 
                                                                               diff_flag=diff_flag_test, 
                                                                               fs=config.TEST.DATA.FS, 
                                                                               hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)

            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    # Generate filename ID for results
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    # Calculate and print metrics
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)

        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            
            elif "AU" in metric:
                pass
            
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            
            else:
                raise ValueError("Wrong Test Metric Type")
    
    elif config.INFERENCE.EVALUATION_METHOD == "Peak Detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))
            
            elif "AU" in metric:
                pass
            
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")