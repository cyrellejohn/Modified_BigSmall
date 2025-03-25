"""
The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """
    Calculate the smallest power of 2 that is greater than or equal to the given integer x.

    Args:
        x (int): The input integer for which the nearest power of 2 is to be calculated.

    Returns:
        int: The smallest power of 2 that is greater than or equal to x.

    Example:
        _next_power_of_2(5)  # Returns 8
        _next_power_of_2(8)  # Returns 8

    Note:
        If x is 0, the function returns 1, as 2^0 is the smallest power of 2.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """
    Detrend PPG signal by removing low-frequency trends using a regularized least squares approach.

    Args:
        input_signal (np.array): The input PPG signal to be detrended.
        lambda_value (float): Regularization parameter controlling the smoothness of the detrending.

    Returns:
        np.array: The detrended PPG signal.
    """
    # Determine the length of the input signal
    signal_length = input_signal.shape[0]
    
    # Create an identity matrix of size equal to the signal length
    H = np.identity(signal_length)

    # Construct arrays for the difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)

    # Create a sparse matrix D to approximate the second derivative
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()

    # Perform the detrending operation using regularized least squares
    detrended_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    
    # Return the detrended signal
    return detrended_signal

def power2db(mag):
    """
    Convert a power magnitude to decibels (dB).

    This function takes a power magnitude and converts it to decibels using the formula:
    dB = 10 * log10(mag)

    Args:
        mag (float or np.array): The power magnitude to be converted. This can be a single float value or a numpy array of values.

    Returns:
        float or np.array: The decibel equivalent of the input power magnitude. The return type matches the input type.
    """
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """
    Calculate heart rate based on PPG using Fast Fourier Transform (FFT).

    Parameters:
    - ppg_signal (np.array): The input PPG signal, a time-series data representing blood volume changes.
    - fs (int, optional): Sampling frequency of the PPG signal. Default is 60 Hz.
    - low_pass (float, optional): Lower bound of the frequency range in Hz for filtering the heart rate. Default is 0.75 Hz.
    - high_pass (float, optional): Upper bound of the frequency range in Hz for filtering the heart rate. Default is 2.5 Hz.

    Returns:
    - fft_hr (float): Calculated heart rate in beats per minute.
    """
    # Expand dimensions of the PPG signal to ensure it is a 2D array
    ppg_signal = np.expand_dims(ppg_signal, 0)

    # Calculate the next power of 2 for the length of the signal for efficient FFT computation
    N = _next_power_of_2(ppg_signal.shape[1])

    # Compute the periodogram to estimate the power spectral density of the signal
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)

    # Create a mask to filter frequencies within the specified range (0.75 Hz to 2.5 Hz)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))

    # Apply the mask to extract the relevant frequency and power spectral density values
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)

    # Find the frequency with the maximum power within the specified range
    # Then convert the frequency from Hz to beats per minute by multiplying by 60
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    # Return the calculated heart rate in beats per minute
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """
    Calculate heart rate from a PPG signal using peak detection.

    Args:
        ppg_signal (np.array): The input PPG signal, a 1D array representing the recorded signal over time.
        fs (int or float): The sampling frequency of the PPG signal, indicating the number of samples per second.

    Returns:
        float: The estimated heart rate in beats per minute (BPM).
    """
    # Detect peaks in the PPG signal, which correspond to heartbeats
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)

    # Calculate the average number of samples between consecutive peaks
    # Convert this to time in seconds by dividing by the sampling frequency
    # Convert the heart rate from beats per second to beats per minute by multiplying by 60
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)

    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """
    Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
    
    Args:
        pred_ppg_signal(np.array): predicted PPG signal 
        label_ppg_signal(np.array): ground truth, label PPG signal
        
    Returns:
        MACC(float): Maximum Amplitude of Cross-Correlation
    """
    # Create deep copies of the input signals to avoid modifying the originals
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)

    # Remove single-dimensional entries from the shape of the signals
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    # Truncate both signals to the minimum length to ensure equal size
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # Create an array of lag values from 0 to the length of the signal minus one
    lags = np.arange(0, len(pred)-1, 1)

    # Initialize a list to store cross-correlation values
    tlcc_list = []

    # Iterate over each lag value
    for lag in lags:
        # Compute the absolute cross-correlation between the predicted signal
        # and a lagged version of the ground truth signal
        cross_corr = np.abs(np.corrcoef(pred, np.roll(gt, lag))[0][1])

        # Append the cross-correlation value to the list
        tlcc_list.append(cross_corr)

    # Find the maximum value in the list of cross-correlation values
    macc = max(tlcc_list)

    # Return the maximum amplitude of cross-correlation
    return macc

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a predicted PPG signal.

    The SNR is computed as the ratio of the power around the first and second harmonics
    of the ground truth heart rate frequency to the power of the remainder of the frequency
    spectrum within a specified range.

    Args:
        pred_ppg_signal (np.array): The predicted PPG signal.
        hr_label (float): The ground truth heart rate in beats per minute.
        fs (int or float, optional): The sampling rate of the signal. Defaults to 30 Hz.
        low_pass (float, optional): The lower bound of the frequency range to consider. Defaults to 0.75 Hz.
        high_pass (float, optional): The upper bound of the frequency range to consider. Defaults to 2.5 Hz.

    Returns:
        float: The calculated SNR in decibels (dB).
    """
    # Convert heart rate to first and second harmonic frequencies in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz

    # Calculate the FFT of the predicted PPG signal
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Identify indices for the first and second harmonics and the remainder of the spectrum
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Extract power values for harmonics and remainder
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate signal power for harmonics and remainder
    signal_power_hm1 = np.sum(pxx_harmonic1**2)
    signal_power_hm2 = np.sum(pxx_harmonic2**2)
    signal_power_rem = np.sum(pxx_remainder**2)

    # Calculate SNR as the ratio of harmonic power to remainder power
    if not signal_power_rem == 0: # Avoid division by zero
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0

    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """
    Calculate video-level Heart Rate (HR) and Signal-to-Noise Ratio (SNR) from PPG signals.

    Args:
        predictions (np.array): The predicted PPG signal.
        labels (np.array): The ground truth PPG signal.
        fs (int, optional): Sampling frequency of the signals. Default is 30 Hz.
        diff_flag (bool, optional): If True, assumes the input signals are the first derivative of the PPG signal.
        use_bandpass (bool, optional): If True, applies a bandpass filter to the signals.
        hr_method (str, optional): Method to calculate heart rate, either 'FFT' or 'Peak'. Default is 'FFT'.

    Returns:
        tuple: A tuple containing:
            - hr_label (float): Ground truth heart rate.
            - hr_pred (float): Predicted heart rate.
            - SNR (float): Signal-to-Noise Ratio of the predicted signal.
            - macc (float): Maximum Amplitude of Cross-Correlation between predicted and ground truth signals.
    """

    # Detrend the signals. If diff_flag is True, integrate the signals before detrending.
    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # Apply a bandpass filter to the signals if use_bandpass is True.
    if use_bandpass:
        # Bandpass filter between [0.75, 2.5] Hz, equivalent to [45, 150] beats per minute.
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    # Calculate the Maximum Amplitude of Cross-Correlation (MACC).
    macc = _compute_macc(predictions, labels)

    # Calculate heart rate using the specified method (FFT or Peak).
    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')

    # Calculate the Signal-to-Noise Ratio (SNR) of the predicted signal.
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)

    # Return the calculated metrics.
    return hr_label, hr_pred, SNR, macc