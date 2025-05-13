import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def detrend(signal, lambda_value):
    """
    Detrend a signal using Tarvainen's smoothness prior method.

    Args:
        signal (np.ndarray): 1D input signal
        lambda_value (float): smoothing parameter

    Returns:
        np.ndarray: detrended signal
    """
    signal = signal.reshape(-1, 1)
    N = len(signal)
    I = np.eye(N)
    D = np.diff(np.eye(N), n=2, axis=0)
    A = I + lambda_value**2 * (D.T @ D)

    detrended = np.linalg.solve(A, signal)

    return (signal - detrended).flatten()

def magnitude_to_decibels(mag):
    return 10 * np.log10(np.maximum(mag, 1e-12))  # Avoid log(0)

def apply_FFT(signal, fps):
    nfft = next_power_of_2(len(signal))
    return periodogram(signal, fs=fps, nfft=nfft, detrend=False)

def calculate_FFT_HR(signal, fps=60, low_cutoff=0.75, high_cutoff=2.5):
    freqs, power = apply_FFT(signal, fps)
    mask = (freqs >= low_cutoff) & (freqs <= high_cutoff)
    peak_freq = freqs[mask][np.argmax(power[mask])]
    return peak_freq * 60

def calculate_peak_HR(signal, fps):
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        return 0.0
    return 60.0 / (np.mean(np.diff(peaks)) / fps)

def compute_MACC(pred, ground_truth):
    pred, ground_truth = map(np.squeeze, (pred, ground_truth))
    min_len = min(len(pred), len(ground_truth))
    pred, ground_truth = pred[:min_len], ground_truth[:min_len]

    corrs = [abs(np.corrcoef(pred, np.roll(ground_truth, lag))[0, 1]) for lag in range(0, min_len)]
    return max(corrs) if corrs else 0.0

def calculate_SNR(signal, HR_ground_truth, fps=30, low_cutoff=0.75, high_cutoff=2.5):
    freqs, power = apply_FFT(signal, fps)
    power = np.squeeze(power)

    hr_freq = HR_ground_truth / 60.0
    harmonics = [(hr_freq, 6 / 60), (2 * hr_freq, 6 / 60)]

    harmonic_power = 0
    for freq, dev in harmonics:
        mask = (freqs >= freq - dev) & (freqs <= freq + dev)
        harmonic_power += np.sum(power[mask] ** 2)

    noise_mask = (freqs >= low_cutoff) & (freqs <= high_cutoff)
    for freq, dev in harmonics:
        noise_mask &= ~((freqs >= freq - dev) & (freqs <= freq + dev))

    noise_power = np.sum(power[noise_mask] ** 2)
    return magnitude_to_decibels(harmonic_power / noise_power) if noise_power > 0 else 0.0

def butter_filter(data, low_cutoff=None, high_cutoff=None, sampling_rate=30, order=4, filter_type='band'):
    """
    Apply a Butterworth filter (bandpass, lowpass, or highpass) to the input signal.

    Args:
        data (array-like): Input signal.
        low_cutoff (float, optional): Low cutoff frequency (Hz).
        high_cutoff (float, optional): High cutoff frequency (Hz).
        sampling_rate (int): Sampling rate of the signal (Hz).
        order (int): Order of the filter.
        filter_type (str): 'band', 'low', or 'high'.

    Returns:
        np.ndarray: Filtered signal.
    """
    data = np.asarray(data)
    nyquist = 0.5 * sampling_rate

    if filter_type == 'band':
        if low_cutoff is None or high_cutoff is None:
            raise ValueError("Both low_cutoff and high_cutoff must be provided for a bandpass filter.")
        cutoff = [low_cutoff / nyquist, high_cutoff / nyquist]
    elif filter_type == 'low':
        if high_cutoff is None:
            raise ValueError("high_cutoff must be provided for a lowpass filter.")
        cutoff = high_cutoff / nyquist
    elif filter_type == 'high':
        if low_cutoff is None:
            raise ValueError("low_cutoff must be provided for a highpass filter.")
        cutoff = low_cutoff / nyquist
    else:
        raise ValueError("Invalid filter_type. Use 'band', 'low', or 'high'.")

    b, a = butter(order, cutoff, btype=filter_type)
    return filtfilt(b, a, data)

def calculate_metric_per_video(pred, ground_truth, diff_normalized=True, apply_bandpass=True, fps=30, hr_method='FFT'):
    order = 3
    low_cutoff = 0.5
    high_cutoff = 4.0
    
    if diff_normalized:
        pred = np.cumsum(pred)
        ground_truth = np.cumsum(ground_truth)

    pred = detrend(pred, 100)
    ground_truth = detrend(ground_truth, 100)

    if apply_bandpass:
        pred = butter_filter(pred, low_cutoff, high_cutoff, fps, order)
        ground_truth = butter_filter(ground_truth, low_cutoff, high_cutoff, fps, order)

    MACC = compute_MACC(pred, ground_truth)

    if hr_method == 'FFT':
        HR_function = lambda data, fps: calculate_FFT_HR(data, fps, low_cutoff, high_cutoff)
    else:
        HR_function = calculate_peak_HR

    HR_pred = HR_function(pred, fps)
    HR_ground_truth = HR_function(ground_truth, fps)

    SNR = calculate_SNR(pred, HR_ground_truth, fps, low_cutoff, high_cutoff)

    return HR_ground_truth, HR_pred, SNR, MACC