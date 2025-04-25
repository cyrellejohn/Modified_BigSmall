import numpy as np
from scipy.signal import butter, filtfilt

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