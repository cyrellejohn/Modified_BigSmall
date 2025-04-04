import scipy.signal as signal

def butter_filter(data, low_cutoff=None, high_cutoff=None, sampling_rate=100, order=3, filter_type='band'):
    nyquist = 0.5 * sampling_rate

    if filter_type == 'band':
        if high_cutoff is None or low_cutoff is None:
            raise ValueError("Both low and high cutoff frequencies must be provided for bandpass filter.")
        
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
    elif filter_type == 'low':
        if high_cutoff is None:
            raise ValueError("High cutoff frequency must be provided for lowpass filter.")
        
        high = high_cutoff / nyquist
        b, a = signal.butter(order, high, btype='low')
    elif filter_type == 'high':
        if low_cutoff is None:
            raise ValueError("Low cutoff frequency must be provided for highpass filter.")
        
        low = low_cutoff / nyquist
        b, a = signal.butter(order, low, btype='high')
    else:
        raise ValueError("Unsupported filter type. Use 'band', 'low', or 'high'.")

    # Use filtfilt for zero-phase filtering
    new_data = signal.filtfilt(b, a, data)
    
    return new_data