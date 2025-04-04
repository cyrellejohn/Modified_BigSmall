import math
import numpy as np
from signal_processing.ppg import filters

def POS_WANG(frames, fps, WinSec=1.6):
    l = math.ceil(WinSec * fps)
    N = frames.shape[0]
    H = np.zeros(N)

    # Calculate the average RGB values for each frame. Spatial Average
    C = np.mean(frames, axis=(1, 2)).T

    for n in range(N - l):
        # Extract the current window of frames
        current_window = C[:, n:(n + l)]

        # Normalize the current window of frames. Temporal Normalization
        Cn = current_window / np.mean(current_window, axis=1, keepdims=True)

        # Projection Matrix S
        projection_matrix = np.array([[0, 1, -1],
                                      [-2, 1, 1]])
        S = projection_matrix @ Cn

        # Signal Tuning
        h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]

        # Overlap-adding
        H[n:n + l] += h - np.mean(h)

    return H

def POS(frames, fps):
    PPG = POS_WANG(frames, fps)

    # Apply a bandpass filter to the PPG signal
    PPG = filters.butter_filter(PPG, low_cutoff=0.75, high_cutoff=3, 
                                sampling_rate=fps, order=1, filter_type='band')
    
    return PPG