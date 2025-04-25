import math
import numpy as np
from signal_processing.ppg import filters

def POS_WANG(frames, fps, WinSec=1.6):
    l = int(np.ceil(WinSec * fps))
    N = frames.shape[0]
    H = np.zeros(N)

    # Spatial average of RGB values for each frame: shape (N, 3)
    C = np.mean(frames, axis=(1, 2))  # (N, 3)

    # Transpose to shape (3, N) for easier column slicing
    C = C.T if C.shape[1] == 3 else C.T[:3]

    # Projection matrix (precomputed)
    P = np.array([[0, 1, -1], [-2, 1, 1]])

    for n in range(N - l):
        window = C[:, n:n + l]  # shape (3, l)

        # Temporal normalization
        mean_per_channel = np.mean(window, axis=1, keepdims=True)
        Cn = window / (mean_per_channel + 1e-8)  # avoid division by zero

        # Apply projection
        S = P @ Cn  # shape (2, l)

        # Signal tuning
        std_ratio = np.std(S[0]) / (np.std(S[1]) + 1e-8)  # avoid division by zero
        h = S[0] + std_ratio * S[1]
        h -= np.mean(h)

        # Overlap-add
        H[n:n + l] += h

    return H

def POS(frames, fps):
    PPG = POS_WANG(frames, fps)

    # Apply a bandpass filter to the PPG signal
    PPG = filters.butter_filter(PPG, low_cutoff=0.75, high_cutoff=3, 
                                sampling_rate=fps, order=1, filter_type='band')
    
    return PPG