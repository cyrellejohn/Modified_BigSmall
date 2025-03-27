import math
import numpy as np
from scipy import signal
from unsupervised_methods import utils


def process_video(frames):
    """
    Processes a sequence of video frames to extract the average RGB values.

    Parameters:
    frames (list or ndarray): A list or array of video frames, where each frame is a 3D array 
                              representing the pixel values in RGB format.

    Returns:
    ndarray: A 2D array where each row corresponds to the average RGB values of a frame.
             The shape of the array is (number_of_frames, 3).
    
    The function calculates the average RGB values for each frame by summing the pixel values 
    across the width and height of the frame and then dividing by the total number of pixels 
    in the frame. This results in a single RGB value for each frame, which is appended to the 
    RGB list and returned as a numpy array.
    """
    # Initialize an empty list to store average RGB values for each frame.
    RGB = []

    for frame in frames:
        # Calculate the sum of pixel values across height and width for each color channel.
        summation = np.sum(np.sum(frame, axis=0), axis=0)

        # Calculate the average RGB values by dividing the summation by the total number of pixels.
        average_rgb = summation / (frame.shape[0] * frame.shape[1])

        # Append the average RGB values to the RGB list.
        RGB.append(average_rgb)

    # Convert the list of average RGB values to a NumPy array and return it.
    return np.asarray(RGB)


def POS(frames, fps, order=1, low_cutoff=0.75, high_cutoff=3):
    """
    Extracts the photoplethysmography (PPG) signal from a sequence of video frames using the method
    described by Wang et al. (2017).

    Parameters:
    frames (list or ndarray): A sequence of video frames, where each frame is a 3D array representing
                              the pixel values in RGB format.
    fps (float): The sampling frequency of the video frames.

    Returns:
    ndarray: The estimated blood volume pulse (PPG) signal extracted from the video frames.
    """
    # Define the window size in seconds.
    WinSec = 1.6

    # Process the video frames to compute average RGB values for each frame.
    RGB = process_video(frames)

    # Number of frames.
    N = RGB.shape[0]

    # Initialize an array to store the processed signal.
    H = np.zeros((1, N))

    # Calculate the number of frames corresponding to the window size.
    l = math.ceil(WinSec * fps)

    # Iterate over each frame to process the signal.
    for n in range(N):
        m = n - l
        if m >= 0:
            # Normalize the RGB values by dividing by their mean.
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H

            # Transform the normalized values using matrix multiplication.
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

            # Combine components of S to form the signal h.
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]

            # Subtract the mean from h to adjust the signal.
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h

            # Update the array H with the processed signal h.
            H[0, m:n] = H[0, m:n] + (h[0])

    # Initialize the PPG signal with the processed signal H.
    PPG = H

    # Detrend the PPG signal using a utility function.
    PPG = utils.detrend(np.mat(PPG).H, 100)

    # Convert the detrended signal to a 1D array.
    PPG = np.asarray(np.transpose(PPG))[0]

    # Design a bandpass filter with specified frequency range.
    b, a = signal.butter(order, [low_cutoff / fps * 2, high_cutoff / fps * 2], btype='bandpass')

    # Apply the bandpass filter to the PPG signal.
    PPG = signal.filtfilt(b, a, PPG.astype(np.double))

    # Return the filtered PPG signal.
    return PPG