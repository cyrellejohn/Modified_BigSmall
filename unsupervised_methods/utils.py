import numpy as np
from scipy import sparse

def detrend(input_signal, lambda_value):
    """
    Removes the trend from a given input signal using a smoothing technique.

    Parameters:
    input_signal (numpy.ndarray): The input signal to be detrended. It should be a 1D array.
    lambda_value (float): The smoothing parameter that controls the amount of smoothing. 
                          Higher values result in a smoother signal.

    Returns:
    numpy.ndarray: The detrended signal, with the trend removed.

    Description:
    This function applies a detrending operation to the input signal by constructing an 
    observation matrix and a difference matrix. It then uses these matrices to compute 
    a filtered signal that has the trend removed. The lambda_value parameter is used to 
    control the degree of smoothing applied to the signal. The method involves solving 
    a linear system to obtain the detrended signal.
    """
    # Determine the length of the input signal
    signal_length = input_signal.shape[0]
    
    # Create an identity matrix of size signal_length
    H = np.identity(signal_length)

    # Construct arrays for the difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)

    # Create the data and index arrays for the sparse difference matrix
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])

    # Construct the difference matrix D using sparse diagonals
    D = sparse.spdiags(diags_data, diags_index, signal_length - 2, signal_length).toarray()

    # Calculate H plus lambda squared times the dot product of D transpose and D
    H_plus_lambda_D = H + (lambda_value ** 2) * np.dot(D.T, D)
    
    # Compute the inverse of the above result
    H_plus_lambda_D_inv = np.linalg.inv(H_plus_lambda_D)
    
    # Subtract the inverse from H to get the intermediate result
    temp_matrix = H - H_plus_lambda_D_inv
    
    # Calculate the filtered signal by taking the dot product of the intermediate result and the input signal
    filtered_signal = np.dot(temp_matrix, input_signal)

    # Return the detrended signal
    return filtered_signal

def process_video(frames):
    """
    Processes a sequence of video frames to extract average RGB values.

    Parameters:
    frames (list or numpy.ndarray): A list or array of video frames, where each frame is 
                                    expected to be a 3D numpy array with dimensions 
                                    (height, width, channels).

    Returns:
    numpy.ndarray: A 3D numpy array of shape (1, 3, number_of_frames) containing the 
                   average RGB values for each frame. The first dimension is a singleton 
                   dimension, the second dimension corresponds to the RGB channels, and 
                   the third dimension corresponds to the frame index.

    Description:
    This function iterates over each frame in the input sequence, calculates the average 
    RGB values by summing over the height and width dimensions, and normalizes by the 
    total number of pixels in the frame. The resulting RGB values for each frame are 
    stored in a list, which is then converted to a numpy array. The array is transposed 
    and reshaped to have a shape of (1, 3, number_of_frames) before being returned.
    """
    # Initialize an empty list to store average RGB values for each frame
    RGB = []

    # Iterate over each frame in the input list of frames
    for frame in frames:
        # Compute the sum of pixel values along the height and width of the frame
        summation = np.sum(np.sum(frame, axis=0), axis=0)

        # Calculate the average RGB values by dividing the sum by the total number of pixels
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))

    # Convert the list of average RGB values to a numpy array
    RGB = np.asarray(RGB)

    # Transpose the array to switch axes and reshape it to (1, 3, -1)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    
    # Return the processed RGB values as a NumPy array
    return np.asarray(RGB)