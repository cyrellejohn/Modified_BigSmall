"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC
"""
import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
import math
from multiprocessing import Pool, Process, Value, Array, Manager
from unsupervised_methods.algorithm import POS
import time

'''
Unsupervised methods will not work on this project.

from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
'''

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
# from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface | pip install retinaface

class BaseLoader(Dataset):
    """
    The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument("--cached_path", default=None, type=str)
        parser.add_argument("--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """
        Initializes the BaseLoader instance.

        Args:
            dataset_name (str): The name of the dataset.
            raw_data_path (str): The path to the raw data.
            config_data (CfgNode): Configuration data containing various settings.

        Attributes:
            inputs (list): List to store input data paths.
            labels (list): List to store label data paths.
            dataset_name (str): Name of the dataset.
            raw_data_path (str): Path to the raw data.
            cached_path (str): Path to the cached preprocessed data.
            file_list_path (str): Path to the file list.
            preprocessed_data_len (int): Length of the preprocessed dataset.
            data_format (str): Format of the data (e.g., 'NDCHW').
            do_preprocess (bool): Flag indicating whether to preprocess data.
            config_data (CfgNode): Configuration data.

        Raises:
            ValueError: If the cached path or file list path does not exist and preprocessing is not enabled.
        """

        # Initialize lists to store input data paths and labels
        self.inputs = list()
        self.labels = list()

        # Store dataset name and paths from arguments
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH

        # Validate configuration settings for data splitting
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data

        # Validate configuration settings
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)

        # If preprocessing is required
        if config_data.DO_PREPROCESS:
            # Retrieve raw data directories
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)

            # Preprocess the dataset
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            # Check if cached data path exists
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name, 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            
            # Check if file list path exists
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')

                # Retrieve raw data directories
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)

                # Generate file list retroactively
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            # Load preprocessed data
            self.load_preprocessed_data_lightning()

        # Print paths and dataset length for logging
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """
        Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        
        # Split the raw data directories based on the specified begin and end indices
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)

        # Process the split data directories using multi-process management. Likely involves parallel processing to speed up preprocessing
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess) 

        # Build a list of files from the processed data. This list is used to keep track of the preprocessed data files
        self.build_file_list(file_list_dict)

        # Load the preprocessed data and their corresponding labels to ensure consistency
        self.load_preprocessed_data()
        
        # Log the total number of raw files that were preprocessed
        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=10):
        """
        Allocate dataset preprocessing across multiple processes efficiently.

        Args:
            data_dirs (List[str]): A list of video file paths to be processed.
            config_preprocess (Dict): A dictionary containing preprocessing configurations.
            multi_process_quota (int): Maximum number of subprocesses to spawn for multiprocessing.

        Returns:
            file_list_dict (Dict): A dictionary containing information regarding processed data (path names).
        """
        print('Preprocessing dataset...')

        file_num = len(data_dirs)
        manager = Manager()
        file_list_dict = manager.dict()
        
        processes = []
        pbar = tqdm(total=file_num, desc="Preprocessing files")

        i = 0
        while i < file_num or processes:
            # Clean up finished processes
            for p in processes[:]:  # Iterate over a copy
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
                    pbar.update(1)

            # Start new processes if below the quota
            while len(processes) < multi_process_quota and i < file_num:
                p = Process(target=self.preprocess_dataset_subprocess, 
                            args=(data_dirs, config_preprocess, i, file_list_dict))
                p.start()
                processes.append(p)
                i += 1

            # Sleep briefly to prevent busy waiting
            time.sleep(30.0)

        pbar.close()
        return file_list_dict

    def preprocess(self, frames, bvps, config_preprocess):
        """
        Preprocesses video frames and BVP signals according to specified configurations.

        Args:
            frames (np.array): Video frames to be processed.
            bvps (np.array): Blood volume pulse signals corresponding to the frames.
            config_preprocess (CfgNode): Configuration settings for preprocessing.

        Returns:
            frames_clips (np.array): Processed and optionally chunked video frames.
            bvps_clips (np.array): Processed and optionally chunked BVP signals.
        """

        # Step 1: Resize frames and crop for face region based on configuration
        frames = self.crop_face_resize(frames,
                                       config_preprocess.CROP_FACE.DO_CROP_FACE,
                                       config_preprocess.CROP_FACE.BACKEND,
                                       config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                       config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                       config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                       config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                       config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                       config_preprocess.RESIZE.W,
                                       config_preprocess.RESIZE.H)
        
        # Step 2: Transform video data based on specified data types
        data = list()  # Initialize list to store transformed video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy() # Copy frames to avoid modifying original data

            if data_type == "Raw":
                data.append(f_c) # Append raw frames
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c)) # Apply standardization
            else:
                raise ValueError("Unsupported data type!") # Raise error for unsupported data types
            
        data = np.concatenate(data, axis=-1)  # Concatenate transformed data along the last axis
        
        # Step 3: Transform BVP signals based on specified label types
        if config_preprocess.LABEL_TYPE == "Raw":
            pass # Use raw BVP signals
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps) # Apply difference normalization
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps) # Apply standardization
        else:
            raise ValueError("Unsupported label type!") # Raise error for unsupported types

        # Step 4: Optionally chunk data into smaller segments
        if config_preprocess.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH) # Chunk data and BVP signals
        else:
            frames_clips = np.array([data]) # Wrap data in a single-element array
            bvps_clips = np.array([bvps]) # Wrap BVP signals in a single-element array

        # Return processed video frames and BVP signals
        return frames_clips, bvps_clips

    def crop_face_resize(self, frames, use_face_detection, backend, use_larger_box, larger_box_coef,
                     use_dynamic_detection, detection_freq, use_median_box, width, height):
        """
        Crop and resize face regions in video frames.
        """
        num_frames = frames.shape[0]
        frame_h, frame_w = frames.shape[1:3]
        resized_frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

        # Determine how many times to detect faces
        num_detections = ceil(num_frames / detection_freq) if use_dynamic_detection else 1

        # Detect faces on sampled frames
        face_region_all = []
        for idx in range(num_detections):
            frame_idx = idx * detection_freq
            if use_face_detection:
                face_region = self.face_detection(frames[frame_idx], backend, use_larger_box, larger_box_coef)
            else:
                face_region = [0, 0, frame_w, frame_h]
            face_region_all.append(face_region)

        face_region_all = np.asarray(face_region_all, dtype=int)

        # Compute median box if needed
        face_region_median = np.median(face_region_all, axis=0).astype(int) if use_median_box else None

        # Precompute for loop
        get_face_region = (
            lambda i: face_region_median if use_median_box else face_region_all[i // detection_freq]
            if use_dynamic_detection else face_region_all[0]
        )

        for i, frame in enumerate(frames):
            if use_face_detection:
                x, y, w, h = get_face_region(i)
                top = max(y, 0)
                bottom = min(y + h, frame_h)
                left = max(x, 0)
                right = min(x + w, frame_w)
                frame = frame[top:bottom, left:right]

            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        return resized_frames

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """
        Detects a face in a given video frame using specified backend and optionally enlarges the detected face box.

        Args:
            frame (np.array): The video frame in which to detect a face.
            backend (str): The face detection backend to use. Options are:
                - "HC": OpenCV's Haar Cascade.
                - "RF": TensorFlow-based RetinaFace.
            use_larger_box (bool, optional): Whether to enlarge the detected face box. Defaults to False.
            larger_box_coef (float, optional): The coefficient by which to enlarge the face box. Defaults to 1.0.

        Returns:
            list: Coordinates of the detected face box in the format [x_coord, y_coord, width, height].

        Raises:
            ValueError: If an unsupported backend is specified.

        Notes:
            - For "HC" backend, the method uses a pre-trained Haar Cascade model to detect faces.
            - For "RF" backend, the method uses RetinaFace to detect faces, which can utilize both CPU and GPU.
            - If no face is detected, the entire frame is returned as the face box.
            - If multiple faces are detected with "HC", the largest face is selected.
            - The face box can be enlarged by setting `use_larger_box` to True and specifying `larger_box_coef`.
        """
        if backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml') # Alternative implementation: import cv2 then print(cv2.data.haarcascades)

            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone to define using
            # the computed width and height.
            face_zone = detector.detectMultiScale(frame)

            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                # If no face is detected, set the face box coordinates to cover the entire frame.
                face_box_coor = [0, 0, frame.shape[1], frame.shape[0]]

            elif len(face_zone) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one faces are detected. Only cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        
        ''' RETINA FACE | NOT FULLY IMPLEMENTED | NOT FINAL
        elif backend == "RF":
            # Use a TensorFlow-based RetinaFace implementation for face detection
            # This utilizes both the CPU and GPU
            res = RetinaFace.detect_faces(frame)

            if len(res) > 0:
                # Pick the highest score
                highest_score_face = max(res.values(), key=lambda x: x['score'])
                face_zone = highest_score_face['facial_area']

                # This implementation of RetinaFace returns a face_zone in the
                # form [x_min, y_min, x_max, y_max] that corresponds to the 
                # corners of a face zone
                x_min, y_min, x_max, y_max = face_zone

                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)
                
                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]
            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        
        else:
            raise ValueError("Unsupported face detection backend!")
        '''
        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        
        return face_box_coor

    @staticmethod
    def diff_normalize_data(data):
        """
        Compute difference-normalized video data along time axis, then normalize.
        
        Args:
            data (np.ndarray): Input video array of shape (n, h, w, c).
        
        Returns:
            np.ndarray: Difference-normalized and standardized video data (float32).
        """
        # Ensure input is float32
        data = data.astype(np.float32)

        # Compute difference and summation between consecutive frames
        diff = data[1:] - data[:-1]
        summation = data[1:] + data[:-1] + 1e-7  # prevent division by zero

        # Normalize the differences
        diff_normalized = np.divide(diff, summation, where=summation != 0)

        # Normalize by standard deviation
        std = np.std(diff_normalized, dtype=np.float32)
        if std != 0:
            diff_normalized = diff_normalized / std

        # Replace any NaNs with zeros
        np.nan_to_num(diff_normalized, copy=False)

        # Pad to match original number of frames, using the same dtype
        padding = np.zeros((1, *data.shape[1:]), dtype=diff_normalized.dtype)
        result = np.concatenate((diff_normalized, padding), axis=0)

        return result.astype(np.float32)  # Final dtype guaranteed

    @staticmethod
    def standardized_data(data):
        """
        Z-score standardization for video data.

        Args:
            data (np.ndarray): The input video data to be standardized.

        Returns:
            np.ndarray: The standardized video data.
        """
        mean = np.mean(data)
        std = np.std(data)

        # Avoid division by zero by checking if std is non-zero
        if std == 0:
            return np.zeros_like(data, dtype=np.float32)

        # Perform standardization and handle NaNs (if any)
        standardized = (data - mean) / std
        np.nan_to_num(standardized, copy=False)  # replaces NaNs with 0 in-place

        return standardized.astype(np.float32)

    @staticmethod
    def diff_normalize_label(label):
        """
        Calculate the discrete difference in labels along the time-axis and normalize by its standard deviation.
        Returns an array of the same length as the input.
        """
        # Compute discrete difference
        diff_label = np.diff(label, axis=0)

        # Compute standard deviation once
        std = np.std(diff_label)
        if std == 0 or np.isnan(std):
            normalized = np.zeros_like(label[:-1])
        else:
            normalized = diff_label / std

        # Pad with zero to match original length
        result = np.concatenate([normalized, [0]])

        return result

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        mean = np.mean(label)
        std = np.std(label)

        if std == 0 or np.isnan(std):
            return np.zeros_like(label)

        return (label - mean) / std

    def chunk(self, frames, bvps, chunk_length):
        """
        Splits the input video frames and corresponding BVP signals into fixed-length chunks.

        Args:
            frames (np.array): The array of video frames to be chunked.
            bvps (np.array): The array of blood volume pulse (BVP) signals to be chunked.
            chunk_length (int): The length of each chunk.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - frames_clips (np.array): Chunks of video frames.
                - bvps_clips (np.array): Chunks of BVP signals.
        """
        # Calculate the number of complete chunks that can be created
        clip_num = frames.shape[0] // chunk_length

        # Create chunks of frames and BVP signals
        frames_clips, bvps_clips = map(np.array, zip(*[
            (frames[i * chunk_length:(i + 1) * chunk_length], 
            bvps[i * chunk_length:(i + 1) * chunk_length]) 
            for i in range(clip_num)
        ]))

        ''' 
        ADD THIS IN THE FUTURE IF WE WANT TO HANDLE THE REMAINING FRAMES
        # Handle any remaining frames that don't fit into a complete chunk
        if frames.shape[0] % chunk_length != 0:
            start_index = clip_num * chunk_length
            frames_clips.append(frames[start_index:])
        
        ADD THIS IN THE FUTURE IF WE WANT TO HANDLE THE REMAINING FRAMES
        if bvps.shape[0] % chunk_length != 0:
            start_index = clip_num * chunk_length
            bvps_clips.append(bvps[start_index:])
        '''
        
        # Return the chunks as numpy arrays
        return frames_clips, bvps_clips

    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """
        Saves preprocessed video frames and corresponding BVP signal labels to disk in a multi-process environment.

        Args:
            frames_clips (np.array): Array of video frame clips to be saved.
            bvps_clips (np.array): Array of BVP signal labels corresponding to each video frame clip.
            filename (str): Base filename to use for saving the clips.

        Returns:
            tuple: Two lists containing the paths of the saved input and label files, respectively.
        """

        # Ensure the cached path directory exists, create it if it doesn't
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        # Initialize counter and lists to store file paths
        count = 0
        input_path_name_list = []
        label_path_name_list = []

        # Iterate over each BVP clip to save corresponding video frames and labels
        for i in range(len(bvps_clips)):
            # Ensure the number of inputs matches the number of labels
            assert (len(self.inputs) == len(self.labels))

            # Construct unique filenames for input and label files
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))

            # Append the constructed file paths to the respective lists
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

            # Save the video frames and BVP labels to .npy files
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])

            # Increment the counter for unique filenames
            count += 1
        
        # Return the lists of saved file paths
        return input_path_name_list, label_path_name_list

    def build_file_list(self, file_list_dict):
        """
        Build a list of files used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict (dict): A dictionary where keys are process numbers and values are lists of file paths.

        Raises:
            ValueError: If no files are found in the file list.
        """
        
        # Initialize an empty list to store file paths
        file_list = []
        
        # Iterate through the dictionary, collecting file paths from each process
        for process_num, file_paths in file_list_dict.items():
            # Append the list of file paths from each process to the main file list
            file_list = file_list + file_paths

        # Check if the file list is empty, raise an error if no files were added
        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        # Create a Pandas DataFrame from the file list with a single column 'input_files'
        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        
        # Ensure the directory for saving the file list exists, create it if necessary
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        
        # Save the DataFrame to a CSV file at the specified file list path
        file_list_df.to_csv(self.file_list_path)

    def load_preprocessed_data(self):
        """Loads preprocessed data file paths and their corresponding labels from a CSV file.

        This method reads a CSV file containing a list of preprocessed data files, extracts the file paths,
        generates corresponding label file paths, and stores them in the class attributes for later use.
        It also checks for errors in loading the data and calculates the length of the preprocessed dataset.
        """

        # Retrieve the path to the file list CSV
        file_list_path = self.file_list_path

        # Read the CSV file into a Pandas DataFrame
        file_list_df = pd.read_csv(file_list_path)

        # Extract the 'input_files' column as a list of input file paths
        inputs = file_list_df['input_files'].tolist()

        # Check if the list of inputs is empty and raise an error if so
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')

        # Sort the list of input file paths
        inputs = sorted(inputs)

        # Generate a list of label file paths by replacing "input" with "label" in each input file path
        labels = [input_file.replace("input", "label") for input_file in inputs]

        # Store the sorted input file paths in the class attribute
        self.inputs = inputs

        # Store the label file paths in the class attribute
        self.labels = labels

        # Calculate and store the length of the preprocessed dataset
        self.preprocessed_data_len = len(inputs)

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """
        Retroactively builds a list of preprocessed data files for a specified subset of raw data directories.
        The list is saved to a CSV file for further processing or training.

        Args:
            data_dirs (List[str]): A list of raw data directories.
            begin (float): The starting index for the data split.
            end (float): The ending index for the data split.

        Raises:
            ValueError: If no preprocessed files are found in the specified directory.

        Returns:
            None
        
        This method helps ensure that the data loader has a list of preprocessed files to work with, 
        which is essential for loading data efficiently during training or evaluation.
        """

        # Get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # Generate a list of unique raw-data file names from the subset
        filename_list = []
        for i in range(len(data_dirs_subset)):
            # Extract the 'index' key from each directory dictionary.
            filename_list.append(data_dirs_subset[i]['index'])

        # Ensure all indexes are unique.
        filename_list = list(set(filename_list))

        # Generate a list of all preprocessed / chunked data files.
        file_list = []
        for fname in filename_list:
            # Find all preprocessed data files matching the pattern in the cached path.
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            
            # Add the found files to the file list.
            file_list += processed_file_data

        # Check if the file list is empty and raise an error if so.
        if not file_list:
            raise ValueError(self.dataset_name, 'File list empty. Check preprocessed data folder exists and is not empty.')
        
        # Convert the file list to a Pandas DataFrame.
        file_list_df = pd.DataFrame(file_list, columns=['input_files'])

        # Ensure the directory for the file list path exists, creating it if necessary.
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)

        # Save the DataFrame to a CSV file at the specified file list path.
        file_list_df.to_csv(self.file_list_path)

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Retrieves a data sample and its corresponding label from the dataset.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - data (np.ndarray): The video data, transposed to the appropriate format.
                - label (np.ndarray): The corresponding label for the video data.
                - filename (str): The base filename of the data sample, indicating its source.
                - chunk_id (str): The identifier for the specific chunk of data.

        Raises:
            ValueError: If the data format specified in `self.data_format` is unsupported.
        """
        # Load the data and label from the preprocessed files using the index
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])

        # Transpose the data array based on the specified data format
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        # Convert data and label to float32 for compatibility with PyTorch
        data = np.float32(data)
        label = np.float32(label)
        
        # START: GETTING THE BASE FILE NAME
        # Extract the file path of the current data item
        item_path = self.inputs[index]

        # Extract the filename from the file path
        item_path_filename = item_path.split(os.sep)[-1]

        # Find the index to split the filename to get the base filename and chunk ID
        split_idx = item_path_filename.rindex('_')

        # Extract the base filename (e.g., '501' from '501_input0.npy')
        filename = item_path_filename[:split_idx]
        # END: GETTING THE BASE FILE NAME

        # Extract the chunk ID (e.g., '0' from '501_input0.npy')
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

        # Return the data, label, base filename, and chunk ID
        return data, label, filename, chunk_id
      
    def read_npy_video(self, video_file):
        """
        Reads a video file in the numpy format (.npy) and returns its frames as a NumPy array in RGB format.

        Args:
            video_file (list): A list containing the path to the .npy video file.

        Returns:
            np.ndarray: A NumPy array of shape (T, H, W, 3) containing the video frames,
                        where T is the number of frames, H is the height, W is the width,
                        and 3 represents the RGB color channels.
        """
        
        # Load the video data from the .npy file
        frames = np.load(video_file[0])
       
        # Check if the frames are of integer type and within the 0-255 range
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            # Convert frames to uint8 and keep only the first three channels (RGB)
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        
        # Check if the frames are of floating-point type and within the 0.0-1.0 range
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            # Scale frames to 0-255, convert to uint8, and keep only the first three channels (RGB)
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        
        # Raise an exception if the frames do not meet the expected type or range
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '\
            + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        
        # Return the processed frames as a NumPy array
        return np.asarray(processed_frames)

    def save(self, frames_clips, bvps_clips, filename):
        """
        Saves processed video frames and corresponding BVP signal labels to disk.

        Args:
            frames_clips (list of np.array): List of video frame chunks to be saved.
            bvps_clips (list of np.array): List of BVP signal chunks to be saved.
            filename (str): Base filename to use for saving the clips.

        Returns:
            int: The number of clips saved.
        """
        
        # Ensure the cached path directory exists; create it if it doesn't
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)

        count = 0 # Initialize a counter for the number of clips saved

        # Iterate over each BVP clip
        for i in range(len(bvps_clips)):
            # Ensure the number of input paths matches the number of label paths
            assert (len(self.inputs) == len(self.labels))
            
            # Generate unique file paths for the input and label using the base filename and count
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            
            # Append the generated file paths to the inputs and labels lists
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)

            # Save the current frame and BVP clip to their respective file paths
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])

            count += 1 # Increment the counter after saving each clip

        return count

    @staticmethod
    def resample_signal(input_signal, target_length):
        """
        Efficiently resamples a 1D input signal to the specified target length using linear interpolation.
        
        Args:
            input_signal (np.ndarray): The original 1D signal to be resampled.
            target_length (int): The desired length of the resampled signal.

        Returns:
            np.ndarray: Resampled 1D signal.
        """
        original_length = input_signal.shape[0]
        
        if original_length == target_length:
            return input_signal

        # Generate new resampled indices in the range [0, original_length - 1]
        new_indices = np.linspace(0, original_length - 1, target_length)

        # Original indices are simply [0, 1, ..., original_length - 1]
        return np.interp(new_indices, np.arange(original_length), input_signal)

    @staticmethod
    def center_crop_square(image, height, width, square_size):
        """Crops the center square from an image (supports both grayscale & color)"""
        
        y = (height - square_size) >> 1  # Bitwise shift is marginally faster than //
        x = (width - square_size) >> 1
        return image[y:y + square_size, x:x + square_size]

    @staticmethod
    def resize(image, target_size=(144, 144), downsample=False):
        """
        Resizes an image to a specified target size.
        
        Args:
            image (np.ndarray): Input image.
            target_size (tuple): Desired (width, height).
            downsample (bool): Use INTER_AREA for downsampling, INTER_LINEAR otherwise.
        
        Returns:
            np.ndarray: Resized image.
        """
        if image.shape[1::-1] != target_size:
            interpolation = cv2.INTER_AREA if downsample else cv2.INTER_LINEAR
            image = cv2.resize(image, target_size, interpolation=interpolation)
        return image

    def generate_pos_ppg(self, frames, fps=30):
        pos_ppg = POS.POS(frames, fps)

        # Compute the Hilbert transform to obtain the analytic signal
        analytic_signal = signal.hilbert(pos_ppg) 

        # Calculate the amplitude envelope of the analytic signal
        amplitude_envelope = np.abs(analytic_signal)

        # Normalize the BVP signal by its amplitude envelope
        env_norm_bvp = pos_ppg / amplitude_envelope  # Normalize by envelope

        # Return the normalized BVP signal as a NumPy array
        return np.array(env_norm_bvp)