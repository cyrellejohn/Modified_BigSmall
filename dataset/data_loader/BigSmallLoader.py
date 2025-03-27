import glob
import zipfile
import os
import re

import cv2
# from skimage.util import img_as_float
import numpy as np
import pandas as pd
import pickle 

# from unsupervised_methods.methods import POS_WANG
# from unsupervised_methods import utils
from scipy import signal
from scipy import sparse
import math
from math import ceil
from itertools import zip_longest

from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class MDMERLoader(BaseLoader):
    """The data loader for the MDMER dataset."""

    def __init__(self, name, data_path, config_data):
        """
        Initializes an instance of the UBFCrPPGLoader class.

        Args:
            name (str): The name of the data loader.
            data_path (str): The path to the directory containing raw video and BVP data.
                             The directory should have the following structure:
                             -----------------
                                  RawData/
                                  |   |-- subject1/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                                  |   |-- subject2/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                                  |...
                                  |   |-- subjectn/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                             -----------------
            config_data (CfgNode): Configuration settings for the data loader, referenced from config.py.
        """
        # Call the initializer of the superclass BaseLoader with the provided arguments
        super().__init__(name, data_path, config_data)
    
    def get_raw_data(self, data_path):
        """Returns data directories under the path of MDMER dataset"""
        
        # Use glob to find all directories matching the pattern "subject*" in the given data_path
        data_dirs = glob.glob(data_path + os.sep + "subject*")

        # Raise an error if no directories are found
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        # Extract subject index from each directory name and create a list of dictionaries
        # "index": The extracted subject index (e.g., "subject1", "subject2").
        # "path": The full path to the directory.
        dirs = [{"index": re.search('subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]

        # Return the list of dictionaries containing subject indices and paths
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        
        # If begin is 0 and end is 1, return the entire list of directories
        if begin == 0 and end == 1:
            return data_dirs

        # Calculate the total number of directories
        file_num = len(data_dirs)

        # Determine the range of indices to select based on begin and end values
        choose_range = range(int(begin * file_num), int(end * file_num))

        # Initialize a new list to store the selected subset of directories
        data_dirs_new = []

        # Append directories within the calculated range to the new list
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        # Return the subset of directories
        return data_dirs_new
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi-process data preprocessing."""
        
        # Extract the subject path and saved filename
        subject_path = data_dirs[i]['path']
        saved_filename = data_dirs[i]['index']

        # Paths to the raw video, camera trigger, and raw PPG files
        raw_video = os.path.join(subject_path, "vid.mp4")
        camera_trigger = pd.read_csv(os.path.join(subject_path, "camera.csv"), header=None)
        raw_ppg = pd.read_csv(os.path.join(subject_path, "raw_ppg.csv"), header=None, usecols=[0, 1])

        # Working directory of the AUCoding from OpenFace
        openface_au_static_dir = os.path.join(subject_path, "facs", "openface", "au_static")

        start_frame, start_time_ppg = self.get_start_time(camera_trigger.values[0, 2], raw_ppg.values[0, 0])
        
        # EXTRACT THE FRAMES FROM THE INPUT VIDEO OR .NPY FILES
        if 'None' in config_preprocess.DATA_AUG:
            # Use dataset-specific function to read video frames from .mp4 file
            frames = self.read_video(raw_video, start_frame, config_preprocess)
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Use general function to read video frames from .npy files
            frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            # Raise an error if DATA_AUG configuration is unsupported
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')
        
        # Align Data
        phys_labels = self.resample_ppg(raw_ppg.values[start_time_ppg:, 1], frames.shape[0])



        # phys_labels = self.align_data(raw_ppg, start_time, frames.shape[0])

        # EXTRACT RAW PHYSIOLOGICAL SIGNAL LABELS 
        # phys_labels = self.read_phys_labels(data_dirs[i]['path'])

        # GENERATE PSUEDO PHYSIOLOGICAL SIGNAL LABELS 
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            psuedo_phys_labels = self.generate_pos_ppg(frames, fps=self.config_data.FS)
            
        # EXTRACT AU OCCURENCE AND INTENSITY LABELS 
        au_occ, au_occ_names = self.read_au_labels(openface_au_static_dir, '_c')
        au_occ_names = au_occ_names.str.replace('_c', '_occ')
        au_int, au_int_names = self.read_au_labels(openface_au_static_dir, '_r')
        au_int_names = au_int_names.str.replace('_r', '_int')

        # Check for shape mismatches between video frames, physiological labels, au occurence and au intensity
        mismatched = [name for name, label in zip(["phys_labels", "psuedo_phys_labels", "au_occ", "au_int"], \
                                                  (phys_labels, psuedo_phys_labels, au_occ, au_int)) if label.shape[0] != frames.shape[0]]
        assert not mismatched, f"Shape mismatch in: {', '.join(mismatched)}. Expected shape: {frames.shape[0]}"

        if self.dataset_name == "train" and i == 0:
            label_list_path = os.path.dirname(self.raw_data_path)
            label_names = pd.Index(['ppg', 'pos_env_norm_ppg'])

            combine_label_names = label_names.append(au_occ_names).append(au_int_names)
            self.save_label_names(combine_label_names, label_list_path)

        # Preprocess the video frames and labels
        big_clips, small_clips, label_clips = self.preprocess(frames, phys_labels, psuedo_phys_labels, au_occ, au_int, config_preprocess)
        
        # Save the processed data with its file chunks and update the file list dictionary
        input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, label_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def get_start_time(start_camera_trigger, start_raw_ppg, sampling_rate=100):
        """
        Finds the start frame of the video based on the camera trigger and raw PPG signal.

        Args:
            start_camera_trigger (ms): The start time of the camera trigger.
            start_raw_ppg (ms): The start time of the raw PPG signal.
            sampling_rate (int): The sampling rate of the PPG signal.
        """
        start_frame = 0
        start_time_ppg = 0

        # If the camera trigger starts before the raw PPG signal, set the start frame to the difference between the two times
        if start_camera_trigger < start_raw_ppg:
            start_frame = start_raw_ppg - start_camera_trigger

        # If the raw PPG signal starts before the camera trigger, set the start time of the PPG signal to the difference between the two times
        elif start_camera_trigger > start_raw_ppg:
            start_time_ppg = start_camera_trigger - start_raw_ppg

            # Round milliseconds up to the nearest 1000
            start_time_ppg = math.ceil(start_time_ppg / 1000.0)

            # Calculate the number of samples to trim from the PPG signal
            start_time_ppg = int(start_time_ppg * sampling_rate)

        return start_frame, start_time_ppg
    
    def read_video(self, video_file, start_time, config_preprocess):
        """
        Reads a video file and returns its frames as a NumPy array in RGB format.
        
        Args:
            video_file (str): The path to the video file to be read.
            start_time (ms): The start time of the video.
        
        Returns:
            np.ndarray: A NumPy array of shape (T, H, W, 3) containing the video frames,
                        where T is the number of frames, H is the height, W is the width,
                        and 3 represents the RGB color channels.
        """
        # Initialize a VideoCapture object to read the video file
        VidObj = cv2.VideoCapture(video_file)

        # Get the height and width of the video
        height = int(VidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(VidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        square_size = min(height, width)  # Determine the square size
        
        # Set the video position to the start tim
        VidObj.set(cv2.CAP_PROP_POS_MSEC, start_time)

        # Read the first frame from the video
        success, frame = VidObj.read()

        # Initialize a list to store the center cropped frames
        center_cropped_frames = list()
        i = 0
        # Loop to read frames until no more frames are available
        while success:
            # Center crop the frame to a square
            center_cropped_frame = self.center_crop_square(frame, height, width, square_size)

            # Resize the frame optionally
            if not config_preprocess.CROP_FACE.DO_CROP_FACE:
                center_cropped_frame = self.resize(center_cropped_frame, downsample=True)

            # Convert the frame from BGR to RGB format
            center_cropped_frame = cv2.cvtColor(np.array(center_cropped_frame), cv2.COLOR_BGR2RGB)

            # Convert the frame to a NumPy array
            center_cropped_frame = np.asarray(center_cropped_frame)

            # Append the frame to the center crop frames list
            center_cropped_frames.append(center_cropped_frame)

            # Read the next frame
            success, frame = VidObj.read()
            i += 1
            print(i)
        # Convert the list of frames to a NumPy array and return it
        return np.asarray(center_cropped_frames)
    
    @staticmethod
    def read_phys_labels(working_dir):
        ppg = np.array([float(x) for x in open(os.path.join(working_dir, "ground_truth.txt")).readline().split()])

        return ppg

    @staticmethod
    def read_au_labels(aucoding_dir, pattern):
        full_path = os.path.join(aucoding_dir, "AUCoding.csv")
        
        # Read the CSV file and filter columns ending with a pattern
        df = pd.read_csv(full_path, header=0)
        au_label = df.filter(like=pattern).to_numpy()
        label_names = df.filter(like=pattern).columns

        return au_label, label_names

    @staticmethod
    def save_label_names(data, label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        # Overwrite the file instead of appending to avoid duplication
        with open(full_path, "w") as file:
            for column in data:  # Iterate directly over Index
                file.write(f"{column.strip()}\n")  # Strip leading/trailing spaces

    def load_label_names(label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        with open(full_path, "r") as file:
            label_list = [line.strip() for line in file]  # Read and strip spaces

        return label_list

    def preprocess(self, frames, phys_labels, psuedo_phys_labels, au_occ, au_int, config_preprocess):
        ##########################################
        ########## PREPROCESSING FRAMES ##########
        ##########################################

        # CROP FACE AND RESIZE FRAMES
        if config_preprocess.CROP_FACE.DO_CROP_FACE:
            frames = self.crop_face_resize(frames,
                                           config_preprocess.CROP_FACE.BACKEND,
                                           config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                           config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                           config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                           config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                           config_preprocess.BIGSMALL.RESIZE.BIG_W,
                                           config_preprocess.BIGSMALL.RESIZE.BIG_H)

        # PREPROCESS VIDEO DATA FOR BIG PATH
        big_data = list()
        for data_type in config_preprocess.BIGSMALL.BIG_DATA_TYPE:
            f_c = frames.copy() # Copy frames to avoid modifying original data

            if data_type == "Raw":
                big_data.append(f_c) # Append raw frames
            elif data_type == "DiffNormalized":
                big_data.append(BaseLoader.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                big_data.append(BaseLoader.standardized_data(f_c)) # Apply standardization
            else:
                raise ValueError("Unsupported data type!") # Raise error for unsupported data types
        big_data = np.concatenate(big_data, axis=-1) # Concatenate transformed data along the last axis

        # PREPROCESS VIDEO DATA FOR SMALL PATH
        small_data = list()
        for data_type in config_preprocess.BIGSMALL.SMALL_DATA_TYPE:
            f_c = frames.copy() # Copy frames to avoid modifying original data

            if data_type == "Raw":
                small_data.append(f_c) # Append raw frames
            elif data_type == "DiffNormalized":
                small_data.append(BaseLoader.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                small_data.append(BaseLoader.standardized_data(f_c)) # Apply standardization
            else:
                raise ValueError("Unsupported data type!") # Raise error for unsupported data types
        small_data = np.concatenate(small_data, axis=-1) # Concatenate transformed data along the last axis

        # RESIZE FRAMES TO SMALL SIZE: 9x9 AT DEFAULT
        small_data = self.crop_face_resize(small_data,
                                           config_preprocess.CROP_FACE.DO_CROP_FACE,
                                           config_preprocess.CROP_FACE.BACKEND,
                                           config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                           config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                           config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                           config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                           config_preprocess.BIGSMALL.RESIZE.SMALL_W,
                                           config_preprocess.BIGSMALL.RESIZE.SMALL_H)


        ##########################################
        ########## PREPROCESSING LABELS ##########
        ##########################################

        # REMOVING OUTLIERS - PENDING

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            phys_labels = BaseLoader.diff_normalize_label(phys_labels)
            psuedo_phys_labels = BaseLoader.diff_normalize_label(psuedo_phys_labels)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            phys_labels = BaseLoader.standardized_label(phys_labels)
            psuedo_phys_labels = BaseLoader.standardized_label(psuedo_phys_labels)
        else:
            raise ValueError("Unsupported label type!")


        ######################################
        ##########  COMBINE LABELS ###########
        ######################################

        # If phys_labels and psuedo_phys_labels is 1D, reshape it to 2D with one column
        if phys_labels.ndim == 1:
            phys_labels = phys_labels.reshape(-1, 1)

        if psuedo_phys_labels.ndim == 1:
            psuedo_phys_labels = psuedo_phys_labels.reshape(-1, 1)

        labels = np.concatenate((phys_labels, psuedo_phys_labels, au_occ, au_int), axis=1)


        ######################################
        ######## CHUNK DATA / LABELS #########
        ######################################

        # Chunk data and labels into smaller segments
        if config_preprocess.DO_CHUNK:
            chunk_len = config_preprocess.CHUNK_LENGTH
            big_clips, small_clips, labels_clips = self.chunk(big_data, small_data, labels, chunk_len)
        else:
            big_clips = np.array([big_data])
            small_clips = np.array([small_data])
            labels_clips = np.array([labels])

        # Return processed video frames and labels
        return big_clips, small_clips, labels_clips

    def chunk(self, big_frames, small_frames, labels, chunk_len):
        """
        Splits the input video frames and corresponding labels into fixed-length chunks.

        Args:
            big_frames (np.array): The array of big video frames to be chunked.
            small_frames (np.array): The array of small video frames to be chunked.
            labels (np.array): The array of labels to be chunked.
            chunk_length (int): The length of each chunk.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - big_clips (np.array): Chunks of big video frames.
                - small_clips (np.array): Chunks of small video frames.
                - labels_clips (np.array): Chunks of labels.
        """

        # Calculate the number of complete chunks that can be created
        clip_num = labels.shape[0] // chunk_len
        
        # Create chunks of frames and BVP signals
        big_clips, small_clips, labels_clips = map(np.array, zip(*[
            (big_frames[i * chunk_len:(i + 1) * chunk_len], 
            small_frames[i * chunk_len:(i + 1) * chunk_len], 
            labels[i * chunk_len:(i + 1) * chunk_len]) 
            for i in range(clip_num)
        ]))

        return big_clips, small_clips, labels_clips

    def save_multi_process(self, big_clips, small_clips, label_clips, filename):
        """
        Saves preprocessed video frames and corresponding labels to disk in a multi-process environment.

        Args:
            big_clips (np.array): Array of big video frame clips to be saved.
            small_clips (np.array): Array of small video frame clips to be saved.
            label_clips (np.array): Array of labels corresponding to each video frame clip.
            filename (str): Base filename to use for saving the clips.

        Returns:
            tuple: Two lists containing the paths of the saved input and label files, respectively.
        """
        # Ensure the cached path directory exists
        os.makedirs(self.cached_path, exist_ok=True)

        # Ensure the number of inputs matches the number of labels
        assert len(big_clips) == len(small_clips) == len(label_clips), \
            f"Mismatch in list lengths: big_clips={len(big_clips)}, small_clips={len(small_clips)}, label_clips={len(label_clips)}"

        input_path_name_list = []
        label_path_name_list = []

        # Iterate over each label clip
        for i in range(len(label_clips)):
            # Construct file paths
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{i}.pickle")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{i}.npy")

            frames_dict = {0: big_clips[i], 1: small_clips[i]}

            # Save label as .npy
            np.save(label_path_name, label_clips[i])

            # Save frames as .pickle
            with open(input_path_name, 'wb') as handle:
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Store file paths
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

        return input_path_name_list, label_path_name_list

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
        labels = [input_file.replace("input", "label").replace('.pickle', '.npy') for input_file in inputs]

        # Store the sorted input file paths in the class attribute
        self.inputs = inputs

        # Store the label file paths in the class attribute
        self.labels = labels

        # Calculate and store the length of the preprocessed dataset
        self.preprocessed_data_len = len(inputs)

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

        ##########################################
        ############## LOADING DATA ##############
        ##########################################

        # Load the data from the preprocessed files using the index
        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        # Transpose the data array based on the specified data format and convert to float32 for compatibility with PyTorch
        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        ############################################
        ############## LOADING LABELS ##############
        ############################################

        # Load the label from the preprocessed files using the index
        label = np.load(self.labels[index])

        # Convert label to float32 for compatibility with PyTorch
        label = np.float32(label)
        

        ###############################################
        ############## GETTING FILE NAME ##############
        ###############################################

        # Extract the file path of the current data item
        item_path = self.inputs[index]

        # Extract the filename from the file path
        item_path_filename = item_path.split(os.sep)[-1]

        # Find the index to split the filename to get the base filename and chunk ID
        split_idx = item_path_filename.rindex('_')

        # Extract the base filename (e.g., '501' from '501_input0.npy')
        filename = item_path_filename[:split_idx]


        ################################################
        ############## GETTING CHUNK ID ################
        ################################################

        # Extract the chunk ID (e.g., '0' from '501_input0.npy')
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

        # Return the data, label, base filename, and chunk ID
        return data, label, filename, chunk_id

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
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            
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

class UBFCrPPGLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data):
        """
        Initializes an instance of the UBFCrPPGLoader class.

        Args:
            name (str): The name of the data loader.
            data_path (str): The path to the directory containing raw video and BVP data.
                             The directory should have the following structure:
                             -----------------
                                  RawData/
                                  |   |-- subject1/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                                  |   |-- subject2/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                                  |...
                                  |   |-- subjectn/
                                  |       |-- vid.avi
                                  |       |-- ground_truth.txt
                             -----------------
            config_data (CfgNode): Configuration settings for the data loader, referenced from config.py.
        """
        # Call the initializer of the superclass BaseLoader with the provided arguments
        super().__init__(name, data_path, config_data)
    
    def get_raw_data(self, data_path):
        """Returns data directories under the path of UBFC-rPPG dataset"""
        
        # Use glob to find all directories matching the pattern "subject*" in the given data_path
        data_dirs = glob.glob(data_path + os.sep + "subject*")

        # Raise an error if no directories are found
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        # Extract subject index from each directory name and create a list of dictionaries
        # "index": The extracted subject index (e.g., "subject1", "subject2").
        # "path": The full path to the directory.
        dirs = [{"index": re.search('subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        
        # Return the list of dictionaries containing subject indices and paths
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """
        Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        
        # If begin is 0 and end is 1, return the entire list of directories
        if begin == 0 and end == 1:
            return data_dirs

        # Calculate the total number of directories
        file_num = len(data_dirs)

        # Determine the range of indices to select based on begin and end values
        choose_range = range(int(begin * file_num), int(end * file_num))

        # Initialize a new list to store the selected subset of directories
        data_dirs_new = []

        # Append directories within the calculated range to the new list
        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        # Return the subset of directories
        return data_dirs_new
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi-process data preprocessing."""

        # Working directory of the AUCoding from OpenFace
        aucoding_dir = os.path.join(data_dirs[i]['path'], "facs", "openface", "au_static")

        # Extract the filename and index for saving processed data
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # EXTRACT THE FRAMES FROM THE INPUT VIDEO OR .NPY FILES
        if 'None' in config_preprocess.DATA_AUG:
            # Use dataset-specific function to read video frames from .avi file
            frames = self.read_video(os.path.join(data_dirs[i]['path'], "vid.avi"), config_preprocess)
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Use general function to read video frames from .npy files
            frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            # Raise an error if DATA_AUG configuration is unsupported
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')
        
        # EXTRACT RAW PHYSIOLOGICAL SIGNAL LABELS 
        phys_labels = self.read_phys_labels(data_dirs[i]['path'])

        # GENERATE PSUEDO PHYSIOLOGICAL SIGNAL LABELS 
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            psuedo_phys_labels = self.generate_pos_ppg(frames, fps=self.config_data.FS)
            
        # EXTRACT AU OCCURENCE AND INTENSITY LABELS 
        au_occ, au_occ_names = self.read_au_labels(aucoding_dir, '_c')
        au_occ_names = au_occ_names.str.replace('_c', '_occ')
        au_int, au_int_names = self.read_au_labels(aucoding_dir, '_r')
        au_int_names = au_int_names.str.replace('_r', '_int')

        # Check for shape mismatches between video frames, physiological labels, au occurence and au intensity
        mismatched = [name for name, label in zip(["phys_labels", "psuedo_phys_labels", "au_occ", "au_int"], \
                                                  (phys_labels, psuedo_phys_labels, au_occ, au_int)) if label.shape[0] != frames.shape[0]]
        assert not mismatched, f"Shape mismatch in: {', '.join(mismatched)}. Expected shape: {frames.shape[0]}"

        if self.dataset_name == "train" and i == 0:
            label_list_path = os.path.dirname(self.raw_data_path)
            label_names = pd.Index(['ppg', 'pos_env_norm_ppg'])

            combine_label_names = label_names.append(au_occ_names).append(au_int_names)
            self.save_label_names(combine_label_names, label_list_path)

        # Preprocess the video frames and labels
        big_clips, small_clips, label_clips = self.preprocess(frames, phys_labels, psuedo_phys_labels, au_occ, au_int, config_preprocess)
        
        # Save the processed data with its file chunks and update the file list dictionary
        input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, label_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file, config_preprocess):
        """
        Reads a video file and returns its frames as a NumPy array in RGB format.
        
        Args:
            video_file (str): The path to the video file to be read.
        
        Returns:
            np.ndarray: A NumPy array of shape (T, H, W, 3) containing the video frames,
                        where T is the number of frames, H is the height, W is the width,
                        and 3 represents the RGB color channels.
        """
        # Initialize a VideoCapture object to read the video file
        VidObj = cv2.VideoCapture(video_file)

        # Get the height and width of the video
        height = int(VidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(VidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        square_size = min(height, width)  # Determine the square size
        
        # Set the video position to the start (0 milliseconds)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

        # Read the first frame from the video
        success, frame = VidObj.read()

        # Initialize a list to store the center cropped frames
        center_cropped_frames = list()

        # Loop to read frames until no more frames are available
        while success:
            # Center crop the frame to a square
            center_cropped_frame = BaseLoader.center_crop_square(frame, height, width, square_size)

            # Resize the frame optionally
            if not config_preprocess.CROP_FACE.DO_CROP_FACE:
                center_cropped_frame = BaseLoader.resize(center_cropped_frame, downsample=True)

            # Convert the frame from BGR to RGB format
            center_cropped_frame = cv2.cvtColor(np.array(center_cropped_frame), cv2.COLOR_BGR2RGB)

            # Convert the frame to a NumPy array
            center_cropped_frame = np.asarray(center_cropped_frame)

            # Append the frame to the center crop frames list
            center_cropped_frames.append(center_cropped_frame)

            # Read the next frame
            success, frame = VidObj.read()
        
        # Convert the list of frames to a NumPy array and return it
        return np.asarray(center_cropped_frames)
    
    @staticmethod
    def read_phys_labels(working_dir):
        ppg = np.array([float(x) for x in open(os.path.join(working_dir, "ground_truth.txt")).readline().split()])

        return ppg

    @staticmethod
    def read_au_labels(aucoding_dir, pattern):
        full_path = os.path.join(aucoding_dir, "AUCoding.csv")
        
        # Read the CSV file and filter columns ending with a pattern
        df = pd.read_csv(full_path, header=0)
        au_label = df.filter(like=pattern).to_numpy()
        label_names = df.filter(like=pattern).columns

        return au_label, label_names

    @staticmethod
    def save_label_names(data, label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        # Overwrite the file instead of appending to avoid duplication
        with open(full_path, "w") as file:
            for column in data:  # Iterate directly over Index
                file.write(f"{column.strip()}\n")  # Strip leading/trailing spaces

    def load_label_names(label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        with open(full_path, "r") as file:
            label_list = [line.strip() for line in file]  # Read and strip spaces

        return label_list

    def preprocess(self, frames, phys_labels, psuedo_phys_labels, au_occ, au_int, config_preprocess):
        ##########################################
        ########## PREPROCESSING FRAMES ##########
        ##########################################

        # CROP FACE AND RESIZE FRAMES
        if config_preprocess.CROP_FACE.DO_CROP_FACE:
            frames = self.crop_face_resize(frames,
                                           config_preprocess.CROP_FACE.BACKEND,
                                           config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                           config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                           config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                           config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                           config_preprocess.BIGSMALL.RESIZE.BIG_W,
                                           config_preprocess.BIGSMALL.RESIZE.BIG_H)

        # PREPROCESS VIDEO DATA FOR BIG PATH
        big_data = list()
        for data_type in config_preprocess.BIGSMALL.BIG_DATA_TYPE:
            f_c = frames.copy() # Copy frames to avoid modifying original data

            if data_type == "Raw":
                big_data.append(f_c) # Append raw frames
            elif data_type == "DiffNormalized":
                big_data.append(BaseLoader.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                big_data.append(BaseLoader.standardized_data(f_c)) # Apply standardization
            else:
                raise ValueError("Unsupported data type!") # Raise error for unsupported data types
        big_data = np.concatenate(big_data, axis=-1) # Concatenate transformed data along the last axis

        # PREPROCESS VIDEO DATA FOR SMALL PATH
        small_data = list()
        for data_type in config_preprocess.BIGSMALL.SMALL_DATA_TYPE:
            f_c = frames.copy() # Copy frames to avoid modifying original data

            if data_type == "Raw":
                small_data.append(f_c) # Append raw frames
            elif data_type == "DiffNormalized":
                small_data.append(BaseLoader.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                small_data.append(BaseLoader.standardized_data(f_c)) # Apply standardization
            else:
                raise ValueError("Unsupported data type!") # Raise error for unsupported data types
        small_data = np.concatenate(small_data, axis=-1) # Concatenate transformed data along the last axis

        # RESIZE FRAMES TO SMALL SIZE: 9x9 AT DEFAULT
        small_data = self.crop_face_resize(small_data,
                                           config_preprocess.CROP_FACE.DO_CROP_FACE,
                                           config_preprocess.CROP_FACE.BACKEND,
                                           config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                           config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                           config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                           config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                           config_preprocess.BIGSMALL.RESIZE.SMALL_W,
                                           config_preprocess.BIGSMALL.RESIZE.SMALL_H)


        ##########################################
        ########## PREPROCESSING LABELS ##########
        ##########################################

        # REMOVING OUTLIERS - PENDING

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            phys_labels = BaseLoader.diff_normalize_label(phys_labels)
            psuedo_phys_labels = BaseLoader.diff_normalize_label(psuedo_phys_labels)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            phys_labels = BaseLoader.standardized_label(phys_labels)
            psuedo_phys_labels = BaseLoader.standardized_label(psuedo_phys_labels)
        else:
            raise ValueError("Unsupported label type!")


        ######################################
        ##########  COMBINE LABELS ###########
        ######################################

        # If phys_labels and psuedo_phys_labels is 1D, reshape it to 2D with one column
        if phys_labels.ndim == 1:
            phys_labels = phys_labels.reshape(-1, 1)

        if psuedo_phys_labels.ndim == 1:
            psuedo_phys_labels = psuedo_phys_labels.reshape(-1, 1)

        labels = np.concatenate((phys_labels, psuedo_phys_labels, au_occ, au_int), axis=1)


        ######################################
        ######## CHUNK DATA / LABELS #########
        ######################################

        # Chunk data and labels into smaller segments
        if config_preprocess.DO_CHUNK:
            chunk_len = config_preprocess.CHUNK_LENGTH
            big_clips, small_clips, labels_clips = self.chunk(big_data, small_data, labels, chunk_len)
        else:
            big_clips = np.array([big_data])
            small_clips = np.array([small_data])
            labels_clips = np.array([labels])

        # Return processed video frames and labels
        return big_clips, small_clips, labels_clips

    def chunk(self, big_frames, small_frames, labels, chunk_len):
        """
        Splits the input video frames and corresponding labels into fixed-length chunks.

        Args:
            big_frames (np.array): The array of big video frames to be chunked.
            small_frames (np.array): The array of small video frames to be chunked.
            labels (np.array): The array of labels to be chunked.
            chunk_length (int): The length of each chunk.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - big_clips (np.array): Chunks of big video frames.
                - small_clips (np.array): Chunks of small video frames.
                - labels_clips (np.array): Chunks of labels.
        """

        # Calculate the number of complete chunks that can be created
        clip_num = labels.shape[0] // chunk_len
        
        # Create chunks of frames and BVP signals
        big_clips, small_clips, labels_clips = map(np.array, zip(*[
            (big_frames[i * chunk_len:(i + 1) * chunk_len], 
            small_frames[i * chunk_len:(i + 1) * chunk_len], 
            labels[i * chunk_len:(i + 1) * chunk_len]) 
            for i in range(clip_num)
        ]))

        return big_clips, small_clips, labels_clips

    def save_multi_process(self, big_clips, small_clips, label_clips, filename):
        """
        Saves preprocessed video frames and corresponding labels to disk in a multi-process environment.

        Args:
            big_clips (np.array): Array of big video frame clips to be saved.
            small_clips (np.array): Array of small video frame clips to be saved.
            label_clips (np.array): Array of labels corresponding to each video frame clip.
            filename (str): Base filename to use for saving the clips.

        Returns:
            tuple: Two lists containing the paths of the saved input and label files, respectively.
        """
        # Ensure the cached path directory exists
        os.makedirs(self.cached_path, exist_ok=True)

        # Ensure the number of inputs matches the number of labels
        assert len(big_clips) == len(small_clips) == len(label_clips), \
            f"Mismatch in list lengths: big_clips={len(big_clips)}, small_clips={len(small_clips)}, label_clips={len(label_clips)}"

        input_path_name_list = []
        label_path_name_list = []

        # Iterate over each label clip
        for i in range(len(label_clips)):
            # Construct file paths
            input_path_name = os.path.join(self.cached_path, f"{filename}_input{i}.pickle")
            label_path_name = os.path.join(self.cached_path, f"{filename}_label{i}.npy")

            frames_dict = {0: big_clips[i], 1: small_clips[i]}

            # Save label as .npy
            np.save(label_path_name, label_clips[i])

            # Save frames as .pickle
            with open(input_path_name, 'wb') as handle:
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Store file paths
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

        return input_path_name_list, label_path_name_list

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
        labels = [input_file.replace("input", "label").replace('.pickle', '.npy') for input_file in inputs]

        # Store the sorted input file paths in the class attribute
        self.inputs = inputs

        # Store the label file paths in the class attribute
        self.labels = labels

        # Calculate and store the length of the preprocessed dataset
        self.preprocessed_data_len = len(inputs)

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

        ##########################################
        ############## LOADING DATA ##############
        ##########################################

        # Load the data from the preprocessed files using the index
        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

        # Transpose the data array based on the specified data format and convert to float32 for compatibility with PyTorch
        if self.data_format == 'NDCHW':
            data[0] = np.float32(np.transpose(data[0], (0, 3, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (0, 3, 1, 2)))
        elif self.data_format == 'NCDHW':
            data[0] = np.float32(np.transpose(data[0], (3, 0, 1, 2)))
            data[1] = np.float32(np.transpose(data[1], (3, 0, 1, 2)))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        ############################################
        ############## LOADING LABELS ##############
        ############################################

        # Load the label from the preprocessed files using the index
        label = np.load(self.labels[index])

        # Convert label to float32 for compatibility with PyTorch
        label = np.float32(label)
        

        ###############################################
        ############## GETTING FILE NAME ##############
        ###############################################

        # Extract the file path of the current data item
        item_path = self.inputs[index]

        # Extract the filename from the file path
        item_path_filename = item_path.split(os.sep)[-1]

        # Find the index to split the filename to get the base filename and chunk ID
        split_idx = item_path_filename.rindex('_')

        # Extract the base filename (e.g., '501' from '501_input0.npy')
        filename = item_path_filename[:split_idx]


        ################################################
        ############## GETTING CHUNK ID ################
        ################################################

        # Extract the chunk ID (e.g., '0' from '501_input0.npy')
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

        # Return the data, label, base filename, and chunk ID
        return data, label, filename, chunk_id

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
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.pickle".format(fname)))
            
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