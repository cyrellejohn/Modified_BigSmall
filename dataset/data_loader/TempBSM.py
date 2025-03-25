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

from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class UBFCrPPGLoader(BaseLoader):
    
    def __init__(self, name, data_path, config_data):
        """
        Initializes the BigSmallLoader with the given dataset name, data path, and configuration data.

        Args:
            name (str): The name of the dataset.
            data_path (str): The path to the raw data.
            config_data (object): Configuration object containing various settings and paths.

        Attributes:
            inputs (list): List to store input data.
            labels (list): List to store corresponding labels.
            dataset_name (str): Name of the dataset.
            raw_data_path (str): Path to the raw data.
            cached_path (str): Path to the cached preprocessed data.
            file_list_path (str): Path to the file list.
            preprocessed_data_len (int): Length of the preprocessed dataset.
            data_format (str): Format of the data (e.g., 'NDCHW', 'NCDHW').
            do_preprocess (bool): Flag indicating whether to preprocess the data.

        Raises:
            ValueError: If the cached path does not exist and preprocessing is not enabled.
        """

        # Initialize attributes
        self.inputs = list()
        self.labels = list()
        self.dataset_name = name
        self.raw_data_path = data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        
        # Ensure valid data split configuration
        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)

        # Preprocess data if required
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data, config_data.BEGIN, config_data.END)
        else:
            # Check if cached data path exists
            if not os.path.exists(self.cached_path):
                raise ValueError(self.dataset_name, 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            
            # Generate file list if it doesn't exist
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            # Load preprocessed data
            self.load_preprocessed_data()
            
        # Print paths and dataset length
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def get_raw_data(self, data_path):
        """Returns data directories under the path of UBFC-rPPG dataset)"""
        
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

    def preprocess_dataset(self, data_dirs, config_data, begin, end):
        print('Starting Preprocessing...')

        # GET DATASET INFORMATION (PATHS AND OTHER META DATA REGARDING ALL VIDEO TRIALS)
        data_dirs = self.split_raw_data(data_dirs, begin, end)

        # REMOVE ALREADY PREPROCESSED SUBJECTS
        data_dirs = self.adjust_data_dirs(data_dirs, config_data)

        # CREATE CACHED DATA PATH
        cached_path = config_data.CACHED_PATH
        if not os.path.exists(cached_path):
            os.makedirs(cached_path, exist_ok=True)

        # READ RAW DATA, PREPROCESS, AND SAVE PROCESSED DATA FILES
        file_list_dict = self.multi_process_manager(data_dirs, config_data) 
        
        self.build_file_list(file_list_dict)  # build file list

        self.load_preprocessed_data()  # load all data and corresponding labels (sorted for consistency)
        
        # Log the total number of raw files that were preprocessed
        print("Total Number of raw files preprocessed:", len(data_dirs), end='\n\n')
        print("Num loaded files", self.preprocessed_data_len)

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

    def adjust_data_dirs(self, data_dirs, config_preprocess):
        """ Reads data folder and only preprocess files that have not already been preprocessed."""

        cached_path = config_preprocess.CACHED_PATH
        file_list = glob.glob(os.path.join(cached_path, '*label*.npy'))
        trial_list = [f.replace(cached_path, '').split('_')[0].replace(os.sep, '') for f in file_list]
        trial_list = list(set(trial_list)) # get a list of completed video trials

        adjusted_data_dirs = []
        for d in data_dirs:
            idx = d['index']

            if not idx in trial_list: # if trial has already been processed
                adjusted_data_dirs.append(d)
        return adjusted_data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_data, i, file_list_dict):
        """Invoked by preprocess_dataset for multi-process data preprocessing."""
        
        config_preprocess = config_data.PREPROCESS
        
        # Extract the filename and index for saving processed data
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Use dataset-specific function to read video frames from .avi file
            frames = self.read_video(os.path.join(data_dirs[i]['path'],"vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Use general function to read video frames from .npy files
            frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            # Raise an error if DATA_AUG configuration is unsupported
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            # Generate pseudo PPG labels if specified in the configuration
            labels = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS) # NOT IMPLEMENTED IN BASELOADER | NOT FINAL
        else:
            # Read ground truth BVP data from text file
            labels = self.read_labels(os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
        
        # Preprocess the big and small frames and labels
        big_clips, small_clips, labels_clips = self.preprocess(frames, labels, config_preprocess)

        # Save the processed data and update the file list dictionary
        count, input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, labels_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
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
        
        # Set the video position to the start (0 milliseconds)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

        # Read the first frame from the video
        success, frame = VidObj.read()

        # Initialize a list to store the frames
        frames = list()

        # Loop to read frames until no more frames are available
        while success:
            # Convert the frame from BGR to RGB format
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

            # Convert the frame to a NumPy array
            frame = np.asarray(frame)

            # Append the frame to the frames list
            frames.append(frame)

            # Read the next frame
            success, frame = VidObj.read()
        
        # Convert the list of frames to a NumPy array and return it
        return np.asarray(frames)

    @staticmethod
    def read_labels(labels_file):
        # Open the label file in read mode
        with open(labels_file, "r") as f:
            # Read the entire content of the file into a string
            str1 = f.read()

            # Split the string into a list of lines
            str1 = str1.split("\n")

            # Take the first line, split it by spaces, and convert each element to a float
            bvp = [float(x) for x in str1[0].split()]

        # Convert the list of floats to a NumPy array and return it
        return np.asarray(bvp)

    def preprocess(self, frames, labels, config_preprocess):
        #######################################
        ########## PROCESSING FRAMES ##########
        #######################################

        # RESIZE FRAMES TO BIG SIZE (144x144 DEFAULT)
        frames = self.crop_face_resize(frames,
                                       config_preprocess.CROP_FACE.DO_CROP_FACE,  
                                       config_preprocess.CROP_FACE.BACKEND,
                                       config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
                                       config_preprocess.CROP_FACE.LARGE_BOX_COEF,
                                       config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
                                       config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                       config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
                                       config_preprocess.BIGSMALL.RESIZE.BIG_W,
                                       config_preprocess.BIGSMALL.RESIZE.BIG_H)
        
        # PROCESS BIG FRAMES
        big_data = list()
        for data_type in config_preprocess.BIGSMALL.BIG_DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                big_data.append(f_c)
            elif data_type == "DiffNormalized":
                big_data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                big_data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        big_data = np.concatenate(big_data, axis=-1)

        # PROCESS SMALL FRAMES
        small_data = list()
        for data_type in config_preprocess.BIGSMALL.SMALL_DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                small_data.append(f_c)
            elif data_type == "DiffNormalized":
                small_data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                small_data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        small_data = np.concatenate(small_data, axis=-1)

        # RESIZE SMALL FRAMES TO LOWER RESOLUTION (9x9 DEFAULT)
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
        

        ######################################
        ########## PROCESSED LABELS ##########
        ######################################

        if config_preprocess.LABEL_TYPE == "Raw":
            pass 
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            labels = BaseLoader.diff_normalize_label(labels) 
        elif config_preprocess.LABEL_TYPE == "Standardized":
            labels = BaseLoader.standardized_label(labels)
        else:
            raise ValueError("Unsupported label type!")


        ######################################
        ########## CHUNK DATA / LABELS ######
        ######################################

        if config_preprocess.DO_CHUNK:
            big_clips, small_clips, labels_clips = self.chunk(big_data, small_data, labels, config_preprocess.CHUNK_LENGTH)
        else:
            big_clips = np.array([big_data])
            small_clips = np.array([small_data])
            labels_clips = np.array([labels])

        return big_clips, small_clips, labels_clips

    def chunk(self, big_frames, small_frames, labels, chunk_len):
        """Chunks the data into clips."""

        clip_num = labels.shape[0] // chunk_len
        big_clips = [big_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        small_clips = [small_frames[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]
        labels_clips = [labels[i * chunk_len:(i + 1) * chunk_len] for i in range(clip_num)]

        return np.array(big_clips), np.array(small_clips), np.array(labels_clips)

    def save_multi_process(self, big_clips, small_clips, label_clips, filename):
        """
        Saves processed video data and corresponding labels to disk.

        Args:
            big_clips (list): List of big-sized video clips.
            small_clips (list): List of small-sized video clips.
            label_clips (list): List of labels corresponding to each video clip.
            filename (str): Base filename to use for saving the data.
            config_preprocess (object): Configuration object for preprocessing settings.

        Returns:
            tuple: A tuple containing:
                - count (int): Number of processed clips saved.
                - input_path_name_list (list): List of file paths for saved input data.
                - label_path_name_list (list): List of file paths for saved labels.
        """

        # Ensure the cached path directory exists, create if not
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        
        # Initialize counter and lists to store file paths
        count = 0
        input_path_name_list = []
        label_path_name_list = []

        # Iterate over each label clip
        for i in range(len(label_clips)):
            # Ensure the lengths of big_clips, small_clips, and label_clips are equal
            assert (len(big_clips) == len(label_clips) and len(small_clips) == len(label_clips))
            
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.pickle".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            
            # Construct file paths for input data
            frames_dict = dict()
            frames_dict[0] = big_clips[i]
            frames_dict[1] = small_clips[i]

            # Append file paths to respective lists
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)

            # Save labels to a .npy file
            np.save(label_path_name, label_clips[i]) # save out labels npy file
            
            # Save frame dictionary to a .pickle file
            with open(input_path_name, 'wb') as handle: # save out frame dict pickle file
                pickle.dump(frames_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Increment the count of processed clips
            count += 1

        # Return the count and lists of file paths
        return count, input_path_name_list, label_path_name_list

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

    def __getitem__(self, index):
        with open(self.inputs[index], 'rb') as handle:
            data = pickle.load(handle)

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

        label = np.load(self.labels[index])
        label = np.float32(label)
        
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.index('_')
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id
