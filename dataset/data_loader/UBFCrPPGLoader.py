"""
The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

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
        
        # Extract the filename and index for saving processed data
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Use dataset-specific function to read video frames from .avi file
            frames = self.read_video(os.path.join(data_dirs[i]['path'], "vid.avi"))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Use general function to read video frames from .npy files
            frames = self.read_npy_video(glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            # Raise an error if DATA_AUG configuration is unsupported
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')
        
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            # Generate pseudo PPG labels if specified in the configuration
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS) # NOT IMPLEMENTED IN BASELOADER | NOT FINAL
        else:
            # Read ground truth BVP data from text file
            bvps = self.read_wave(os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
        
        # Preprocess the frames and BVP data
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        
        # Save the processed data and update the file list dictionary
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
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
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        # Open the BVP file in read mode
        with open(bvp_file, "r") as f:
            # Read the entire content of the file into a string
            str1 = f.read()
            
            # Split the string into a list of lines
            str1 = str1.split("\n")
            
            # Take the first line, split it by spaces, and convert each element to a float
            bvp = [float(x) for x in str1[0].split()]

        # Convert the list of floats to a NumPy array and return it
        return np.asarray(bvp)