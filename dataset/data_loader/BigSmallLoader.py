import glob
import zipfile
import os
import re
from pathlib import Path

import cv2
# from skimage.util import img_as_float
import numpy as np
import pandas as pd
import pickle 

from signal_processing.ppg import filters
from unsupervised_methods.algorithm import POS

# from unsupervised_methods.methods import POS_WANG
# from unsupervised_methods import utils
from scipy.signal import medfilt
from scipy import sparse
import math
from math import ceil
from itertools import zip_longest
from scipy import signal

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

        # Static Configuration
        phys_label_type = 'raw'                     # Options: 'raw', 'filtered', 'both'
        support_backend = 'libreface'                # Options: 'libreface', 'openface', 'both'
        openface_au_setting = 'dynamic'             # Options: 'dynamic', 'dynamic_wild', 'static', 'static_wild'
        emotion_source = 'libreface'                 # Options: 'original', 'libreface'
        original_emotion_setting = 'both'           # Options: 'discrete', 'vad', 'both'
        include_dominance = False

        # Subject info
        subject_info = data_dirs[i]
        subject_path = subject_info['path']
        subject_number = subject_info['index']
        subject_video = os.path.join(subject_path, "vid.mp4")
        libreface_dir = os.path.join(subject_path, "facs", "libreface")
        openface_dir = os.path.join(subject_path, "facs", "openface")

        # Load label data
        def load_csv_values(path, col_slice=slice(1, None)):
            data = pd.read_csv(path, skiprows=1, header=None).values
            return data[:, col_slice] if isinstance(col_slice, slice) else data[:, [col_slice]]

        subject_emotion = load_csv_values(os.path.join(subject_path, "emotion.csv"))
        subject_vad = load_csv_values(os.path.join(subject_path, "vad.csv"))
        subject_ppg = load_csv_values(os.path.join(subject_path, "ppg.csv"), col_slice=3)

        # Load frames
        data_aug = config_preprocess.DATA_AUG
        if 'None' in data_aug:
            frames = self.read_video(subject_video, config_preprocess)
        elif 'Motion' in data_aug:
            npy_files = glob.glob(os.path.join(subject_path, '*.npy'))
            frames = self.read_npy_video(npy_files)
        else:
            raise ValueError(f'Unsupported DATA_AUG: {data_aug}')

        frame_count = frames.shape[0]

        # PPG Preprocessing
        ppg_features = self.preprocess_ppg(subject_ppg.flatten(), frame_count)
        ppg_colnames_map = {'raw': ['ppg'], 'filtered': ['filtered_ppg'], 'both': ['ppg', 'filtered_ppg']}
        phys_label_idx_map = {'raw': 0, 'filtered': 1, 'both': slice(None)}

        if phys_label_type not in ppg_colnames_map:
            raise ValueError(f"Unsupported PHYS_LABEL_TYPE: {phys_label_type}")

        phys_labels = ppg_features[:, phys_label_idx_map[phys_label_type]]
        ppg_colnames = ppg_colnames_map[phys_label_type] + ['pos_ppg']

        # Pseudo PPG
        psuedo_phys_labels = None
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            psuedo_phys_labels = self.generate_pos_ppg(frames, fps=self.config_data.FS)

        # Extract AU and emotion labels
        result = self.extract_face_labels(support_backend, openface_au_setting, emotion_source, 
                                          original_emotion_setting, include_dominance, libreface_dir, 
                                          openface_dir, subject_emotion, subject_vad, frame_count)

        au_occ_all, au_int_all, emotion_all, au_occ_colnames, au_int_colnames, emotion_colnames = result

        # Save column names once
        if self.dataset_name == "train" and i == 0:
            self.save_combined_label_names(ppg_colnames, au_occ_colnames, au_int_colnames, emotion_colnames)

        # Final preprocessing and save
        big_clips, small_clips, label_clips = self.preprocess(frames, phys_labels, psuedo_phys_labels, 
                                                              au_occ_all, au_int_all, emotion_all, config_preprocess)

        input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, label_clips, subject_number)
        file_list_dict[i] = input_name_list + [label_name_list]
    
    def extract_face_labels(self, backend, au_setting, emotion_source, emotion_setting, include_dominance, 
                            libreface_dir, openface_dir, subject_emotion, subject_vad, frame_count):
        """Extract AU and emotion labels based on backend and configuration."""

        au_occ_list, au_int_list = [], []
        au_occ_colnames, au_int_colnames, emotion_colnames = [], [], []

        # Load AU and emotion from LibreFace
        if backend in {'libreface', 'both'}:
            occ_lf, int_lf, emo_lf, occ_cols, int_cols, emo_cols = self.extract_labels_libreface(libreface_dir)
            au_occ_list.append(occ_lf)
            au_int_list.append(int_lf)
            au_occ_colnames += list(occ_cols)
            au_int_colnames += list(int_cols)

            if emotion_source == 'libreface':
                emotion_data = emo_lf
                emotion_colnames = list(emo_cols)

        # Load AU from OpenFace
        if backend in {'openface', 'both'}:
            openface_cfg = {'dynamic': [('au_dynamic', '')],
                            'dynamic_wild': [('au_dynamic_wild', '_wild')],
                            'static': [('au_static', '_static')],
                            'static_wild': [('au_static_wild', '_static_wild')]}

            if au_setting not in openface_cfg:
                raise ValueError(f"Unsupported OPENFACE_AU_SETTING: {au_setting}")

            occ_all, int_all, occ_cols, int_cols = [], [], [], []
            for subdir, suffix in openface_cfg[au_setting]:
                occ, intensity, *cols = self.extract_labels_openface(openface_dir, subdir, suffix)
                occ_all.append(occ)
                int_all.append(intensity)
                occ_cols += cols[0]
                int_cols += cols[1]

            au_occ_list.append(np.concatenate(occ_all, axis=1))
            au_int_list.append(np.concatenate(int_all, axis=1))
            au_occ_colnames += occ_cols
            au_int_colnames += int_cols

        # Ensure AUs were collected
        if not au_occ_list or not au_int_list:
            raise ValueError(f"No AU features loaded with backend '{backend}' and AU setting '{au_setting}'")

        # Combine AU features
        au_occ_all = np.concatenate(au_occ_list, axis=1)
        au_int_all = np.concatenate(au_int_list, axis=1)

        # Load emotion data from original if not libreface
        if emotion_source == 'original':
            if emotion_setting == 'discrete':
                emotion_data = subject_emotion
                emotion_colnames = ['amusement', 'disgust']
            elif emotion_setting == 'vad':
                emotion_data = subject_vad if include_dominance else subject_vad[:, :2]
                emotion_colnames = ['arousal', 'valence', 'dominance'] if include_dominance else ['arousal', 'valence']
            elif emotion_setting == 'both':
                vad_part = subject_vad if include_dominance else subject_vad[:, :2]
                vad_cols = ['arousal', 'valence', 'dominance'] if include_dominance else ['arousal', 'valence']
                emotion_data = np.concatenate([subject_emotion, vad_part], axis=1)
                emotion_colnames = ['amusement', 'disgust'] + vad_cols
            else:
                raise ValueError(f"Unsupported ORIGINAL_EMOTION_SETTING: {emotion_setting}")

        elif emotion_source != 'libreface':
            raise ValueError(f"Unsupported EMOTION_SOURCE: {emotion_source}")

        # Final validation
        for label_name, label_data in {'au_occ_all': au_occ_all, 'au_int_all': au_int_all, 'emotion_all': emotion_data}.items():
            if label_data.shape[0] != frame_count:
                raise ValueError(f"{label_name} shape mismatch: got {label_data.shape[0]}, expected {frame_count}")

        return au_occ_all, au_int_all, emotion_data, au_occ_colnames, au_int_colnames, emotion_colnames
    
    def save_combined_label_names(self, ppg_colnames, au_occ_colnames, au_int_colnames, emotion_colnames):
        """Saves combined column names for all labels used in training."""
        combined_colnames = pd.Index(ppg_colnames + au_occ_colnames + au_int_colnames + emotion_colnames)
        self.save_label_names(combined_colnames, os.path.dirname(self.raw_data_path))

    def read_video(self, video_file, config_preprocess):
        """
        Reads a video file and returns its frames as a NumPy array in RGB format.

        Args:
            video_file (str): Path to the video file.
            config_preprocess: Preprocessing configuration object.

        Returns:
            np.ndarray: Array of shape (T, H, W, 3) with RGB frames.
        """
        VidObj = cv2.VideoCapture(video_file)
        if not VidObj.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        height = int(VidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(VidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        square_size = min(height, width)
        
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)

        do_crop_face = config_preprocess.CROP_FACE.DO_CROP_FACE
        center_cropped_frames = []

        while True:
            success, frame = VidObj.read()
            if not success:
                break

            frame = self.center_crop_square(frame, height, width, square_size)
            if not do_crop_face:
                frame = self.resize(frame, downsample=True)

            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            center_cropped_frames.append(frame_rgb)

        VidObj.release()
        return np.array(center_cropped_frames)
    
    def preprocess_ppg(self, ppg_data, resample_target_length):
        """Applies filtering and resampling to PPG data, returns stacked result."""

        # Apply filters once
        filtered = self.apply_filters(ppg_data)

        # Resample both original and filtered data
        resampled = [self.resample_signal(signal, resample_target_length)
                     for signal in (ppg_data, filtered)]

        # Stack the resampled signals as columns
        return np.column_stack(resampled)
    
    @staticmethod
    def apply_filters(ppg_data):
        """Applies a bandpass Butterworth filter to the PPG data."""

        return filters.butter_filter(ppg_data, 
                                     low_cutoff=0.5,
                                     high_cutoff=4.0,
                                     sampling_rate=100,
                                     order=3,
                                     filter_type='band')
    
    def generate_pos_ppg(self, frames, fps=30):
        """
        Generates a normalized PPG signal using the POS_WANG method from the input video frames.

        Args:
            frames (np.ndarray): Video frames array of shape (T, H, W, C).
            fps (int): Frames per second of the video.

        Returns:
            np.ndarray: Normalized PPG signal.
        """
        # Generate the raw POS PPG signal
        pos_ppg = POS.POS_WANG(frames, fps)

        # TODO: Optional: Apply detrending but check first if there is a baseline drift

        # Apply bandpass filtering to isolate physiological frequencies
        filtered_ppg = self.apply_filters(pos_ppg)

        # Compute the analytic signal using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_ppg)

        # Compute the amplitude envelope (instantaneous magnitude)
        amplitude_envelope = np.abs(analytic_signal)

        # Avoid division by zero (or near-zero values)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_ppg = np.divide(filtered_ppg, amplitude_envelope)
            normalized_ppg[~np.isfinite(normalized_ppg)] = 0  # set inf or NaN to 0

        return normalized_ppg
        
    @staticmethod
    def extract_labels_libreface(aucoding_dir):
        # Locate the correct file
        for fname in ("full_output_processed.csv", "full_output.csv"):
            full_path = os.path.join(aucoding_dir, fname)
            if os.path.exists(full_path):
                break
        else:
            raise FileNotFoundError(f"No suitable CSV file found in {aucoding_dir}")

        # Define known AU columns and emotion
        au_occ_cols = [f'au_{i}' for i in [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]]
        au_int_cols = [f'au_{i}_intensity' for i in [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]]
        emotion_col = ['facial_expression']

        # Columns to load
        usecols = au_occ_cols + au_int_cols + emotion_col
        df = pd.read_csv(full_path, usecols=usecols)

        # Map emotion directly onto the DataFrame
        emotion_mapping = {"Neutral": 0,
                           "Happiness": 1,
                           "Sadness": 2,
                           "Surprise": 3,
                           "Fear": 4,
                           "Disgust": 5,
                           "Anger": 6,
                           "Contempt": 7}
        
        df['facial_expression'] = df['facial_expression'].map(emotion_mapping).fillna(-1).astype(int)

        # Extract column groups
        au_occ = df[au_occ_cols]
        au_int = df[au_int_cols]
        emotion = df[['facial_expression']]

        # Create renamed versions of columns
        au_occ_column_names = [f'AU{col[3:]}_lf' for col in au_occ_cols]
        au_int_column_names = [f'AU{col[3:-10]}_int_lf' for col in au_int_cols]
        emotion_column_name = ['emotion_lf']

        return (au_occ, au_int, emotion,
                au_occ_column_names,
                au_int_column_names,
                emotion_column_name)

    @staticmethod
    def extract_labels_openface(aucoding_dir, dir_name, colname):
        full_path = os.path.join(aucoding_dir, dir_name, "full_output.csv")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"CSV file not found at path: {full_path}")

        # Define AU column names
        au_intensity_cols = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
                             "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r",
                             "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

        au_occurrence_cols = ["AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c",
                              "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c", "AU20_c",
                              "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"]

        # Read only needed columns
        usecols = au_intensity_cols + au_occurrence_cols
        df = pd.read_csv(full_path, usecols=usecols)

        # Get slices
        au_int = df[au_intensity_cols]
        au_occ = df[au_occurrence_cols]

        # Generate renamed column names (don't change DataFrame)
        au_int_column_names = [col.replace('_r', f'_int_of{colname}') for col in au_intensity_cols]
        au_occ_column_names = [col.replace('_c', f'_of{colname}') for col in au_occurrence_cols]

        return au_occ, au_int, au_occ_column_names, au_int_column_names

    @staticmethod
    def save_label_names(data, label_list_path):
        # Ensure the output directory exists
        os.makedirs(label_list_path, exist_ok=True)

        full_path = os.path.join(label_list_path, "label_list.txt")

        # Efficiently write all labels at once
        with open(full_path, "w") as file:
            file.write('\n'.join(data) + '\n')

    def load_label_names(self, label_list_path):
        full_path = os.path.join(label_list_path, "label_list.txt")

        with open(full_path, "r") as file:
            label_list = [line.strip() for line in file]  # Read and strip spaces

        return label_list

    def preprocess(self, frames, phys_labels, psuedo_phys_labels, au_occ, au_int, emotion, config_preprocess):
        """Preprocesses frames and labels into big/small clip format with label transformation."""

        # --------------------------------------
        # 1. Helper functions
        # --------------------------------------
        def to_2d(arr):
            return arr.reshape(-1, 1) if arr is not None and arr.ndim == 1 else arr

        def generate_data(data_types):
            return np.concatenate([
                frames if dtype == "Raw"
                else self.diff_normalize_data(frames) if dtype == "DiffNormalized"
                else self.standardized_data(frames) if dtype == "Standardized"
                else (_ for _ in ()).throw(ValueError(f"Unsupported data type: {dtype}"))
                for dtype in data_types
            ], axis=-1)

        def transform_label(arr):
            if arr is None:
                return None
            if label_type == "Raw":
                return arr
            elif label_type == "DiffNormalized":
                return self.diff_normalize_label(arr)
            elif label_type == "Standardized":
                return self.standardized_label(arr)
            else:
                raise ValueError(f"Unsupported label type: {label_type}")

        # --------------------------------------
        # 2. Frame Preprocessing
        # --------------------------------------
        crop_cfg = config_preprocess.CROP_FACE
        resize_cfg = config_preprocess.BIGSMALL.RESIZE
        big_types = config_preprocess.BIGSMALL.BIG_DATA_TYPE
        small_types = config_preprocess.BIGSMALL.SMALL_DATA_TYPE

        # Crop and resize
        if crop_cfg.DO_CROP_FACE:
            frames = self.crop_face_resize(frames,
                                           crop_cfg.DO_CROP_FACE,
                                           crop_cfg.BACKEND,
                                           crop_cfg.USE_LARGE_FACE_BOX,
                                           crop_cfg.LARGE_BOX_COEF,
                                           crop_cfg.DETECTION.DO_DYNAMIC_DETECTION,
                                           crop_cfg.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           crop_cfg.DETECTION.USE_MEDIAN_FACE_BOX,
                                           resize_cfg.BIG_W,
                                           resize_cfg.BIG_H)

        # Generate big/small data paths
        big_data = generate_data(big_types)
        small_data = generate_data(small_types)

        # Resize small data
        small_data = self.crop_face_resize(small_data,
                                           crop_cfg.DO_CROP_FACE,
                                           crop_cfg.BACKEND,
                                           crop_cfg.USE_LARGE_FACE_BOX,
                                           crop_cfg.LARGE_BOX_COEF,
                                           crop_cfg.DETECTION.DO_DYNAMIC_DETECTION,
                                           crop_cfg.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
                                           crop_cfg.DETECTION.USE_MEDIAN_FACE_BOX,
                                           resize_cfg.SMALL_W,
                                           resize_cfg.SMALL_H)

        # --------------------------------------
        # 3. Label Preprocessing
        # --------------------------------------
        label_type = config_preprocess.LABEL_TYPE

        # Dynamically handle 1D or multi-column PPG
        phys_cols = phys_labels.shape[1] if phys_labels.ndim > 1 else 1
        ppg_components = []

        if phys_cols == 1:  # e.g., raw or filtered only
            ppg_single = transform_label(phys_labels)
            ppg_components.append(to_2d(ppg_single))
        else:  # both
            for col in range(phys_cols):
                transformed = transform_label(phys_labels[:, col])
                ppg_components.append(to_2d(transformed))

        # Optional pseudo PPG
        if psuedo_phys_labels is not None:
            ppg_components.append(to_2d(transform_label(psuedo_phys_labels)))

        # --------------------------------------
        # 4. Combine All Labels
        # --------------------------------------
        label_list = ppg_components + [to_2d(au_occ), to_2d(au_int), to_2d(emotion)]
        labels = np.concatenate([x for x in label_list if x is not None], axis=1)

        # --------------------------------------
        # 5. Chunking
        # --------------------------------------
        if config_preprocess.DO_CHUNK:
            chunk_len = config_preprocess.CHUNK_LENGTH
            big_clips, small_clips, label_clips = self.chunk(big_data, small_data, labels, chunk_len)
        else:
            big_clips = np.array([big_data])
            small_clips = np.array([small_data])
            label_clips = np.array([labels])

        return big_clips, small_clips, label_clips

    def chunk(self, big_frames, small_frames, labels, chunk_len):
        """
        Splits the input video frames and corresponding labels into fixed-length chunks.

        Args:
            big_frames (np.ndarray): Array of big video frames to be chunked.
            small_frames (np.ndarray): Array of small video frames to be chunked.
            labels (np.ndarray): Array of labels to be chunked.
            chunk_len (int): Length of each chunk.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - big_clips (np.ndarray): Chunks of big video frames.
                - small_clips (np.ndarray): Chunks of small video frames.
                - labels_clips (np.ndarray): Chunks of labels.
        """
        total_chunks = labels.shape[0] // chunk_len
        if total_chunks == 0:
            return np.empty((0, chunk_len, *big_frames.shape[1:])), \
                np.empty((0, chunk_len, *small_frames.shape[1:])), \
                np.empty((0, chunk_len, *labels.shape[1:]))

        big_clips = big_frames[:total_chunks * chunk_len].reshape(total_chunks, chunk_len, *big_frames.shape[1:])
        small_clips = small_frames[:total_chunks * chunk_len].reshape(total_chunks, chunk_len, *small_frames.shape[1:])
        labels_clips = labels[:total_chunks * chunk_len].reshape(total_chunks, chunk_len, *labels.shape[1:])

        return big_clips, small_clips, labels_clips

    @staticmethod
    def get_label_dtype(label_names):
        """
        Constructs a structured NumPy dtype based on label names and the following rules:

        - If the label contains 'AU' and does NOT contain 'int' → uint8 ('u1')
        - If the label is in ['emotion_lf', 'amusement', 'disgust', 'arousal', 'valence', 'dominance'] → uint8 ('u1')
        - Otherwise → float32 ('f4')

        Args:
            label_names (list of str): The label field names

        Returns:
            np.dtype: Structured dtype
        """
        emotion_set = {"emotion_lf", "amusement", "disgust", "arousal", "valence", "dominance"}

        dtype_fields = [
            (name, 'u1') if ("AU" in name and "int" not in name) or name in emotion_set else (name, 'f4')
            for name in label_names]

        return np.dtype(dtype_fields)

    def save_multi_process(self, big_clips, small_clips, label_clips, filename):
        """
        Saves big/small/label clips to .npy files using np.save (mmap-compatible).
        Loads label names from label_list.txt and constructs dtype inline.

        Args:
            big_clips (np.ndarray): Shape (num_clips, chunk_len, H, W, C) or similar.
            small_clips (np.ndarray): Same shape as big_clips, but smaller resolution.
            label_clips (np.ndarray): Shape (num_clips, chunk_len, num_labels).
            filename (str): Output file prefix.

        Returns:
            input_paths (list): [big_path, small_path]
            label_path (str): Path to saved label .npy file
        """
        # Load label names and create structured dtype
        label_names = self.load_label_names(os.path.dirname(self.raw_data_path))
        label_dtype = self.get_label_dtype(label_names)

        # Validate label shape
        if label_clips.ndim != 3 or label_clips.shape[2] != len(label_names):
            raise ValueError(f"label_clips must be 3D with shape [clips, chunk_len, {len(label_names)}]. Got: {label_clips.shape}")

        # Output paths
        cached_path = self.cached_path
        os.makedirs(cached_path, exist_ok=True)

        big_path = os.path.join(cached_path, f"{filename}_big.npy")
        small_path = os.path.join(cached_path, f"{filename}_small.npy")
        label_path = os.path.join(cached_path, f"{filename}_label.npy")

        # Save big and small clips as float32 (mmap-compatible)
        np.save(big_path, big_clips.astype(np.float32))
        np.save(small_path, small_clips.astype(np.float32))

        # Reshape and convert labels to structured array
        reshaped_labels = label_clips.reshape(-1, label_clips.shape[2])
        structured_labels = np.core.records.fromarrays(reshaped_labels.T, dtype=label_dtype)

        # Save structured label array
        np.save(label_path, structured_labels)

        return [big_path, small_path], label_path

    def load_preprocessed_data_original(self):
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

    def load_preprocessed_data(self):
        """
        Efficiently loads preprocessed file paths from a CSV file with columns: 'big', 'small', 'label'.
        Prepends cached path to filenames and stores:
            - self.inputs: list of (big_path, small_path) tuples
            - self.labels: list of label paths
            - self.preprocessed_data_len: total number of samples

        Raises:
            ValueError: If required columns are missing, data is empty, or contains NaNs.
            FileNotFoundError: If the CSV file itself is not found.
        """
        required_cols = ['big', 'small', 'label']
        cached_path = self.cached_path

        try:
            file_list_df = pd.read_csv(self.file_list_path, usecols=required_cols)
        except ValueError as e:
            raise ValueError(f"{self.dataset_name}: CSV missing required columns {required_cols}.") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.dataset_name}: CSV file not found at {self.file_list_path}")

        if file_list_df.empty:
            raise ValueError(f"{self.dataset_name}: CSV file is empty.")

        if file_list_df.isnull().values.any():
            raise ValueError(f"{self.dataset_name}: CSV contains NaNs in required columns.")

        # Use NumPy arrays for fast iteration
        big_files = file_list_df['big'].values
        small_files = file_list_df['small'].values
        label_files = file_list_df['label'].values

        # Efficient list comprehension using os.path.basename and join
        big_paths = [os.path.join(cached_path, os.path.basename(p)) for p in big_files]
        small_paths = [os.path.join(cached_path, os.path.basename(p)) for p in small_files]
        label_paths = [os.path.join(cached_path, os.path.basename(p)) for p in label_files]

        self.inputs = list(zip(big_paths, small_paths))
        self.labels = label_paths
        self.preprocessed_data_len = len(self.inputs)

        if self.preprocessed_data_len == 0:
            raise ValueError(f"{self.dataset_name}: No valid file paths loaded (Length is 0 after processing).")

        label_names = self.load_label_names(os.path.dirname(self.raw_data_path))
        self.discrete_emotion_exist = 'emotion_lf' in label_names

    def original_getitem(self, index):
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

    def load_format_frames(self, path, data_format):
        """
        Loads a memory-mapped .npy file and formats it according to the given format.
        
        Supported formats:
            - 'NDCHW': Transpose from (N, H, W, C) to (N, C, H, W)
            - 'NCDHW': Transpose from (D, H, W, C) to (C, D, H, W)
        """
        frames = np.load(path)

        if data_format == 'NDCHW':
            frames = np.transpose(frames, (0, 3, 1, 2))
        elif data_format == 'NCDHW':
            frames = np.transpose(frames, (3, 0, 1, 2))
        elif data_format != 'NHWC':
            raise ValueError(f"Unsupported data_format: {data_format}")

        return frames.astype(np.float32, copy=False)

    def __getitem__(self, index):
        """
        Loads one sample (big, small) and combines all label fields into a single float32 array.

        Returns:
            data (list): [big_frames, small_frames]
            labels (np.ndarray): All labels as float32, shape (T, C)
            filename (str): Subject/session ID (e.g., 'subject01')
            chunk_id (str): Number after 'big' in filename (e.g., '13')
        """
        big_path, small_path = self.inputs[index]
        label_path = self.labels[index]

        # Load and format video frames
        big_frames = self.load_format_frames(big_path, self.data_format)
        small_frames = self.load_format_frames(small_path, self.data_format)

        # Load and convert all label fields to float32
        labels = np.load(label_path).astype(np.float32, copy=False)

        # Extract subject ID and chunk number from big_path filename
        base_filename = os.path.splitext(os.path.basename(big_path))[0]
        filename, _, chunk_str = base_filename.partition('_')
        chunk_id = chunk_str[3:] if chunk_str.startswith('big') else "0"

        return [big_frames, small_frames], labels, filename, chunk_id

    def build_file_list(self, file_list_dict):
        """
        Builds and saves a structured CSV file listing paths to 'big', 'small', and 'label' files.

        Args:
            file_list_dict (dict): Mapping from process ID to a list of [big_path, small_path, label_path].

        Raises:
            ValueError: If any entry is malformed or if the final list is empty.
        """
        # Validate and collect all valid entries
        file_rows = [paths for pid, paths in file_list_dict.items()
                     if isinstance(paths, list) and len(paths) == 3]

        # Check for invalid entries
        if len(file_rows) != len(file_list_dict):
            invalid_pids = [pid for pid, paths in file_list_dict.items() 
                            if not isinstance(paths, list) or len(paths) != 3]
            raise ValueError(f"{self.dataset_name}: Invalid file entries for processes: {invalid_pids}")

        if not file_rows:
            raise ValueError(f"{self.dataset_name}: No valid files found in file list.")

        # Save to CSV
        df = pd.DataFrame(file_rows, columns=['big', 'small', 'label'])
        os.makedirs(os.path.dirname(self.file_list_path) or '.', exist_ok=True)
        df.to_csv(self.file_list_path, index=False)

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
        subject_number = data_dirs[i]['index']

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
        input_name_list, label_name_list = self.save_multi_process(big_clips, small_clips, label_clips, subject_number)
        file_list_dict[i] = input_name_list

    def read_video(self, video_file, config_preprocess):
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
                big_data.append(self.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                big_data.append(self.standardized_data(f_c)) # Apply standardization
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
                small_data.append(self.diff_normalize_data(f_c)) # Apply difference normalization
            elif data_type == "Standardized":
                small_data.append(self.standardized_data(f_c)) # Apply standardization
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
            phys_labels = self.diff_normalize_label(phys_labels)
            psuedo_phys_labels = self.diff_normalize_label(psuedo_phys_labels)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            phys_labels = self.standardized_label(phys_labels)
            psuedo_phys_labels = self.standardized_label(psuedo_phys_labels)
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