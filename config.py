import os, re
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------\
_C.TOOLBOX_MODE = ""
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.LR = 1e-4
_C.TRAIN.OPTIMIZER = CN() # Optimizer
_C.TRAIN.OPTIMIZER.EPS = 1e-4 # Optimizer Epsilon
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999) # Optimizer Betas

# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.MODEL_FILE_NAME = ''
_C.TRAIN.PLOT_LOSSES_AND_LR = True

# Train data settings
_C.TRAIN.DATA = CN()
_C.TRAIN.DATA.INFO = CN()
_C.TRAIN.DATA.INFO.LIGHT = ['']
_C.TRAIN.DATA.INFO.MOTION = ['']
_C.TRAIN.DATA.INFO.EXERCISE = [True]
_C.TRAIN.DATA.INFO.SKIN_COLOR = [1]
_C.TRAIN.DATA.INFO.GENDER = ['']
_C.TRAIN.DATA.INFO.GLASSER = [True]
_C.TRAIN.DATA.INFO.HAIR_COVER = [True]
_C.TRAIN.DATA.INFO.MAKEUP = [True]
_C.TRAIN.DATA.FILTERING = CN()
_C.TRAIN.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.TRAIN.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.TRAIN.DATA.FILTERING.SELECT_TASKS = False
_C.TRAIN.DATA.FILTERING.TASK_LIST = ['']
_C.TRAIN.DATA.FS = 0
_C.TRAIN.DATA.DATA_PATH = ''
_C.TRAIN.DATA.EXP_DATA_NAME = ''
_C.TRAIN.DATA.CACHED_PATH = 'PreprocessedData'
_C.TRAIN.DATA.FILE_LIST_PATH = os.path.join(_C.TRAIN.DATA.CACHED_PATH, 'DataFileLists')
_C.TRAIN.DATA.DATASET = ''
_C.TRAIN.DATA.DO_PREPROCESS = False
_C.TRAIN.DATA.DATA_FORMAT = 'NDCHW'
_C.TRAIN.DATA.BEGIN = 0.0
_C.TRAIN.DATA.END = 1.0
_C.TRAIN.DATA.FOLD = CN()
_C.TRAIN.DATA.FOLD.FOLD_NAME = ''
_C.TRAIN.DATA.FOLD.FOLD_PATH = ''

# Train data preprocessing
_C.TRAIN.DATA.PREPROCESS = CN()
_C.TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.TRAIN.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.DATA_AUG = ['None']
_C.TRAIN.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TRAIN.DATA.PREPROCESS.DO_CHUNK = True
_C.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TRAIN.DATA.PREPROCESS.CROP_FACE = CN()
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.TRAIN.DATA.PREPROCESS.RESIZE = CN()
_C.TRAIN.DATA.PREPROCESS.RESIZE.W = 128
_C.TRAIN.DATA.PREPROCESS.RESIZE.H = 128
_C.TRAIN.DATA.PREPROCESS.BIGSMALL = CN()
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.TRAIN.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9
_C.TRAIN.DATA.PREPROCESS.IBVP = CN()
_C.TRAIN.DATA.PREPROCESS.IBVP.DATA_MODE = 'RGB'

# -----------------------------------------------------------------------------
# Valid settings
# -----------------------------------------------------------------------------\
_C.VALID = CN()
# Valid data settings
_C.VALID.DATA = CN()
_C.VALID.DATA.INFO = CN()
_C.VALID.DATA.INFO.LIGHT = ['']
_C.VALID.DATA.INFO.MOTION = ['']
_C.VALID.DATA.INFO.EXERCISE = [True]
_C.VALID.DATA.INFO.SKIN_COLOR = [1]
_C.VALID.DATA.INFO.GENDER = ['']
_C.VALID.DATA.INFO.GLASSER = [True]
_C.VALID.DATA.INFO.HAIR_COVER = [True]
_C.VALID.DATA.INFO.MAKEUP = [True]
_C.VALID.DATA.FILTERING = CN()
_C.VALID.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.VALID.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.VALID.DATA.FILTERING.SELECT_TASKS = False
_C.VALID.DATA.FILTERING.TASK_LIST = ['']
_C.VALID.DATA.FS = 0
_C.VALID.DATA.DATA_PATH = ''
_C.VALID.DATA.EXP_DATA_NAME = ''
_C.VALID.DATA.CACHED_PATH = 'PreprocessedData'
_C.VALID.DATA.FILE_LIST_PATH = os.path.join(_C.VALID.DATA.CACHED_PATH, 'DataFileLists')
_C.VALID.DATA.DATASET = ''
_C.VALID.DATA.DO_PREPROCESS = False
_C.VALID.DATA.DATA_FORMAT = 'NDCHW'
_C.VALID.DATA.BEGIN = 0.0
_C.VALID.DATA.END = 1.0
_C.VALID.DATA.FOLD = CN()
_C.VALID.DATA.FOLD.FOLD_NAME = ''
_C.VALID.DATA.FOLD.FOLD_PATH = ''

# Valid data preprocessing
_C.VALID.RUN_VALIDATION = True
_C.VALID.DATA.PREPROCESS = CN()
_C.VALID.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.VALID.DATA.PREPROCESS.DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.DATA_AUG = ['None']
_C.VALID.DATA.PREPROCESS.LABEL_TYPE = ''
_C.VALID.DATA.PREPROCESS.DO_CHUNK = True
_C.VALID.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.VALID.DATA.PREPROCESS.CROP_FACE = CN()
_C.VALID.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.VALID.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.VALID.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.VALID.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.VALID.DATA.PREPROCESS.RESIZE = CN()
_C.VALID.DATA.PREPROCESS.RESIZE.W = 128
_C.VALID.DATA.PREPROCESS.RESIZE.H = 128
_C.VALID.DATA.PREPROCESS.BIGSMALL = CN()
_C.VALID.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.VALID.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9
_C.VALID.DATA.PREPROCESS.IBVP = CN()
_C.VALID.DATA.PREPROCESS.IBVP.DATA_MODE = 'RGB'

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------\
_C.TEST = CN()
_C.TEST.OUTPUT_SAVE_DIR = ''
_C.TEST.METRICS = []
_C.TEST.USE_BEST_EPOCH = True

# Test data settings
_C.TEST.DATA = CN()
_C.TEST.DATA.INFO = CN()
_C.TEST.DATA.INFO.LIGHT = ['']
_C.TEST.DATA.INFO.MOTION = ['']
_C.TEST.DATA.INFO.EXERCISE = [True]
_C.TEST.DATA.INFO.SKIN_COLOR = [1]
_C.TEST.DATA.INFO.GENDER = ['']
_C.TEST.DATA.INFO.GLASSER = [True]
_C.TEST.DATA.INFO.HAIR_COVER = [True]
_C.TEST.DATA.INFO.MAKEUP = [True]
_C.TEST.DATA.FILTERING = CN()
_C.TEST.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.TEST.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.TEST.DATA.FILTERING.SELECT_TASKS = False
_C.TEST.DATA.FILTERING.TASK_LIST = ['']
_C.TEST.DATA.FS = 0
_C.TEST.DATA.DATA_PATH = ''
_C.TEST.DATA.EXP_DATA_NAME = ''
_C.TEST.DATA.CACHED_PATH = 'PreprocessedData'
_C.TEST.DATA.FILE_LIST_PATH = os.path.join(_C.TEST.DATA.CACHED_PATH, 'DataFileLists')
_C.TEST.DATA.DATASET = ''
_C.TEST.DATA.DO_PREPROCESS = False
_C.TEST.DATA.DATA_FORMAT = 'NDCHW'
_C.TEST.DATA.BEGIN = 0.0
_C.TEST.DATA.END = 1.0
_C.TEST.DATA.FOLD = CN()
_C.TEST.DATA.FOLD.FOLD_NAME = ''
_C.TEST.DATA.FOLD.FOLD_PATH = ''

# Test data preprocessing
_C.TEST.DATA.PREPROCESS = CN()
_C.TEST.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.TEST.DATA.PREPROCESS.DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.DATA_AUG = ['None']
_C.TEST.DATA.PREPROCESS.LABEL_TYPE = ''
_C.TEST.DATA.PREPROCESS.DO_CHUNK = True
_C.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.TEST.DATA.PREPROCESS.CROP_FACE = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.TEST.DATA.PREPROCESS.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.RESIZE.W = 128
_C.TEST.DATA.PREPROCESS.RESIZE.H = 128
_C.TEST.DATA.PREPROCESS.BIGSMALL = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.BIG_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.SMALL_DATA_TYPE = ['']
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE = CN()
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_W = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.BIG_H = 144
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_W = 9
_C.TEST.DATA.PREPROCESS.BIGSMALL.RESIZE.SMALL_H = 9
_C.TEST.DATA.PREPROCESS.IBVP = CN()
_C.TEST.DATA.PREPROCESS.IBVP.DATA_MODE = 'RGB'

# -----------------------------------------------------------------------------
# Unsupervised method settings
# -----------------------------------------------------------------------------\
_C.UNSUPERVISED = CN()
_C.UNSUPERVISED.METHOD = []
_C.UNSUPERVISED.OUTPUT_SAVE_DIR = ''
_C.UNSUPERVISED.METRICS = []

# Unsupervised data settings
_C.UNSUPERVISED.DATA = CN()
_C.UNSUPERVISED.DATA.INFO = CN()
_C.UNSUPERVISED.DATA.INFO.LIGHT = ['']
_C.UNSUPERVISED.DATA.INFO.MOTION = ['']
_C.UNSUPERVISED.DATA.INFO.EXERCISE = [True]
_C.UNSUPERVISED.DATA.INFO.SKIN_COLOR = [1]
_C.UNSUPERVISED.DATA.INFO.GENDER = ['']
_C.UNSUPERVISED.DATA.INFO.GLASSER = [True]
_C.UNSUPERVISED.DATA.INFO.HAIR_COVER = [True]
_C.UNSUPERVISED.DATA.INFO.MAKEUP = [True]
_C.UNSUPERVISED.DATA.FILTERING = CN()
_C.UNSUPERVISED.DATA.FILTERING.USE_EXCLUSION_LIST = False
_C.UNSUPERVISED.DATA.FILTERING.EXCLUSION_LIST = ['']
_C.UNSUPERVISED.DATA.FILTERING.SELECT_TASKS = False
_C.UNSUPERVISED.DATA.FILTERING.TASK_LIST = ['']
_C.UNSUPERVISED.DATA.FS = 0
_C.UNSUPERVISED.DATA.DATA_PATH = ''
_C.UNSUPERVISED.DATA.EXP_DATA_NAME = ''
_C.UNSUPERVISED.DATA.CACHED_PATH = 'PreprocessedData'
_C.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(_C.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')
_C.UNSUPERVISED.DATA.DATASET = ''
_C.UNSUPERVISED.DATA.DO_PREPROCESS = False
_C.UNSUPERVISED.DATA.DATA_FORMAT = 'NDCHW'
_C.UNSUPERVISED.DATA.BEGIN = 0.0
_C.UNSUPERVISED.DATA.END = 1.0
_C.UNSUPERVISED.DATA.FOLD = CN()
_C.UNSUPERVISED.DATA.FOLD.FOLD_NAME = ''
_C.UNSUPERVISED.DATA.FOLD.FOLD_PATH = ''

# Unsupervised data preprocessing
_C.UNSUPERVISED.DATA.PREPROCESS = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL = False
_C.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE = ['']
_C.UNSUPERVISED.DATA.PREPROCESS.DATA_AUG = ['None']
_C.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE = ''
_C.UNSUPERVISED.DATA.PREPROCESS.DO_CHUNK = True
_C.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = False
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
_C.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = False
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W = 128
_C.UNSUPERVISED.DATA.PREPROCESS.RESIZE.H = 128
_C.UNSUPERVISED.DATA.PREPROCESS.IBVP = CN()
_C.UNSUPERVISED.DATA.PREPROCESS.IBVP.DATA_MODE = 'RGB'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------\
_C.MODEL = CN()
_C.MODEL.NAME = '' # Model name
_C.MODEL.RESUME = '' # Checkpoint to resume, could be overwritten by command line argument

# Dropout rate
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.MODEL_DIR = 'PreTrainedModels'

# Model Settings for BigSmall
_C.MODEL.BIGSMALL = CN()
_C.MODEL.BIGSMALL.FRAME_DEPTH = 3

# Model Settings for PHYSNET
_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 64

# Model Settings for TS-CAN
_C.MODEL.TSCAN = CN()
_C.MODEL.TSCAN.FRAME_DEPTH = 10

# Model Settings for EfficientPhys
_C.MODEL.EFFICIENTPHYS = CN()
_C.MODEL.EFFICIENTPHYS.FRAME_DEPTH = 10

# Model Settings for PhysFormer
_C.MODEL.PHYSFORMER = CN()
_C.MODEL.PHYSFORMER.PATCH_SIZE = 4
_C.MODEL.PHYSFORMER.DIM = 96
_C.MODEL.PHYSFORMER.FF_DIM = 144
_C.MODEL.PHYSFORMER.NUM_HEADS = 4
_C.MODEL.PHYSFORMER.NUM_LAYERS = 12
_C.MODEL.PHYSFORMER.THETA = 0.7

# Specific parameters for iBVPNet parameters
_C.MODEL.iBVPNet = CN()
_C.MODEL.iBVPNet.FRAME_NUM = 160
_C.MODEL.iBVPNet.CHANNELS = 3

# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.EVALUATION_METHOD = 'FFT'
_C.INFERENCE.EVALUATION_WINDOW = CN()
_C.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
_C.INFERENCE.MODEL_PATH = ''

# -----------------------------------------------------------------------------
# Device settings
# -----------------------------------------------------------------------------
_C.DEVICE = "cuda:0"
_C.NUM_OF_GPU_TRAIN = 1

# -----------------------------------------------------------------------------
# Log settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"

def get_config(args):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

def update_config(config, args):
    # Store default file list path for checking against later
    default_TRAIN_FILE_LIST_PATH = config.TRAIN.DATA.FILE_LIST_PATH
    default_VALID_FILE_LIST_PATH = config.VALID.DATA.FILE_LIST_PATH
    default_TEST_FILE_LIST_PATH = config.TEST.DATA.FILE_LIST_PATH
    default_UNSUPERVISED_FILE_LIST_PATH = config.UNSUPERVISED.DATA.FILE_LIST_PATH

    _update_config_from_file(config, args.config_file)
    config.defrost()

     # UPDATE TRAIN PATHS
    if config.TRAIN.DATA.FILE_LIST_PATH == default_TRAIN_FILE_LIST_PATH:
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, 'DataFileLists')
    
    # Check if the experiment data name is not set
    if config.TRAIN.DATA.EXP_DATA_NAME == '':
        # Construct a unique experiment data name by joining various configuration parameters
        config.TRAIN.DATA.EXP_DATA_NAME = "_".join(["TRAINING",
                                                    config.TRAIN.DATA.DATASET,                                     
                                                    "SizeW{0}".format(str(config.TRAIN.DATA.PREPROCESS.RESIZE.W)), 
                                                    "SizeH{0}".format(str(config.TRAIN.DATA.PREPROCESS.RESIZE.W)), 
                                                    "ClipLength{0}".format(str(config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)), 
                                                    "DataType{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_TYPE)),
                                                    "DataAug{0}".format("_".join(config.TRAIN.DATA.PREPROCESS.DATA_AUG)),
                                                    "LabelType{0}".format(config.TRAIN.DATA.PREPROCESS.LABEL_TYPE),
                                                    "CropFace{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                                    "Backend{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                                    "LargeBox{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                                    "LargeSize{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                                    "DynamicDet{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                                    "DetLength{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                                    "MedianFaceBox{0}".format(config.TRAIN.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                                   ])
    
    # Update the cached path to include the experiment data name
    config.TRAIN.DATA.CACHED_PATH = os.path.join(config.TRAIN.DATA.CACHED_PATH, config.TRAIN.DATA.EXP_DATA_NAME)

     # Extract the file name and extension from the TRAIN.DATA.FILE_LIST_PATH
    name, ext = os.path.splitext(config.TRAIN.DATA.FILE_LIST_PATH)
    
    # Check if the path does not have a file extension
    if not ext: 
        # Construct a fold string if a fold name is provided, otherwise use an empty string
        FOLD_STR = '_' + config.TRAIN.DATA.FOLD.FOLD_NAME if config.TRAIN.DATA.FOLD.FOLD_NAME else ''
        # Update the FILE_LIST_PATH to include a detailed file name
        config.TRAIN.DATA.FILE_LIST_PATH = os.path.join(config.TRAIN.DATA.FILE_LIST_PATH, \
                                                        config.TRAIN.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.TRAIN.DATA.BEGIN) + '_' + \
                                                        str(config.TRAIN.DATA.END) + \
                                                        FOLD_STR + '.csv')
    elif ext != '.csv':
        raise ValueError('TRAIN dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    # Check if the file extension is '.csv' and preprocessing is enabled for the training data
    if ext == '.csv' and config.TRAIN.DATA.DO_PREPROCESS:
        # Raise an error if both conditions are true, indicating a conflict
        raise ValueError('User specified TRAIN dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TRAIN dataset FILE_LIST_PATH .csv file.')
    
    # Check if the configuration is set to run the validation and if a validation dataset is specified.
    if config.VALID.RUN_VALIDATION and config.VALID.DATA.DATASET is not None:
        # If the current validation data file list path is the default path,
        # Update it to point to the 'DataFileLists' directory within the cached path.
        if config.VALID.DATA.FILE_LIST_PATH == default_VALID_FILE_LIST_PATH:
            config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, 'DataFileLists')

        # Check if the experiment data name for validation is not set
        if config.VALID.DATA.EXP_DATA_NAME == '':
            # Construct a new experiment data name by joining various configuration parameters
            # These parameters describe the preprocessing settings and characteristics of the dataset
            config.VALID.DATA.EXP_DATA_NAME = "_".join(["VALIDATION",
                                                        config.VALID.DATA.DATASET, 
                                                        "SizeW{0}".format(str(config.VALID.DATA.PREPROCESS.RESIZE.W)),
                                                        "SizeH{0}".format(str(config.VALID.DATA.PREPROCESS.RESIZE.W)), 
                                                        "ClipLength{0}".format(str(config.VALID.DATA.PREPROCESS.CHUNK_LENGTH)), 
                                                        "DataType{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_TYPE)),
                                                        "DataAug{0}".format("_".join(config.VALID.DATA.PREPROCESS.DATA_AUG)),
                                                        "LabelType{0}".format(config.VALID.DATA.PREPROCESS.LABEL_TYPE),
                                                        "CropFace{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                                        "Backend{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                                        "LargeBox{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                                        "LargeSize{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                                        "DynamicDet{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                                        "DetLength{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                                        "MedianFaceBox{0}".format(config.VALID.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                                       ])
        
        # Update the cached path for validation data by appending the experiment data name
        # This organizes the cached data into a directory structure reflecting the experiment configuration
        config.VALID.DATA.CACHED_PATH = os.path.join(config.VALID.DATA.CACHED_PATH, config.VALID.DATA.EXP_DATA_NAME)

        # Extract the file name and extension from the VALID.DATA.FILE_LIST_PATH
        name, ext = os.path.splitext(config.VALID.DATA.FILE_LIST_PATH)

        # Check if the path does not have a file extension
        if not ext: 
            # Construct a fold string if a fold name is provided, otherwise use an empty string
            FOLD_STR = '_' + config.VALID.DATA.FOLD.FOLD_NAME if config.VALID.DATA.FOLD.FOLD_NAME else ''

            # Update the FILE_LIST_PATH to include a detailed file name
            config.VALID.DATA.FILE_LIST_PATH = os.path.join(config.VALID.DATA.FILE_LIST_PATH, \
                                                            config.VALID.DATA.EXP_DATA_NAME + '_' + \
                                                            str(config.VALID.DATA.BEGIN) + '_' + \
                                                            str(config.VALID.DATA.END) + \
                                                            FOLD_STR + '.csv')
        # Check if the file extension is not '.csv'
        elif ext != '.csv':
            # Raise an error if the file extension is not '.csv'
            raise ValueError('VALIDATION dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

        if ext == '.csv' and config.VALID.DATA.DO_PREPROCESS:
            raise ValueError('User specified VALIDATION dataset FILE_LIST_PATH .csv file already exists. \
                            Please turn DO_PREPROCESS to False or delete existing VALIDATION dataset FILE_LIST_PATH .csv file.')
    
    # Check if the configuration is set to run the validation and if a validation dataset is not provided.
    elif config.VALID.RUN_VALIDATION and config.VALID.DATA.DATASET is None:
        # Raise a ValueError if both conditions are true, indicating a configuration issue.
        # The error message suggests that a validation dataset must be provided when
        # RUN_VALIDATION is set to True, as the configuration expects a specific dataset
        # for testing instead of relying on the last epoch's results.
        raise ValueError('VALIDATION dataset is not provided despite RUN_VALIDATION is being set to True!')

    # UPDATE TEST PATHS
    # Check if the test data FILE_LIST_PATH is set to its default value.
    # If so, update it to point to the 'DataFileLists' directory within the CACHED_PATH.
    if config.TEST.DATA.FILE_LIST_PATH == default_TEST_FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, 'DataFileLists')

    # If the EXP_DATA_NAME for the test data is not set, generate a name based on various preprocessing settings.
    # This name includes dataset characteristics like size, chunk length, data type, augmentation, and face cropping settings.
    if config.TEST.DATA.EXP_DATA_NAME == '':
        config.TEST.DATA.EXP_DATA_NAME = "_".join(["TESTING",
                                                   config.TEST.DATA.DATASET, 
                                                   "SizeW{0}".format(str(config.TEST.DATA.PREPROCESS.RESIZE.W)), 
                                                   "SizeH{0}".format(str(config.TEST.DATA.PREPROCESS.RESIZE.H)), 
                                                   "ClipLength{0}".format(str(config.TEST.DATA.PREPROCESS.CHUNK_LENGTH)), 
                                                   "DataType{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_TYPE)),
                                                   "DataAug{0}".format("_".join(config.TEST.DATA.PREPROCESS.DATA_AUG)),
                                                   "LabelType{0}".format(config.TEST.DATA.PREPROCESS.LABEL_TYPE),
                                                   "CropFace{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                                   "Backend{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                                   "LargeBox{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                                   "LargeSize{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                                   "DynamicDet{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                                   "DetLength{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                                   "MedianFaceBox{0}".format(config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX)
                                                  ])
        
    # Update the CACHED_PATH to include the EXP_DATA_NAME, organizing cached data by experiment configuration.
    config.TEST.DATA.CACHED_PATH = os.path.join(config.TEST.DATA.CACHED_PATH, config.TEST.DATA.EXP_DATA_NAME)

    # Extract the file name and extension from the FILE_LIST_PATH.
    name, ext = os.path.splitext(config.TEST.DATA.FILE_LIST_PATH)
    # If no file extension is present, append a detailed file name with a .csv extension.
    # This includes the experiment data name, data range, and fold name if provided.
    if not ext: # no file extension
        FOLD_STR = '_' + config.TEST.DATA.FOLD.FOLD_NAME if config.TEST.DATA.FOLD.FOLD_NAME else ''
        config.TEST.DATA.FILE_LIST_PATH = os.path.join(config.TEST.DATA.FILE_LIST_PATH, \
                                                       config.TEST.DATA.EXP_DATA_NAME + '_' + \
                                                       str(config.TEST.DATA.BEGIN) + '_' + \
                                                       str(config.TEST.DATA.END) + \
                                                       FOLD_STR + '.csv')
    # Raise an error if the file extension is not .csv, as the path must be a directory or a .csv file.
    elif ext != '.csv':
        raise ValueError('TEST dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    # If the FILE_LIST_PATH is a .csv file and preprocessing is enabled, raise an error.
    # This prevents overwriting an existing .csv file with preprocessing results.
    if ext == '.csv' and config.TEST.DATA.DO_PREPROCESS:
        raise ValueError('User specified TEST dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing TEST dataset FILE_LIST_PATH .csv file.')

    # Update MODEL_FILE_NAME if any data augmentation is applied
    if any(aug != 'None' for aug in config.TRAIN.DATA.PREPROCESS.DATA_AUG + config.VALID.DATA.PREPROCESS.DATA_AUG + config.TEST.DATA.PREPROCESS.DATA_AUG):
        # Check if the MODEL_FILE_NAME follows the expected pattern [TRAIN_SET]_[VALID_SET]_[TEST_SET]
        if re.match(r'^[^_]+(_[^_]+)?(_[^_]+)?_[^_]+$', config.TRAIN.MODEL_FILE_NAME):
            # Split the MODEL_FILE_NAME into parts using underscores
            model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')

            # Determine the indices for train, valid, and test dataset names
            if model_file_name_parts[2] == config.TEST.DATA.DATASET:
                train_name_idx = 0
                valid_name_idx = 1
                test_name_idx = 2
            else:
                train_name_idx = 0
                valid_name_idx = None
                test_name_idx = 1
            
            # Prefix 'MA-' to the train dataset part if 'Motion' augmentation is applied
            if 'Motion' in config.TRAIN.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[train_name_idx] = 'MA-' + model_file_name_parts[train_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)
            
            # Prefix 'MA-' to the valid dataset part if 'Motion' augmentation is applied
            if 'Motion' in config.VALID.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[valid_name_idx] = 'MA-' + model_file_name_parts[valid_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)

            # Prefix 'MA-' to the test dataset part if 'Motion' augmentation is applied
            if 'Motion' in config.TEST.DATA.PREPROCESS.DATA_AUG:
                model_file_name_parts = config.TRAIN.MODEL_FILE_NAME.split('_')
                model_file_name_parts[test_name_idx] = 'MA-' + model_file_name_parts[test_name_idx]
                config.TRAIN.MODEL_FILE_NAME = '_'.join(model_file_name_parts)
        else:
            # Raise an error if the MODEL_FILE_NAME does not follow the expected pattern
            raise ValueError(f'MODEL_FILE_NAME does not follow expected naming pattern of [TRAIN_SET]_[VALID_SET]_[TEST_SET]! \
                             \nReceived {config.TRAIN.MODEL_FILE_NAME}.')

    # Ensure pseudo PPG labels are not used for unsupervised methods, as they are not supported for unsupervised methods.
    if config.TOOLBOX_MODE == 'unsupervised_method' and config.UNSUPERVISED.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL == True:
        raise ValueError('Pseudo PPG labels are NOT supported for unsupervised methods.')

    # Update the file list path for unsupervised data if it is set to the default value
    if config.UNSUPERVISED.DATA.FILE_LIST_PATH == default_UNSUPERVISED_FILE_LIST_PATH:
        # Set the file list path to point to the 'DataFileLists' directory within the cached path
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, 'DataFileLists')

    # Generate an experiment data name if it is not already set
    if config.UNSUPERVISED.DATA.EXP_DATA_NAME == '':
        # Construct the experiment data name using various preprocessing settings
        config.UNSUPERVISED.DATA.EXP_DATA_NAME = "_".join(["UNSUPERVISED",
                                                           config.UNSUPERVISED.DATA.DATASET, 
                                                           "SizeW{0}".format(str(config.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W)), 
                                                           "SizeH{0}".format(str(config.UNSUPERVISED.DATA.PREPROCESS.RESIZE.W)), 
                                                           "ClipLength{0}".format(str(config.UNSUPERVISED.DATA.PREPROCESS.CHUNK_LENGTH)),
                                                           "DataType{0}".format("_".join(config.UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE)),
                                                           "DataAug{0}".format("_".join(config.UNSUPERVISED.DATA.PREPROCESS.DATA_AUG)),
                                                           "LabelType{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE),
                                                           "CropFace{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE),
                                                           "Backend{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.BACKEND),
                                                           "LargeBox{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX),
                                                           "LargeSize{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF),
                                                           "DynamicDet{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION),
                                                           "DetLength{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY),
                                                           "MedianFaceBox{0}".format(config.UNSUPERVISED.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX),
                                                           ])
        
    # Update the cached path to include the experiment data name
    config.UNSUPERVISED.DATA.CACHED_PATH = os.path.join(config.UNSUPERVISED.DATA.CACHED_PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME)

    # Determine the file list path based on the presence of a file extension
    name, ext = os.path.splitext(config.UNSUPERVISED.DATA.FILE_LIST_PATH)
    if not ext: # If no file extension is present
        # Construct a fold string if a fold name is provided, otherwise use an empty string
        FOLD_STR = '_' + config.UNSUPERVISED.DATA.FOLD.FOLD_NAME if config.UNSUPERVISED.DATA.FOLD.FOLD_NAME else ''
        # Update the file list path to include a detailed file name with a .csv extension
        config.UNSUPERVISED.DATA.FILE_LIST_PATH = os.path.join(config.UNSUPERVISED.DATA.FILE_LIST_PATH, \
                                                        config.UNSUPERVISED.DATA.EXP_DATA_NAME + '_' + \
                                                        str(config.UNSUPERVISED.DATA.BEGIN) + '_' + \
                                                        str(config.UNSUPERVISED.DATA.END) + \
                                                        FOLD_STR + '.csv')
    elif ext != '.csv': # If the file extension is not .csv
        # Raise an error as the path must be a directory or a .csv file
        raise ValueError('UNSUPERVISED dataset FILE_LIST_PATH must either be a directory path or a .csv file name')

    # Check if preprocessing is enabled and the file list path is a .csv file
    if ext == '.csv' and config.UNSUPERVISED.DATA.DO_PREPROCESS:
        # Raise an error to prevent overwriting an existing .csv file with preprocessing results
        raise ValueError('User specified UNSUPERVISED dataset FILE_LIST_PATH .csv file already exists. \
                         Please turn DO_PREPROCESS to False or delete existing UNSUPERVISED dataset FILE_LIST_PATH .csv file.')

    # Establish the directory to hold pre-trained models from a given experiment inside. The configured log directory (runs/exp by default)
    config.MODEL.MODEL_DIR = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, config.MODEL.MODEL_DIR)
    
    # Set up the directory for saving output files based on the mode of operation.
    # The directory is constructed within the configured log directory (default: runs/exp).
    
    # If the toolbox mode is 'train_and_test' or 'only_test', set the output directory for test results.
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
        # Construct the path for saving test outputs.
        config.TEST.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, config.TEST.DATA.EXP_DATA_NAME, 'saved_test_outputs')
    
    # If the toolbox mode is 'unsupervised_method', set the output directory for unsupervised results.
    elif config.TOOLBOX_MODE == 'unsupervised_method':
        # Construct the path for saving unsupervised outputs.
        config.UNSUPERVISED.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'saved_outputs')
    
    # Raise an error if the toolbox mode is not recognized.
    else:
        raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')
    
    # Freeze the configuration to prevent further modifications.
    config.freeze()
    return

def _update_config_from_file(config, cfg_file):
    """
    Update the configuration object with settings from a YAML configuration file.

    This function reads a YAML file, merges its settings into the provided
    configuration object, and supports hierarchical configuration by allowing
    base configurations to be specified and recursively merged.

    Args:
        config (CfgNode): The configuration object to be updated.
        cfg_file (str): Path to the YAML configuration file.

    Steps:
    1. Defrost the configuration to allow modifications.
    2. Open and parse the YAML configuration file.
    3. Check for a 'BASE' key in the YAML file to handle hierarchical configurations.
    4. Recursively update the configuration with any base configuration files specified.
    5. Merge the current configuration file's settings into the configuration object.
    6. Freeze the configuration to prevent further modifications.
    """
    
    # Allow modifications to the configuration object
    config.defrost()

    # Open the YAML configuration file and parse its contents
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Handle hierarchical configurations by checking for a 'BASE' key
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            # Recursively update the configuration with base configuration files
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    
     # Log the merging of the current configuration file
    print('=> Merging a config file from {}'.format(cfg_file))
    
    # Merge the current file's settings into the configuration object
    config.merge_from_file(cfg_file)

    # Freeze the configuration to prevent further modifications
    config.freeze()
