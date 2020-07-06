'''
Project-specific flags
'''
import os
import numpy as np

TPA_FRAME_SHAPE = [32, 32, 1]
N_CLASSES = 2
FLOATX = 'float32'

THRESHOLD_FILE_PATTERN = "threshold_{}.pkl"
TESTING_RESULTS_FILE_PATTERN = "testing_results_{}.pkl"
REPORT_DATA_FILE_PATTERN = "report_data_{}.pkl"
REPORT_HTML_FILE_PATTERN = "{}.html"

TESTING_RESULTS_DICT_KEYS = ["name", "prefixes", "sample_classes_dict", "labels_dict", "predictions_dict", "timestamps_dict", "optimal_threshold"]


DATASET_POSITIVE_ONE_HOT = np.array([0, 1])
DATASET_NEGATIVE_ONE_HOT = np.array([1, 0])

DATA_PATH = os.path.join("data")
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, "03_processed")
DATA_MODELS_PATH = os.path.join(DATA_PATH, "04_models")
DATA_MODELS_OUTPUT_PATH = os.path.join(DATA_PATH, "05_model_output")
DATA_REPORTING_PATH = os.path.join("data", "06_reporting")

MEAN_STD_JSON = os.path.join(DATA_MODELS_PATH, "normalization.json")


MODEL_CONFIG_KEYS = ['setup', 'dataset_intermediate_path', 'dataset_processed_parent_path', "description",
                     'train_size', 'batch_size', 'loss_function', 'frames', 'frame_shift', 'view_IDs', 'epochs']

TRAIN_LOG_FP = os.path.join("train.log")

DATA_VISUALIZATION_PATH = os.path.join("data", "07_visualization")
