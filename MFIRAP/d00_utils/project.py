'''
Project-specific flags
'''
import os

TPA_FRAME_SHAPE = [32, 32, 1]
N_CLASSES = 2
FLOATX = 'float32'

DATA_MODELS_PATH = os.path.join("data", "04_models")
DATA_MODELS_OUTPUT_PATH = os.path.join("data", "04_models_output")

MODEL_CONFIG_KEYS = ['dataset_intermediate_path', 'dataset_processed_parent_path', 'train_size', 'batch_size', 'rgb', 'loss_function', 'frames', 'frame_shift', 'view_IDs', 'epochs']

TRAIN_LOG_FP = os.path.join("train.log")