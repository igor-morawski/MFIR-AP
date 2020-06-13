'''
Project-specific flags
'''
import os

TPA_FRAME_SHAPE = [32, 32, 1]
N_CLASSES = 2
FLOATX = 'float32'

DATA_MODELS_PATH = os.path.join("data", "04_models")
DATA_MODELS_OUTPUT_PATH = os.path.join("data", "05_model_output")

MODEL_CONFIG_KEYS = ['setup', 'dataset_intermediate_path', 'dataset_processed_parent_path', "description",
                     'train_size', 'batch_size', 'loss_function', 'frames', 'frame_shift', 'view_IDs', 'epochs']

TRAIN_LOG_FP = os.path.join("train.log")
