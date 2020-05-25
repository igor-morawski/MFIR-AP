
import MFIRAP
import MFIRAP.d00_utils.io as io
import MFIRAP.d00_utils.dataset as ds
import MFIRAP.d00_utils.verbosity as vb
import MFIRAP.d00_utils.project as project
vb.VERBOSITY = vb.SPECIFIC

import argparse
import os
import tensorflow as tf
from MFIRAP.d04_modelling.models import Model_Evaluation

from MFIRAP.d00_utils.project import MODEL_CONFIG_KEYS, TRAIN_LOG_FP
from MFIRAP.d04_modelling.models import SETUP_DIC, SETUP_RGB_FLAGS, RGB_FEATURES_LAYER_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_name', type=str)
    args = parser.parse_args()
    
    config_json_name = args.config_json_name.split(".json")[0]
    model_config = io.read_json(os.path.join("settings", config_json_name+".json"))
    for key in MODEL_CONFIG_KEYS:
        try:
            model_config[key]
        except KeyError:
            raise Exception("Key {} not found in model configuration.".format(key))
    try:
        Setup = SETUP_DIC[model_config["setup"]]
    except KeyError:
        raise ValueError("Specified setup {} doesn't exist. Implemented setups: {}".format(model_config["setup"], SETUP_DIC))
    
    name = config_json_name
    vb.print_general("Testing model {}".format(name))
    
    data_models_model_path = os.path.join(project.DATA_MODELS_PATH, name)
    data_models_output_model_path = os.path.join(project.DATA_MODELS_OUTPUT_PATH, name)
    vb.print_specific("Model path {}".format(data_models_model_path))
    setup = Model_Evaluation(data_models_model_path)