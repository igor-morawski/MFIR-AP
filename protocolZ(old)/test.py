'''
1. Read in trained model.
    A Change model to stateful so that prediction can be done frame by frame
        as in online scenario
2. Read in list of clips to test on.
    A Glob
    B Positive and negative samples loop:
        i. Read in the sample
        ii. Predict for each frame
FINALLY: write predictions to a file
'''
import matplotlib.pyplot as plt
import pickle
import MFIRAP.d01_data.tpa_tools as tpa_tools
from MFIRAP.d04_modelling.models import SETUP_DIC, SETUP_RGB_FLAGS, RGB_FEATURES_LAYER_NAME
from MFIRAP.d00_utils.project import MODEL_CONFIG_KEYS, TRAIN_LOG_FP
from MFIRAP.d04_modelling.models import Model_Evaluation, RNN
import argparse
import tensorflow as tf
import numpy as np
import glob
import os
import datetime
import MFIRAP
import MFIRAP.d00_utils.io as io
import MFIRAP.d00_utils.dataset as ds
import MFIRAP.d00_utils.verbosity as vb
import MFIRAP.d00_utils.project as project
from MFIRAP.d00_utils.paths import ensure_parent_exists
import MFIRAP.d05_model_evaluation.plots as plots
vb.VERBOSITY = vb.SPECIFIC
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_name', type=str)
    args = parser.parse_args()

    config_json_name = args.config_json_name.split(".json")[0]
    model_config = io.read_json(os.path.join(
        "settings", config_json_name+".json"))
    for key in MODEL_CONFIG_KEYS:
        try:
            model_config[key]
        except KeyError:
            raise Exception(
                "Key {} not found in model configuration.".format(key))
    try:
        Setup = SETUP_DIC[model_config["setup"]]
    except KeyError:
        raise ValueError("Specified setup {} doesn't exist. Implemented setups: {}".format(
            model_config["setup"], SETUP_DIC))


    #1. Read in the trained model, also mu and sigma, as well as optimal threshold
    name = config_json_name
    data_models_model_path = os.path.join(project.DATA_MODELS_PATH, name)
    data_models_output_model_path = os.path.join(
        project.DATA_MODELS_OUTPUT_PATH, name)
    vb.print_specific("Model path {}".format(data_models_model_path))
    setup = Model_Evaluation(data_models_model_path, stateful=False)
    mu, sigma = [setup.scaling[key] for key in ['mu', 'sigma']]
    # https://support.sas.com/en/books/reference-books/analyzing-receiver-operating-characteristic-curves-with-sas/review.html
    # Gonen, Mithat. 2007. Analyzing Receiver Operating Characteristic Curves with SAS. Cary, NC: SAS Institute Inc.
    with open(os.path.join(data_models_model_path, "threshold.pkl"), "rb") as f:
        optimal_threshold = pickle.load(f)

    #2A. Read in list of clips to test on.
    subjects = ds.read_test_subjects()
    test_set_path = ds.read_test_set_path()
    clips_fps = glob.glob(os.path.join(
        test_set_path, "subject*", "*", "*ID*.TXT"))
    prefixes = list(set([fp.split("ID")[0] for fp in clips_fps]))
    if len(clips_fps)/len(prefixes) != 3:
        raise Exception(
            "It seems that there are some missing/additional views in your test set")

    #2B. loop
    vb.print_general("Testing {} on {}".format(
        config_json_name, test_set_path))
    vb.print_general(
        "Testing: executing main loop (loading in data + prediction)...")
    view_IDs = model_config["view_IDs"]
    predictions_dict,  sample_classes_dict, labels_dict, timestamps_dict = [
        dict() for i in range(4)]
    prefixes = prefixes
    for prefix in prefixes:
        arrays = []
        timestamps = []
        for id in view_IDs:
            # + Z-score!
            a, ts = tpa_tools.txt2np(prefix+"ID"+id+".TXT")
            arrays.append((a-mu)/sigma)
            timestamps.append(ts)
        header = tpa_tools.read_txt_header(prefix+"ID"+view_IDs[0]+".TXT")
        for chunk in header.split(","):
            if "label" in chunk:
                label = int(chunk.split("label")[-1])
        sample_class = 1 if (label > 0) else 0
        # stateless prediction
        # setup.model.reset_states()
        # F, 32, 32 -> 1, F, 32, 32, 1
        inputs = [np.expand_dims(a, [0, -1]) for a in arrays]
        # append only positive prediction
        predictions_dict[prefix] = setup.model.predict(inputs)[0, :, 1]
        sample_classes_dict[prefix] = sample_class
        # ! max, not avg timestamp at each timestep:
        # you can't have a future frame
        timestamps_dict[prefix] = np.array(timestamps).max(axis=0)
        labels_dict[prefix] = label
    # now we have:
    # predictions_dict, timestamps_dict, labels_dict, sample_classes_dict,

    # write:
    testing_results_dict = {}
    testing_results_dict["prefixes"] = prefixes
    testing_results_dict["sample_classes_dict"] = sample_classes_dict
    testing_results_dict["labels_dict"] = labels_dict
    testing_results_dict["predictions_dict"] = predictions_dict
    testing_results_dict["timestamps_dict"] = timestamps_dict
    testing_results_dict["optimal_threshold"] = optimal_threshold

    with open(os.path.join(data_models_model_path, "testing_results.pkl"), "wb") as f:
        pickle.dump(testing_results_dict, f)