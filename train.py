'''
1. Prepare dataset to read in directly for training.
2. Get data generators and loss function.
3. Get Baseline1 model and compile.
4. Train and save.
5. Report plots.
'''

import argparse

import MFIRAP
import MFIRAP.d00_utils.io as io
import MFIRAP.d00_utils.dataset as ds
import MFIRAP.d00_utils.verbosity as vb
vb.VERBOSITY = vb.SPECIFIC
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
K = tf.keras.backend
import datetime

from MFIRAP.d04_modelling.models import Baseline1
from MFIRAP.d04_modelling.losses import Losses_Keras
from MFIRAP.d03_processing.batch_processing import intermediate2processed
from MFIRAP.d04_modelling.training import Train_Validation_Generators, Timestamper_counted_down
from MFIRAP.d04_modelling.metrics import Metrics_Keras, AUC_AP, Precision_AP, Recall_AP, PrecisionAtRecall_AP
from MFIRAP.d00_utils.project import MODEL_CONFIG_KEYS, TRAIN_LOG_FP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_name', type=str)
    args = parser.parse_args()
    config_json_name = args.config_json_name.split(".json")[0]
    if not os.path.exists(TRAIN_LOG_FP):
        with open(TRAIN_LOG_FP, "w") as f:
            dt = datetime.datetime.now()
            f.write("Log started by {} on {} at {}. \n".format(config_json_name, dt.date(), dt.time()))
    # 1. 
    # Read model
    model_config = io.read_json(os.path.join("settings", config_json_name+".json"))
    for key in MODEL_CONFIG_KEYS:
        try:
            model_config[key]
        except KeyError:
            raise Exception("{} not in model configuration.".format(key))
    dataset_path, destination_parent_path = model_config['dataset_intermediate_path'], model_config['dataset_processed_parent_path']
    processed_develop_path = os.path.join(destination_parent_path, os.path.split(dataset_path)[1])
    if os.path.exists(processed_develop_path):
        print("Dataset exists! Remove manually if needed and restart.")
    else:
        # Initialize feature extractor
        base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)
        intermediate2processed(dataset_path, destination_parent_path, ds.read_development_subjects(), ds.read_test_subjects(), base_model)
    # 2.
    vb.print_specific("Creating training and validation data generators...")
    generators = Train_Validation_Generators(dataset_path=processed_develop_path, view_IDs=["121", "122", "123"], train_size=model_config['train_size'], batch_size=model_config['batch_size'], RGB=model_config['rgb'])
    train_generator = generators.get_train()
    valid_generator = generators.get_valid()
    vb.print_specific("Created training generator of lenght {} and validation generator of length {}".format(len(train_generator), len(valid_generator)))
    if not len(valid_generator):
        valid_generator = None
    vb.print_specific("Loss function: {}".format(model_config['loss_function']))
    losses = Losses_Keras(frames=model_config['frames'], frame_shift=model_config['frame_shift'])
    loss_fnc = losses.get_by_name(model_config['loss_function'], from_logits=False)
    
    # 3.
    # set up atta
    #never forget that! cd = counted_down
    train_timestamps_cd = K.variable(train_generator._get_batch_timestamps(0))
    train_timestamper = Timestamper_counted_down(train_timestamps_cd, trainining_generator=train_generator)
    timestampers = [train_timestamper]
    # settting up atta
    metrics = Metrics_Keras(model_config["frames"], model_config["frame_shift"], train_timestamps_cd)
    atta_fnc = metrics.get_average_time_to_accident()
    # 
    ap_metrics = [AUC_AP(), Precision_AP(), Recall_AP(), PrecisionAtRecall_AP(0.8)]
    compile_kwargs = {"loss":loss_fnc, "optimizer":"adam", "metrics":ap_metrics+[atta_fnc]}
    fit_kwargs = {"x":train_generator, "epochs":model_config['epochs'], "validation_data":valid_generator, "callbacks":timestampers}
    baseline1 = Baseline1(name= model_config['name'], compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs, TPA_view_IDs=model_config['view_IDs'])
    vb.print_specific(baseline1.model.summary())
    vb.print_specific("Compiling...")
    baseline1.compile()
    # 4. Train and save.
    baseline1.delete_existing_model_data_and_output()
    baseline1.fit()
    baseline1.save()
    # 5. Report
    baseline1.plot_metrics(plot_val_metrics=valid_generator)
    with open(TRAIN_LOG_FP, "a") as f:
        dt = datetime.datetime.now()
        f.write("Model {} trained on {} at {}\n".format(config_json_name, dt.date(), dt.time()))
