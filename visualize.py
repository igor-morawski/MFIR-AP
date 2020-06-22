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

import imageio
import cv2
import glob

from matplotlib.ticker import PercentFormatter

DPI = 100

pos10 = ["20200605_1418_", "20200605_1510_", "20200605_1503_"]
neg10 = ["20200605_1523_", "20200605_1516_", "20200605_1511_"]
pos11 = ["20200602_1559_", "20200602_1503_", "20200602_1505_"]
neg11 = ["20200602_1600_", "20200602_1606_", "20200602_1611_"]
pos12 = ["20200605_1618_", "20200605_1701_", "20200605_1552_"]
neg12 = ["20200605_1616_", "20200605_1545_", "20200605_1643_"]
pos = pos10+pos11+pos12
neg = neg10+neg11+neg12
prefixes2visualize = list(set(pos + neg))

def read_gif_data(fp):
    reader = imageio.get_reader(fp)
    frames, durations = [], []
    for i, frame in enumerate(reader):
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
        durations.append(reader.get_meta_data()['duration']/1000)
    return np.array(frames), durations

def get_prefix(fp):
    fn = os.path.split(fp)[-1]
    return fn.split("ID")[0]

if __name__ == "__main__":
    ###
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

    reporting_dir = os.path.join("data", "06_reporting", config_json_name)
    reporting_img_dir = os.path.join(reporting_dir, "img")
    html_path = os.path.join(reporting_dir, "report.html")
    save_path = os.path.join(reporting_dir, config_json_name+".pkl")
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(html_path):
        os.remove(html_path)
    ensure_parent_exists(html_path)
    ensure_parent_exists(os.path.join(reporting_img_dir, "dummy.ext"))
    name = config_json_name
    data_models_model_path = os.path.join(project.DATA_MODELS_PATH, name)
    data_models_output_model_path = os.path.join(
        project.DATA_MODELS_OUTPUT_PATH, name)
    with open(os.path.join(data_models_model_path, "testing_results.pkl"), "rb") as f:
        testing_results_dict = pickle.load(f)
    # prefixesprefixes = testing_results_dict["prefixes"] 
    sample_classes_dict = testing_results_dict["sample_classes_dict"] 
    labels_dict = testing_results_dict["labels_dict"]
    predictions_dict = testing_results_dict["predictions_dict"]  
    timestamps_dict = testing_results_dict["timestamps_dict"]  
    optimal_threshold = testing_results_dict["optimal_threshold"] 
    data_models_visualize_dir = project.DATA_VISUALIZATION_PATH

    output_dir = os.path.join(data_models_visualize_dir, name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # delete previous gifs
    pattern = os.path.join(output_dir, "*.gif")
    gifs2delete = glob.glob(pattern)
    if gifs2delete:
        for fp in gifs2delete:
            os.remove(fp)
    gifs_src_path = ds.read_gifs_path()

    ############ MAIN LOOP
    for prefix in prefixes2visualize:
        matches = glob.glob(os.path.join(gifs_src_path,"*",prefix+"*.gif"))
        fp = matches[0]
        array, durations = read_gif_data(fp)
        T, height, width, ch = array.shape 
        key_correspondence = [key if prefix in get_prefix(key) else False for key in predictions_dict.keys()]
        assert any(key_correspondence)
        dict_key = None
        for k in key_correspondence:
            if k:
                dict_key = k
        assert dict_key
        predictions = predictions_dict[dict_key]
        timestamps = timestamps_dict[dict_key]
        assert len(array) == len(durations) == len(predictions)
        plots = []
        plt.close()
        plt.rcParams["figure.dpi"] = DPI 
        figure_width_inches = width/DPI
        figure_height_inches = figure_width_inches*3/4
        fig = plt.figure(num=0, figsize=[figure_width_inches, figure_height_inches], dpi=DPI)
        plt.rcParams["figure.figsize"] = [figure_width_inches, figure_height_inches]
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.title("Predicted probability of the action in the near future")
        plt.xlabel("Time (s)")
        plt.ylabel("Predicted probability")
        plt.xlim([0, timestamps[-1]])
        plt.ylim([0, 1])
        th_line = plt.axhline(optimal_threshold, **{"color": 'green', "linestyle": '--'})
        for idx in range(len(timestamps)):
            x, y = timestamps[:idx], predictions[:idx]
            plt.plot(x, y, color='black') 
            if predictions[idx] >= optimal_threshold:
                th_line.set_color('red')
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plots.append(data)
        fig.clf()
        plt.close()
        big_width = int(2 * width)
        big_height = int(height)
        assert data.shape[-1] == array.shape[-1]
        plots = np.array(plots, dtype=np.uint8)
        W, H = big_width, big_height
        gw, gh = width, height
        ph, pw, _ = plots[0].shape
        video = np.ones([T, H, W, ch], dtype=np.uint8) * 255
        # draw text
        # A: ALARM < grey or red (depends on whether the threshold was exceeded before)
        # B: time to the action onset (s)
        # C: time to the alarm
        font                   = cv2.FONT_HERSHEY_DUPLEX
        alarmLoc = (960-70, 200)
        fontColor              = (0,0,0)
        text_alarm_dict = {"text":"ALARM", "org":alarmLoc, "fontFace":font, "fontScale":1.5, "lineType":2, "thickness": 2}
        tt_alarm_dict = {"fontFace":font, "fontScale":0.75, "lineType":2, "thickness": 2, "color":(0, 0, 0)}
        # logic
        alarm_flag = False
        sample_class = sample_classes_dict[dict_key]
        sample_label = labels_dict[dict_key]
        T_action = timestamps[sample_label] if (sample_label > 0) else np.NaN
        th_ex = (predictions >= optimal_threshold)
        T_alarm = timestamps[np.where(th_ex)].min() if np.any(th_ex) else np.NaN 
        # handle cases by looking at np.NaN
        for idx, f in enumerate(video):
            #A
            if predictions[idx] >= optimal_threshold:
                alarm_flag = True
            if alarm_flag:
                cv2.putText(f, **text_alarm_dict, color=(255, 0, 0))
            else:
                cv2.putText(f, **text_alarm_dict, color=(200, 200, 200))
            #B, C
            current_timestamp = timestamps[idx]
            tta_onset = T_action - current_timestamp
            tta_alarm = T_alarm - current_timestamp
            diff = tta_onset-tta_alarm
            text1 = "Time to the action onset: {: >7.2f} (s)".format(tta_onset)
            text2 = "Time to sounding alarm: {: >8.2f} (s)".format(tta_alarm)
            text3 = "Action predicted {:.2f} s before its onset".format(diff)
            if not np.isnan(tta_onset):
                cv2.putText(f, text=text1, org=(700, 45), **tt_alarm_dict)
            if not np.isnan(tta_alarm):
                cv2.putText(f, text=text2, org=(700, 90), **tt_alarm_dict)
            if not np.isnan(diff):
                cv2.putText(f, text=text3, org=(700, 135), **tt_alarm_dict)

        ###########
        segment_size = 200
        segments = [(i*segment_size, (i+1)*segment_size if (i+1)*segment_size<=T else T) for i in range(int(np.ceil(T/segment_size)))]
        for segment in segments:
            start, stop = segment
            video[start:stop, H-gh:H, 0:gw, :] = array[start:stop, 0:gh, 0:gw, :]
            video[start:stop, H-ph:H, gw:, :] = plots[start:stop, 0:ph, 0:pw, :]
        with imageio.get_writer(os.path.join(output_dir, prefix+"_PLOT_VIZ.gif"), mode='I', duration=durations, loop=True) as writer:
            for image in video:
                writer.append_data(image)
        writer.close()
        video = None
        plots = None
        array = None
    try:
        shutil.copy2(os.path.join(project.DATA_VISUALIZATION_PATH, "gif2mp4.sh"), os.path.join(output_dir, "gif2mp4.sh"))      
        pattern = os.path.join(output_dir, "*.mp4")
        vids2delete = glob.glob(pattern)
        if vids2delete:
            for fp in vids2delete:
                os.remove(fp)
        os.system("cd {}; bash gif2mp4.sh".format(output_dir))
    except FileNotFoundError:
        pass

    
