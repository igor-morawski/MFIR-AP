import pickle
import json
import numpy as np
import HTPA32x32d
import glob
import os
import datetime
import shutil
import imageio
import cv2
import glob
import argparse

import MFIRAP.d00_utils.project as project

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from experiments import TESTING_SUBJ_L as SUBJECTS

VIEW_IDs = ["121", "122", "123"]
DPI = 100

FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30


def get_all_fps(dataset_path, subjects):
    result = []
    fps = []
    for subject in subjects:
        for c in [0, 1]:
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*ID*.TXT")))
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*IDRGB")))
    result = fps.copy()
    return result


def get_prefix(fp):
    fn = os.path.split(fp)[-1]
    return fn.split("ID")[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_hdf5_fp', type=str)
    parser.add_argument('--dataset_path', dest="dataset_path", default="/media/igor/DATA/D01_MFIR-AP-Dataset", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.model_hdf5_fp):
        raise Exception

    head, tail = os.path.split(args.model_hdf5_fp)
    name = tail.split(".")[0]
    testing_results_pkl = os.path.join(head, "testing_results_"+name+".pkl")
    data_models_visualize_dir = os.path.join(project.DATA_VISUALIZATION_PATH, name)
    if not os.path.exists(data_models_visualize_dir):
        os.mkdir(data_models_visualize_dir)
    if not os.path.exists(testing_results_pkl):
        raise Exception

    c_dir = ["fp", "fn", "tp", "tn"]
    [os.mkdir(os.path.join(data_models_visualize_dir, c)) if not os.path.exists(os.path.join(data_models_visualize_dir, c)) else 0 for c in c_dir]


    with open(testing_results_pkl, "rb") as f:
        testing_results_dict = pickle.load(f)
    # prefixesprefixes = testing_results_dict["prefixes"] 
    sample_classes_dict = testing_results_dict["sample_classes_dict"] 
    labels_dict = testing_results_dict["labels_dict"]
    predictions_dict = testing_results_dict["predictions_dict"]  
    timestamps_dict = testing_results_dict["timestamps_dict"]  
    optimal_threshold = testing_results_dict["optimal_threshold"] 

    all_fps = get_all_fps(args.dataset_path, subjects=SUBJECTS)
    all_prefixes = list(set([fp.split("ID")[-2] for fp in all_fps]))
    tp_prefixes, fp_prefixes, tn_prefixes, fn_prefixes = [], [], [], []
    for prefix in all_prefixes:
        sample_class = sample_classes_dict[prefix]
        label = labels_dict[prefix]
        pred = predictions_dict[prefix]
        timestamps = timestamps_dict[prefix]
        thresh = pred > optimal_threshold
        if sample_class:
            # TP
            if any(thresh):
                tte = timestamps[label]-timestamps
                TTA = (tte * thresh).max()
                if TTA > 0:
                    tp_prefixes.append(prefix)
                else:
                    fn_prefixes.append(prefix)
            # FN
            else:
                fn_prefixes.append(prefix)
        # if N
        if not sample_class:
            # TN
            if not any(thresh):
                tn_prefixes.append(prefix)
            # FP
            else:
                fp_prefixes.append(prefix)

    for prefix in None:
        cat = None
        if prefix in tn_prefixes: 
            cat = "tn"  
        if prefix in fp_prefixes:
            cat = "fp" 
        if prefix in tp_prefixes:
            cat = "tp" 
        if prefix in fn_prefixes:
            cat = "fn" 
        assert cat
        video_fp = os.path.join(data_models_visualize_dir, cat, os.path.split(prefix)[-1]+".mp4")
        print(video_fp)
        if os.path.exists(video_fp):
            continue
        tpa_fps = [prefix+"ID{}.TXT".format(id) for id in VIEW_IDs]
        rgb_dir = prefix+"IDRGB"
        _s = HTPA32x32d.dataset.TPA_RGB_Sample_from_filepaths(tpa_fps, rgb_dir)
        arrays, timestamps = _s.TPA.arrays, _s.TPA.timestamps
        rgb_fps, rgb_ts = _s.RGB.filepaths, _s.RGB.timestamps
        img_sequence = [cv2.imread(fp) for fp in rgb_fps]
        rgb_sequence = np.array(img_sequence).astype(np.uint8)
        v_idx = [_s.TPA.ids.copy().index(v) for v in ["123", "121", "122"]]
        data = np.concatenate([_s.TPA.arrays[i] for i in v_idx], axis=2)
        pc = HTPA32x32d.tools.np2pc(data)
        rgb_height, rgb_width = (cv2.imread(_s.RGB.filepaths[0]).shape)[0:2]
        # 
        pc = np.insert(pc, range(pc.shape[2]//len(_s.TPA.arrays), pc.shape[2], pc.shape[2]//len(_s.TPA.arrays)), 0, axis=2)
        old_h, old_w = pc.shape[1:3]
        new_width = int((rgb_width/old_w)*old_w)
        new_height = int((rgb_width/old_w)*old_h)
        #
        pc_reshaped = [cv2.resize(frame, dsize=(
            new_width, new_height), interpolation=cv2.INTER_NEAREST) for frame in pc]
        pc_reshaped = np.array(pc_reshaped).astype(np.uint8)
        margin_size = rgb_width-new_width
        pc_frames, pc_height, pc_width, pc_ch = pc_reshaped.shape
        pc = np.concatenate([pc_reshaped, np.zeros(
            [pc_frames, pc_height, margin_size, pc_ch], dtype=np.uint8)], axis=2)
        img_sequence = [cv2.imread(fp) for fp in _s.RGB.filepaths]
        rgb_sequence = np.array(img_sequence).astype(np.uint8)
        vis = np.concatenate([pc, rgb_sequence], axis=1)
        ts = np.sum(_s._TPA_RGB_timestamps, axis=0) / \
            len(_s._TPA_RGB_timestamps)
        durations = HTPA32x32d.tools.timestamps2frame_durations(ts)
        
        del(_s)
        T, height, width, ch = vis.shape 
        array = vis

        key_correspondence = [key if os.path.split(prefix)[-1] in get_prefix(key) else False for key in predictions_dict.keys()]
        assert any(key_correspondence)
        dict_key = None
        for k in key_correspondence:
            if k:
                dict_key = k
        assert dict_key
        predictions = predictions_dict[dict_key]
        timestamps = timestamps_dict[dict_key]
        assert len(vis) == len(durations) == len(predictions)

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
        frame_size = (video.shape[2], video.shape[1])
        out = cv2.VideoWriter(video_fp, FOURCC, FPS, frame_size)
        timer = 0
        duration = 0
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
            f[H-gh:H, 0:gw, :] = array[idx, 0:gh, 0:gw, ::-1]
            f[H-ph:H, gw:, :] = plots[idx, 0:ph, 0:pw, :]
            while True:
                out.write(f[:, :, ::-1])
                timer = timer+1/FPS
                duration = duration+durations[idx] 
                if (duration > timer):
                    break
        out.release()


'''
        """
        Writes visualization gif to same directory as in self.filepaths,
        the filename follows the template: FILE_PREFIX_ID{id1}-{id2}-...-{idn}.gif

        vis_order - preferred order, 
        """
        if not self.test_alignment():
            raise Exception("Unaligned sequences cannot be synchronized!")
        if vis_order:
            try:
                assert len(vis_order) == len(self.TPA.ids)
                assert set(vis_order) == set(self.TPA.ids)
                v_idx = [self.TPA.ids.copy().index(v) for v in vis_order]
            except:
                print("[WARNING] Visualization order ignored")
                v_idx = list(range(len(self.TPA.ids)))
        else:
            v_idx = list(range(len(self.TPA.ids)))
        data = np.concatenate([self.TPA.arrays[i] for i in v_idx], axis=2)
        pc = HTPA32x32d.tools.np2pc(data)
        rgb_height, rgb_width = (cv2.imread(self.RGB.filepaths[0]).shape)[0:2]
        # 
        pc = np.insert(pc, range(pc.shape[2]//len(self.TPA.arrays), pc.shape[2], pc.shape[2]//len(self.TPA.arrays)), 0, axis=2)
        old_h, old_w = pc.shape[1:3]
        new_width = int((rgb_width/old_w)*old_w)
        new_height = int((rgb_width/old_w)*old_h)
        #
        pc_reshaped = [cv2.resize(frame, dsize=(
            new_width, new_height), interpolation=cv2.INTER_NEAREST) for frame in pc]
        pc_reshaped = np.array(pc_reshaped).astype(np.uint8)
        margin_size = rgb_width-new_width
        pc_frames, pc_height, pc_width, pc_ch = pc_reshaped.shape
        pc = np.concatenate([pc_reshaped, np.zeros(
            [pc_frames, pc_height, margin_size, pc_ch], dtype=np.uint8)], axis=2)
        img_sequence = [cv2.imread(fp) for fp in self.RGB.filepaths]
        rgb_sequence = np.array(img_sequence).astype(np.uint8)
        vis = np.concatenate([pc, rgb_sequence], axis=1)
        ts = np.sum(self._TPA_RGB_timestamps, axis=0) / \
            len(self._TPA_RGB_timestamps)
        duration = HTPA32x32d.tools.timestamps2frame_durations(ts)
        head, tail = os.path.split(self.TPA.filepaths[0])
        fn = _TPA_get_file_prefix(tail) + "ID" + \
            "-".join(self.TPA.ids) + "-RGB" + ".gif"
        fp = os.path.join(head, fn)
        HTPA32x32d.tools.write_pc2gif(vis, fp, duration=duration)
'''