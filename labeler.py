import pickle
import json
import numpy as np
import HTPA32x32d
import glob
import os
import datetime
import glob
import argparse
import MFIRAP.d00_utils.io as io


import numpy as np
import cv2
from matplotlib import pyplot as plt

DEFAULT_LABEL = "label"

VIEWS_IDS = ["121", "122", "123"]
def get_json(dataset_path, output_dir, label_name, **kwargs):
    assert os.path.exists(output_dir)
    filepath = os.path.join(output_dir, label_name+".json")
    abs_prefixes = get_all_absolute_prefixes(dataset_path)
    data = {}
    for abs_prefix in abs_prefixes:
        labels = []
        for id in VIEWS_IDS:
            fp = abs_prefix+"ID{}.TXT".format(id)
            header = HTPA32x32d.tools.read_txt_header(fp)
            label=None
            for chunk in header.split(","):
                if label_name in chunk:
                    label = int(chunk.split(label_name)[-1])
            labels.append(label)
        if not len(set(labels)) == 1:
            raise Exception("Label not consistent for {} ({})".format(abs_prefix, labels))
        data[get_prefix(abs_prefix)] = labels[0]
        default_label = None
        if not labels[0]:
            for chunk in header.split(","):
                if DEFAULT_LABEL in chunk:
                    default_label = int(chunk.split(DEFAULT_LABEL)[-1])
            if default_label:
                if default_label < 0:
                    data[get_prefix(abs_prefix)] = default_label
    with open(filepath, "w") as f:
        json.dump(data, f)

def get_prefix(fp):
    fn = os.path.split(fp)[-1]
    return fn.split("ID")[0]

def get_absolute_prefix(fp):
    return fp.split("ID")[-2]


def get_all_absolute_prefixes(dataset_path):
    fps = []
    for c in (0, 1):
        fps.extend(glob.glob(os.path.join(dataset_path, "*", str(c), "*ID*.TXT")))
    prefixes = []
    for fp in fps:
        prefixes.append(get_absolute_prefix(fp))
    prefixes = list(set(prefixes))
    return prefixes


def write_from_json(dataset_path, label_json_fp, label_name, **kwargs):
    data = io.read_json(label_json_fp)
    fps = []
    for c in (0, 1):
        fps.extend(glob.glob(os.path.join(dataset_path, "*", str(c), "*ID*.TXT")))
    for fp in fps:
        prefix = get_prefix(fp)
        header = HTPA32x32d.tools.read_txt_header(fp)
        label = None
        for chunk in header.split(","):
            if label_name in chunk:
                label = int(chunk.split(label_name)[-1])
        if label:
            print("Label for {} exists, ignoring!".format(fp))
        if not label:
            new_label = data[get_prefix(fp)]
            assert type(new_label) == int
            new_header = header+","+label_name+str(new_label)
            HTPA32x32d.tools.modify_txt_header(fp, new_header)


def label_gui(dataset_path, label_json_fp, label_name, label_batch_size, **kwargs):
    global global_curr_pos
    global global_max_pos
    global global_timesteps
    global global_img_fp_dict
    global global_label_pos
    data = io.read_json(label_json_fp)
    fps = []
    for c in (0, 1):
        fps.extend(glob.glob(os.path.join(dataset_path, "*", str(c), "*ID*.TXT")))
    loop_idx = 0
    for fp in fps:
        prefix = get_prefix(fp)
        if data[prefix]:
            continue
        absolute_prefix = get_absolute_prefix(fp)
        rgb_dir = absolute_prefix+"IDRGB"
        img_fp_list = glob.glob(os.path.join(rgb_dir, "*.jpg"))
        global_img_fp_dict = {}
        for img_fp in img_fp_list:
            timestep = float(os.path.split(img_fp)[-1].split(".jpg")[0].replace("-","."))
            global_img_fp_dict[timestep] = img_fp
        global_timesteps = list(global_img_fp_dict.keys())
        global_timesteps.sort()


        global_max_pos = len(global_timesteps)-1
        global_curr_pos = global_max_pos
        global_label_pos = None
        def key_event(e):
            global global_curr_pos
            global global_max_pos
            global global_timesteps
            global global_img_fp_dict
            global global_label_pos
            if e.key == " ":
                global_label_pos = global_curr_pos
                plt.close()
                return
            if e.key == "right":
                global_curr_pos = global_curr_pos + 1
                if global_curr_pos > global_max_pos:
                    global_curr_pos = 0
            elif e.key == "left":
                global_curr_pos = global_curr_pos - 1
                if global_curr_pos < 0:
                    global_curr_pos = global_max_pos
            ax.cla()
            fig.canvas.set_window_title(str(global_curr_pos))
            implot = ax.imshow(cv2.imread(global_img_fp_dict[global_timesteps[global_curr_pos]])[:,:,::-1])
            fig.canvas.draw()


        ax = plt.gca()
        fig = plt.gcf()
        plt.xticks([]), plt.yticks([]) 
        fig.canvas.set_window_title(str(global_curr_pos))
        implot = ax.imshow(cv2.imread(global_img_fp_dict[global_timesteps[global_curr_pos]])[:,:,::-1])
        cid = implot.figure.canvas.mpl_connect('key_press_event', key_event)
        plt.show()

        with open(os.path.join(rgb_dir, "timesteps.pkl"), "rb") as handle:
            timesteps_pkl = pickle.load(handle)
        real_label = timesteps_pkl.index(("{0:.2f}".format(global_timesteps[global_label_pos])).replace(".", "-")+".jpg")
        print("Labeling {} as {} at {}!".format(absolute_prefix, real_label, timesteps_pkl[real_label]))
        data[prefix] = real_label
        loop_idx += 1
        if loop_idx >= label_batch_size:
            break
    with open(label_json_fp, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label_name', type=str)
    parser.add_argument('--dataset_path', dest="dataset_path", default="/media/igor/DATA/D01_MFIR-AP-Dataset", type=str)
    parser.add_argument('--output_dir', dest="output_dir", default="data", type=str)
    parser.add_argument('--get_json', action="store_true")
    parser.add_argument('--write_from_json', action="store_true")
    parser.add_argument('--label_gui', action="store_true")
    parser.add_argument('--label_json_fp', dest="label_json_fp", type=str)
    parser.add_argument('--label_batch_size', dest="label_batch_size", type=int)
    args = parser.parse_args()

    tasks_flags = [args.get_json, args.write_from_json, args.label_gui]
    assert any(tasks_flags)
    assert (np.array(tasks_flags)*1.).sum() == 1

    if args.get_json:
        get_json(**vars(args))

    if args.write_from_json:
        write_from_json(**vars(args))

    if args.label_gui:
        label_gui(**vars(args))