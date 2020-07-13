
import HTPA32x32d
import MFIRAP.d00_utils.project as project
import MFIRAP.d00_utils.io as io
import glob
import os
import numpy as np
import argparse
import json


SUBJECTS_L = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8", "subject9", "subject10", "subject11", "subject12"]


def get_all_headers(dataset_path, subjects):
    result = []
    fps = []
    for subject in subjects:
        for c in [0, 1]:
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*ID*.TXT")))
    for f in fps:
        header = HTPA32x32d.tools.read_txt_header(f)
        result.append(header)
    return result

def get_all_last_timestamps(dataset_path, subjects):
    result = []
    fps = []
    for subject in subjects:
        for c in [0, 1]:
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*ID*.TXT")))
    for f in fps:
        with open(f, 'r') as handle:
            lines = handle.read().splitlines()
            last_line = lines[-1]
            result.append(float(last_line.split("t: ")[1]))
    return result

def get_all_fps(dataset_path, subjects):
    result = []
    fps = []
    for subject in subjects:
        for c in [0, 1]:
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*ID*.TXT")))
    for f in fps:
        with open(f, 'r') as handle:
            lines = handle.read().splitlines()[1:]
            ts = [float(l.split("t: ")[1]) for l in lines]
        result.extend(np.ediff1d(ts))
    return result
        

def store_value(filepath, key, value):
    if not os.path.exists(filepath):
        data = {}
    else:
        data = io.read_json(filepath)
    data[key] = str(value)
    with open(filepath, "w") as f:
        json.dump(data, f)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/media/igor/DATA/D01_MFIR-AP-Dataset")
    parser.add_argument('--ambient', action='store_true')
    parser.add_argument('--length', action='store_true')
    parser.add_argument('--fps', action='store_true')

    json_file = os.path.join(project.DATA_REPORTING_PATH, "dataset.json")

    store_value(json_file, "control", "OK")

    FLAGS = parser.parse_args()

    fnc_default_arg = (FLAGS.dataset_path, SUBJECTS_L)

    if FLAGS.ambient:
        all_headers = get_all_headers(*fnc_default_arg)
        temperatures = np.array([float(h.split(",")[1]) for h in all_headers], dtype=np.float32)
        hygros = np.array([float(h.split(",")[2].replace("%", "")) for h in all_headers], dtype=np.float32)
        ambient_temperature_mean, ambient_temperature_std = temperatures.mean(), temperatures.std()
        store_value(json_file, "ambient_temperature_mean", ambient_temperature_mean)
        store_value(json_file, "ambient_temperature_std", ambient_temperature_std)
        ambient_hygro_mean, ambient_hygro_std = hygros.mean(), hygros.std()
        store_value(json_file, "ambient_hygro_mean", ambient_hygro_mean)
        store_value(json_file, "ambient_hygro_std", ambient_hygro_std)
        ambient_temperature_min, ambient_temperature_max = temperatures.min(), temperatures.max()
        store_value(json_file, "ambient_temperature_min", ambient_temperature_min)
        store_value(json_file, "ambient_temperature_max", ambient_temperature_max)
        ambient_hygro_min, ambient_hygro_max = hygros.min(), hygros.max()
        store_value(json_file, "ambient_hygro_min", ambient_hygro_min)
        store_value(json_file, "ambient_hygro_max", ambient_hygro_max)
    if FLAGS.length:
        all_last_timestamps = get_all_last_timestamps(*fnc_default_arg)
        times = np.array(all_last_timestamps, dtype=np.float32)
        total_length = times.sum()
        store_value(json_file, "total_length", total_length)
        length_mean, lenght_std = times.mean(), times.std()
        store_value(json_file, "length_mean", length_mean)
        store_value(json_file, "lenght_std", lenght_std)
        length_min, length_max = times.min(), times.max()
        store_value(json_file, "length_min", length_min)
        store_value(json_file, "length_max", length_max)
    if FLAGS.fps:
        all_fps = get_all_fps(*fnc_default_arg)
        fps = np.array(all_fps)
        fps = fps[np.where(fps != 0)]
        fps = 1/fps
        fps_mean, fps_std = fps.mean(), fps.std()
        store_value(json_file, "fps_mean", fps_mean)
        store_value(json_file, "fps_std", fps_std)





