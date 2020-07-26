import pickle
import json
import numpy as np
import HTPA32x32d
import glob
import os
import datetime
import glob
import argparse

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
                    data[get_prefix(abs_prefix)] = default_label20200520_1309_
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('label_name', type=str)
    parser.add_argument('--dataset_path', dest="dataset_path", default="/media/igor/DATA/D01_MFIR-AP-Dataset", type=str)
    parser.add_argument('--output_dir', dest="output_dir", default="data", type=str)
    parser.add_argument('--get_json', action="store_true")
    parser.add_argument('--write_from_json', action="store_true")
    parser.add_argument('--label_json_fp', dest="label_json_fp", type=str)
    args = parser.parse_args()

    tasks_flags = [args.get_json, args.write_from_json]
    assert any(tasks_flags)
    assert (np.array(tasks_flags)*1.).sum() == 1

    if args.get_json:
        get_json(**vars(args))