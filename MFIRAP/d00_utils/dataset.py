import os
import numpy as np
import glob
import MFIRAP.d00_utils.paths
import MFIRAP.d00_utils.io

def get_TPA_mean_and_std(sample_list):
    if not sample_list:
        raise ValueError("Sample list is empty!")
    data = read_tpa123_from_npz(sample_list[0], dtype=np.float16)[0][0][0][0][0]
    for sample in sample_list:
        tpa1, tpa2, tpa3 = read_tpa123_from_npz(sample, dtype=np.float16)
        tpas = np.concatenate([tpa1, tpa2, tpa3]).flatten()
        data = np.append(data, tpas)
    data = data.astype(np.float32)
    return data.mean(), data.std()

def filter_subjects_from_fp_list(fp_list, target_subjects):
    result = []
    for fp in fp_list:
        fp_s = MFIRAP.d00_utils.paths._split_path_into_components(fp)
        subj_in = []
        for subj in target_subjects:
            if subj in fp_s:
                subj_in.append(True)
            else:
                subj_in.append(False)
        if any(subj_in):
            result.append(fp)
    return result

def get_pos_neg_fp_lists(dataset_path):
    pos_fp_list = glob.glob(os.path.join(dataset_path, "*", "1", "*.npz"))
    neg_fp_list = glob.glob(os.path.join(dataset_path, "*", "0", "*.npz"))
    return pos_fp_list, neg_fp_list

def samples_from_fp_list(fp_list):
    return [np.load(f) for f in fp_list]

def read_tpa123_from_npz(npz_sample, dtype=np.float32):
    ids = ('121', '122', '123')
    return [np.expand_dims(npz_sample['array_ID{}'.format(id)],axis=-1).astype(dtype) for id in ids]

def read_rgb_from_npz(npz_sample):
    return npz_sample['array_IDRGB']

def normalize_TPA(array, a, b, scale=1):
    return scale*np.tanh(a*(array-b))

def standarize_TPA(array, mean, std):
    return (array - mean) / std

def standarize_RGB(rgb_sequence):
    return (rgb_sequence - rgb_sequence.mean()) / rgb_sequence.std()

def read_development_subjects(dataset_config_json = os.path.join("settings", "dataset.json")):
    return MFIRAP.d00_utils.io.read_json_key(dataset_config_json, 'development_subjects')

def read_test_subjects(dataset_config_json = os.path.join("settings", "dataset.json")):
    return MFIRAP.d00_utils.io.read_json_key(dataset_config_json, 'test_subjects')
