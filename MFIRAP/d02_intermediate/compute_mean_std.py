import platform
if not (platform.system() == 'Linux'):
    raise Exception("This script is only tested on Linux!")
import argparse
parser = argparse.ArgumentParser(description='Unaligned > aligned raw.')
parser.add_argument('directory', type=str)
args = parser.parse_args()
import psutil
import numpy as np
import glob
import os
from sys import getsizeof
DATASET_NAME = args.directory
if not os.path.exists(DATASET_NAME):
    raise ValueError
def get_pos_neg_fp_lists(dataset_name):
    pos_fp_list = glob.glob(os.path.join(dataset_name, "*", "1", "*.npz"))
    neg_fp_list = glob.glob(os.path.join(dataset_name, "*", "0", "*.npz"))
    return pos_fp_list, neg_fp_list

def samples_from_fp_list(fp_list):
    return [np.load(f) for f in fp_list]

def read_tpa123_from_npz(npz_sample, dtype=np.float32):
    ids = ('121', '122', '123')
    return [np.expand_dims(npz_sample['array_ID{}'.format(id)],axis=-1).astype(dtype) for id in ids]

pos_fp_list, neg_fp_list = get_pos_neg_fp_lists(DATASET_NAME)
pos_samples = samples_from_fp_list(pos_fp_list)
neg_samples = samples_from_fp_list(neg_fp_list)

import random
data = np.array([0], dtype=np.float16)
random_samples = pos_samples + neg_samples
random.shuffle(random_samples)
idx = 0
mem = psutil.virtual_memory()
available_MiB = mem.available >> 20
raw_data_limit_MiB = available_MiB//3
for sample in random_samples:
    tpa1, tpa2, tpa3 = read_tpa123_from_npz(sample, dtype=np.float16)
    tpas = np.concatenate([tpa1, tpa2, tpa3]).flatten()
    data = np.append(data, tpas)
    mem_used_MiB = (getsizeof(data) >> 20) * 2 # this will be in float32 later
    idx += 1
    if mem_used_MiB >= raw_data_limit_MiB:
        break
data = data.astype(np.float32)
std = data.std()
mean = data.mean()
import pickle
import os
with open(os.path.join("..", "..", "data", "02_intermediate", "mean_std.pkl"), "wb") as f:
    pickle.dump({'mean' : mean, "std" : std, 'samples_used': idx}, f)
print('Used {} samples'.format(idx))