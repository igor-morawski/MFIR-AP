import argparse
parser = argparse.ArgumentParser(description='Unaligned > aligned raw.')
parser.add_argument('directory', type=str)
args = parser.parse_args()
print(args.directory)
try:
    import HTPA32x32d
except:
    raise Exception("Can't import HTPA32x32d")
HTPA32x32d.dataset.VERBOSE = True

import os
dataset_dir = args.directory  

import json
HTPA32x32d.dataset.convert_TXT2NPZ_TPA_RGB_Dataset(dataset_dir, frames=100, frame_shift=0, crop_to_center=True, size=(299, 299))
