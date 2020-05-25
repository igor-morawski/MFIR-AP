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
raw_dir = args.directory # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT

import json
config_f = "align_config.json"
a_file = open(config_f, "r")
json_object = json.load(a_file)
a_file.close()
json_object["raw_input_dir"] = raw_dir
json_object["processed_destination_dir"] = os.path.join(os.path.split(raw_dir)[0], raw_dir+"_aligned")
a_file = open(config_f, "w")
json.dump(json_object, a_file)
a_file.close()

preparer = HTPA32x32d.dataset.TPA_RGB_Preparer()
preparer.config(config_f)
HTPA32x32d.dataset.SYNCHRONIZATION_MAX_ERROR = 5
preparer.prepare() # now fill labels and make_config.json.
