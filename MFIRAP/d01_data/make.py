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
processed_dir = args.directory  

import json
config_f = "make_config.json"
a_file = open(config_f, "r")
json_object = json.load(a_file)
a_file.close()
json_object["processed_input_dir"] = processed_dir
json_object["labels_filepath"] = os.path.join(os.path.join(processed_dir, "labels.json"))
json_object["dataset_destination_dir"] =  os.path.join(os.path.split(processed_dir)[0], processed_dir+"_labeled")
a_file = open(config_f, "w")
a_file = open(config_f, "w")
json.dump(json_object, a_file)
a_file.close()

maker = HTPA32x32d.dataset.TPA_RGB_Dataset_Maker()
maker.config(config_f)
maker.make()
