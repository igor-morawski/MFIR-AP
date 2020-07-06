"""
Augment negative samples by extracting non-overlapping segments.
"""
import argparse
import os
import sys
import glob
import numpy as np
import tqdm
import HTPA32x32d
import copy

DTYPE = "float32"
VIEW_IDS = ["121", "122", "123"]


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def read_txt_header(filepath: str):
    """
    Read Heimann HTPA .txt header.
    Parameters
    ----------
    filepath : str
    Returns
    -------
    str
        TPA file header
    """
    with open(filepath) as f:
        header = f.readline().rstrip()
    return header


def txt2np(filepath: str, array_size: int = 32):
    """
    Convert Heimann HTPA .txt to NumPy array shaped [frames, height, width].
    Parameters
    ----------
    filepath : str
    array_size : int, optional
    Returns
    -------
    np.array
        3D array of temperature distribution sequence, shaped [frames, height, width].
    list
        list of timestamps
    """
    with open(filepath) as f:
        # discard the first line
        _ = f.readline()
        # read line by line now
        line = "dummy line"
        frames = []
        timestamps = []
        while line:
            line = f.readline()
            if line:
                split = line.split(" ")
                frame = split[0: array_size ** 2]
                timestamp = split[-1]
                frame = np.array([int(T) for T in frame], dtype=DTYPE)
                frame = frame.reshape([array_size, array_size], order="F")
                frame *= 1e-2
                frames.append(frame)
                timestamps.append(float(timestamp))
        frames = np.array(frames)
        # the array needs rotating 90 CW
        frames = np.rot90(frames, k=-1, axes=(1, 2))
    return frames, timestamps


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    Credit: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


class Logger:
    def __init__(self, f):
        self.logs = []
        self.f = f

    def log(self, msg):
        print(msg)
        self.f.write(msg+"\n")


def split_into_segments(prefix, frames, relative_to, view_IDs=["121", "122", "123"], rgb=True):
    """
    arguments: fp, frames, relative_to
    returns: {segment_rel_prefix: [start, end]}
    """
    result = {}
    arrays, timestamps = [], []
    for id in view_IDs:
        a, ts = txt2np(prefix+"ID"+id+".TXT")
        arrays.append(a)
        timestamps.append(ts)
    header = read_txt_header(prefix+"ID"+view_IDs[0]+".TXT")
    for chunk in header.split(","):
        if "label" in chunk:
            label = int(chunk.split("label")[-1])
    if label != -1:
        raise ValueError("Sample not asserted to be negative")
    ts0 = timestamps[0]
    ts_n = len(ts0)
    if (ts_n//frames) <= 1:
        start = 0
        end = ts_n
        return {os.path.relpath(prefix, relative_to): [start, end]}
    segs = []
    i_range = ts_n//frames if ((ts_n/frames) -
                               (ts_n//frames)) <= 0.75 else ts_n//frames+1
    for i in range(i_range):
        segs.append([i*frames, (i+1)*frames])
    segs[-1] = [segs[-1][0], ts_n]
    for idx, segment in enumerate(segs):
        rel_path = os.path.relpath(prefix, relative_to)
        rel_path += "{}_".format(idx)
        result[rel_path] = segment
    return result


def get_subject_list(prefixes_list):
    src = list(prefixes_list.copy())
    result = []
    for f in src:
        chunks = splitall(f)
        for chunk in chunks:
            if "subject" in chunk:
                result.append(chunk)
    result = list(set(result))
    result.sort()
    return result


def sort_prefixes_by_subject(prefixes_list):
    src = list(prefixes_list.copy())
    result = {}
    for f in src:
        chunks = splitall(f)
        for chunk in chunks:
            if "subject" in chunk:
                if chunk in result.keys():
                    result[chunk].append(f)
                else:
                    result[chunk] = [f]
    return result


if __name__ == "__main__":
    with open("aug.log", "w") as f:
        f.write("Starting...\n")
    with open("aug.log", "a") as f:
        ### 1
        logger = Logger(f)
        parser = argparse.ArgumentParser(
            description='Augment negative samples by extracting non-overlapping segments.')
        parser.add_argument('--do', action='store_true')
        parser.add_argument('--frames', type=int, required=True)
        parser.add_argument('directory', type=str, help="Labeled directory")
        args = parser.parse_args()
        _parent, _name = os.path.split(args.directory)
        output_directory = os.path.join(_parent, _name+"_augmented")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        ### /1

        ### 2
        negative_fps = glob.glob(os.path.join(
            args.directory, "*", "0", "*.TXT"))
        if not negative_fps:
            raise ValueError
        prefixes = list(set([fp.split("ID")[0] for fp in negative_fps]))
        segments_dict = {}
        for prefix in tqdm.tqdm(prefixes):
            segments_dict[prefix] = split_into_segments(
                prefix, frames=args.frames, relative_to=args.directory)
        for key in segments_dict.keys():
            logger.log("Prefix {} split into {} segments".format(
                key, len(segments_dict[key])))
        subjects = get_subject_list(prefixes)
        logger.log("Subjects: {}".format(", ".join(subjects)))
        prefixes_by_subject = sort_prefixes_by_subject(prefixes)
        for subject in subjects:
            segments_sum = 0
            for prefix in prefixes_by_subject[subject]:
                segments_sum += len(segments_dict[prefix].keys())
            logger.log("Subject {} split into {} segments".format(
                subject, segments_sum))

        ### 3
        if args.do:
            print("The following actions will be performed if you continue.")
            ans = query_yes_no("Continue?", default="no")
        else:
            exit(0)
        if not ans:
            exit(0)
        logger.log("Continuing...")
        ### /3

        ### 4
        for prefix in segments_dict.keys():
            # load
            tpa_fps, rgb_fp = [
                prefix+"ID{}.TXT".format(id) for id in VIEW_IDS], prefix+"IDRGB"
            sample = HTPA32x32d.dataset.TPA_RGB_Sample_from_filepaths(
                tpa_fps, rgb_fp)
            header = sample.get_header()
            a, ts, ids = sample.TPA.arrays, sample.TPA.timestamps, sample.TPA.ids
            for rel_fp in segments_dict[prefix].keys():
                start, stop = segments_dict[prefix][rel_fp]
                tpa_output_filepaths = [os.path.join(
                    output_directory, rel_fp+"ID{}.TXT".format(id)) for id in sample.TPA.ids]
                rgb_output_directory = os.path.join(
                    output_directory, rel_fp+"IDRGB")
                HTPA32x32d.tools.ensure_parent_exists(rgb_output_directory)
                s_a, s_ts, s_ids = [array[start:stop] for array in a], [
                    timestamps[start:stop] for timestamps in ts], ids
                segment = HTPA32x32d.dataset.TPA_RGB_Sample_from_data(
                    s_a, s_ts, s_ids, rgb_fp, tpa_output_filepaths, rgb_output_directory, header)

                segment.RGB.timestamps = segment.RGB.timestamps[start:stop]
                segment._update_TPA_RGB_timestamps()
                _sample_T0_min = np.min(
                    [ts[0] for ts in segment._TPA_RGB_timestamps])
                _timestamps = [list(np.array(ts)-_sample_T0_min)
                               for ts in segment._TPA_RGB_timestamps]
                segment.TPA.timestamps = _timestamps[:-1]
                segment.RGB.timestamps = _timestamps[-1]
                segment._update_TPA_RGB_timestamps()
                segment.RGB.filepaths = segment.RGB.filepaths[start:stop]
                segment.write()
        ### /4
        logger.log("Success!")
