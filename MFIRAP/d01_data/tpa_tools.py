import numpy as np
import MFIRAP.d00_utils.project as pr
DTYPE = pr.FLOATX

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
