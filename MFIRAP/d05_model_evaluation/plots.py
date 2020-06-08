import numpy as np

def _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=False):
    if not (y_pred_dict.keys() == labels_dict.keys()):
        raise ValueError("Number/IDs of samples not consistent.")
    samples_n = len(y_pred_dict.keys())
    y_score = np.empty(samples_n)
    y_true = np.empty(samples_n)
    # get max score in sequence (if crop_past_label=True then only before the action onset)
    for idx, sample in enumerate(y_pred_dict.keys()):
        sample_label = labels_dict[sample]
        sample_class = 1 if (sample_label > 0) else 0
        stop = sample_label if crop_past_label else -1
        y_score[idx] = y_pred_dict[sample][:stop].max()
        y_true[idx] = sample_class
    # sort
    y_true = (y_true == 1)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]    
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def _precision_recall_curve(fps, tps, thresholds):
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

def prediction_pr_curve(labels_dict, y_pred_dict):
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=False)
    return _precision_recall_curve(fps, tps, thresholds)

def detection_pr_curve(labels_dict, y_pred_dict):
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=True)
    return _precision_recall_curve(fps, tps, thresholds)

# XXX
def _at_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=False):
    if not (y_pred_dict.keys() == labels_dict.keys() == timestamps_dict.keys()):
        raise ValueError("Number/IDs of samples not consistent.")
    samples_n = len(y_pred_dict.keys())
    y_score = np.empty(samples_n)
    y_true = np.empty(samples_n)
    ttas = np.empty(samples_n)
    # get max score in sequence (if crop_past_label=True then only before the action onset)
    for idx, sample in enumerate(y_pred_dict.keys()):
        sample_label = labels_dict[sample]
        sample_class = 1 if (sample_label > 0) else 0
        stop = sample_label if crop_past_label else -1
        y_score[idx] = y_pred_dict[sample][:stop].max()
        y_true[idx] = sample_class
    # sort
    y_true = (y_true == 1)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]    
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps


    
        

