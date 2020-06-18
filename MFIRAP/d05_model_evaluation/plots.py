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
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=True)
    return _precision_recall_curve(fps, tps, thresholds)

def detection_pr_curve(labels_dict, y_pred_dict):
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=False)
    return _precision_recall_curve(fps, tps, thresholds)

def _roc_curve(fps, tps, thresholds):
    # scikit-learn implementation
    optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
    fps = fps[optimal_idxs]
    tps = tps[optimal_idxs]
    thresholds = thresholds[optimal_idxs]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    return fpr, tpr, thresholds

def prediction_roc_curve(labels_dict, y_pred_dict):
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=True)
    return _roc_curve(fps, tps, thresholds)

def detection_roc_curve(labels_dict, y_pred_dict):
    fps, tps, thresholds = _tfps_per_thresh(labels_dict, y_pred_dict, crop_past_label=False)
    return _roc_curve(fps, tps, thresholds)

def auc(x, y):
    """
    Adapted from scikit
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/_ranking.py#L42
    """
    if x.shape[0] < 2:
        return np.NaN

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return np.NaN
                             
    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area
 
def _atta_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=False):
    if not (y_pred_dict.keys() == labels_dict.keys() == timestamps_dict.keys()):
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

    thresholds = y_score[threshold_idxs]
    attas = np.zeros(len(thresholds))
    for t, th in np.ndenumerate(thresholds):
        ttas = []
        for idx, sample in enumerate(y_pred_dict.keys()):
            # load the sample info
            sample_label = labels_dict[sample]
            sample_class = 1 if (sample_label > 0) else 0
            # dismiss if not positive
            if not sample_class:
                continue
            stop = sample_label if crop_past_label else -1
            # dismiss if not classified as positive
            if not(y_pred_dict[sample][:stop].max() >= th):
                continue
            # load the timestamps
            timestamps = timestamps_dict[sample]
            # get timestamps to action (pos, ..., 0, ... neg)
            tte = timestamps[sample_label]-timestamps
            alarm_idx = np.where(y_pred_dict[sample][:stop] >= th)
            tmp = tte[:stop][alarm_idx]
            ttas.append(tmp.max())
        attas[t] = np.array(ttas).mean() if len(ttas) else np.NaN
        if len(ttas) != tps[t]:
            raise ValueError("This should never happen, TP count doesn't agree in two for loops")
    valid_ids = np.where(attas != np.NaN)
    return attas[valid_ids], fps[valid_ids], tps[valid_ids], y_score[threshold_idxs][valid_ids]


def _atta_recall_curve(attas, fps, tps, thresholds):
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return attas[sl], recall[sl], thresholds[sl]

def prediction_atta_recall_curve(labels_dict, y_pred_dict, timestamps_dict):
    attas, fps, tps, thresholds = _atta_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=True)
    return _atta_recall_curve(attas, fps, tps, thresholds)

def detection_atta_recall_curve(labels_dict, y_pred_dict, timestamps_dict):
    attas, fps, tps, thresholds = _atta_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=False)
    return _atta_recall_curve(attas, fps, tps, thresholds)

def _atta_precision_curve(attas, fps, tps, thresholds):
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return attas[sl], precision[sl], thresholds[sl]

def prediction_atta_precision_curve(labels_dict, y_pred_dict, timestamps_dict):
    attas, fps, tps, thresholds = _atta_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=True)
    return _atta_precision_curve(attas, fps, tps, thresholds)

def detection_atta_precision_curve(labels_dict, y_pred_dict, timestamps_dict):
    attas, fps, tps, thresholds = _atta_per_thresh(labels_dict, y_pred_dict, timestamps_dict, crop_past_label=False)
    return _atta_precision_curve(attas, fps, tps, thresholds)

def compute_max_min_average_time_bounds(labels_dict, timestamps_dict):
    min_t_list = []
    max_t_list = []
    for sample in labels_dict.keys():
        sample_label = labels_dict[sample]
        sample_class = 1 if (sample_label > 0) else 0
        if not sample_class:
            continue
        timestamps = timestamps_dict[sample]
        tte = timestamps[sample_label]-timestamps
        min_t_list.append(tte.min())
        max_t_list.append(tte.max())
    return np.array(min_t_list).mean(), np.array(max_t_list).mean()
