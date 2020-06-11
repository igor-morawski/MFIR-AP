import os
import argparse
import pickle
import numpy as np

import pandas as pd
from plotnine import *

theme_set(theme_void())

dict2pd = pd.DataFrame.from_dict
def pad_keys_flat(d):
    src = d.copy()
    result = d.copy()
    max_len = {}
    for key in src.keys():
        max_len[key] = 0
        for model in src[key].keys():
            a = np.array([src[key][model]]).flatten()
            if len(a) > max_len[key]:
                max_len[key] = len(a)
        for model in src[key].keys():
            a = np.array([src[key][model]]).flatten()
            pad_width = (0, max_len[key]-len(a))
            result[key][model] = np.pad(a, pad_width, mode='edge')
    return result

def extend_model_plot_values(d):
    src = d.copy()
    result = {}
    for key in src.keys():
        result[key] = src[key]
    result = pd.DataFrame(result)
    return result

def flat_df(d_dict):
    src = d_dict.copy()
    src = pad_keys_flat(dict2pd(src).T)
    frames = []
    for key in src.keys():
        for model in src[key].keys():
            tmp = extend_model_plot_values(src.T[model])
            tmp['model'] = model
            frames.append(tmp)
    result = pd.concat(frames)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()
    models = args.input

    models_data_dict = {}
    for model in models:
        with open(os.path.join("data", "06_reporting", model, model+".pkl"), 'rb') as f:
            d = pickle.load(f)
        models_data_dict[model] = d

    # ROC and PRC
    rocs, det_rocs, prc, det_prc = [{} for i in range(4)]
    for model in models:
        rocs[model] = models_data_dict[model]['plots_dict']['roc']
        det_rocs[model] = models_data_dict[model]['plots_dict']['detection_roc']
        prc[model] = models_data_dict[model]['plots_dict']['precision_recall_curve']
        det_prc[model] = models_data_dict[model]['plots_dict']['detection_precision_recall_curve']
    #ROC
    # roc_df = pad_keys_flat(dict2pd(rocs).T)
    df = flat_df(rocs)
    roc_plot = (ggplot(df)
        + geom_line(aes(x='fpr', y='tpr', color='model'))
        + ggtitle('Action Prediction ROC')
        )
'''
        (roc_plot + ggplot(extend_model_plot_values(roc_df.T[model]))
                + geom_line(aes(x='fpr', y='tpr'))
                + ggtitle('Action Prediction ROC')
                )

''' 
        