'''
1. Read in trained model.
    A Change model to stateful so that prediction can be done frame by frame
        as in online scenario
2. Read in list of clips to test on.
    A Glob
    B Positive and negative samples loop:
        i. Read in the sample
        ii. Predict for each frame
3. Calculate metrics.
    A Generate precision-recall curves
4. Generate report data.
5. Save data to compare models.
'''
import MFIRAP.d01_data.tpa_tools as tpa_tools
from MFIRAP.d04_modelling.models import SETUP_DIC, SETUP_RGB_FLAGS, RGB_FEATURES_LAYER_NAME
from MFIRAP.d00_utils.project import MODEL_CONFIG_KEYS, TRAIN_LOG_FP
from MFIRAP.d04_modelling.models import Model_Evaluation, RNN
import argparse
import tensorflow as tf
import numpy as np
import glob
import os
import datetime
import MFIRAP
import MFIRAP.d00_utils.io as io
import MFIRAP.d00_utils.dataset as ds
import MFIRAP.d00_utils.verbosity as vb
import MFIRAP.d00_utils.project as project
from MFIRAP.d00_utils.paths import ensure_parent_exists
import MFIRAP.d05_model_evaluation.plots as plots
vb.VERBOSITY = vb.SPECIFIC

import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    optimal_threshold = 0.8  # XXX
    parser = argparse.ArgumentParser()
    parser.add_argument('config_json_name', type=str)
    args = parser.parse_args()

    config_json_name = args.config_json_name.split(".json")[0]
    model_config = io.read_json(os.path.join(
        "settings", config_json_name+".json"))
    for key in MODEL_CONFIG_KEYS:
        try:
            model_config[key]
        except KeyError:
            raise Exception(
                "Key {} not found in model configuration.".format(key))
    try:
        Setup = SETUP_DIC[model_config["setup"]]
    except KeyError:
        raise ValueError("Specified setup {} doesn't exist. Implemented setups: {}".format(
            model_config["setup"], SETUP_DIC))

    reporting_dir = os.path.join("data", "06_reporting", config_json_name)
    reporting_img_dir = os.path.join(reporting_dir, "img")
    html_path = os.path.join(reporting_dir, "report.html")
    save_path = os.path.join(reporting_dir, config_json_name+".pkl")
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(html_path):
        os.remove(html_path)
    ensure_parent_exists(html_path)
    ensure_parent_exists(os.path.join(reporting_img_dir, "dummy.ext"))
    
    #1. Read in trained model, also mu and sigma.
    name = config_json_name
    data_models_model_path = os.path.join(project.DATA_MODELS_PATH, name)
    data_models_output_model_path = os.path.join(
        project.DATA_MODELS_OUTPUT_PATH, name)
    vb.print_specific("Model path {}".format(data_models_model_path))
    setup = Model_Evaluation(data_models_model_path, stateful=False)
    mu, sigma = [setup.scaling[key] for key in ['mu', 'sigma']]

    #2A. Read in list of clips to test on.
    subjects = ds.read_test_subjects()
    test_set_path = ds.read_test_set_path()
    clips_fps = glob.glob(os.path.join(
        test_set_path, "subject*", "*", "*ID*.TXT"))
    prefixes = list(set([fp.split("ID")[0] for fp in clips_fps]))
    if len(clips_fps)/len(prefixes) != 3:
        raise Exception(
            "It seems that there are some missing/additional views in your test set")

    #2B. loop
    vb.print_general("Testing {} on {}".format(
        config_json_name, test_set_path))
    vb.print_general(
        "Testing: executing main loop (loading in data + prediction)...")
    view_IDs = model_config["view_IDs"]
    predictions_dict,  sample_classes_dict, labels_dict, timestamps_dict = [
        dict() for i in range(4)]
    prefixes = prefixes
    for prefix in prefixes:
        arrays = []
        timestamps = []
        for id in view_IDs:
            # + Z-score!
            a, ts = tpa_tools.txt2np(prefix+"ID"+id+".TXT")
            arrays.append((a-mu)/sigma)
            timestamps.append(ts)
        header = tpa_tools.read_txt_header(prefix+"ID"+view_IDs[0]+".TXT")
        for chunk in header.split(","):
            if "label" in chunk:
                label = int(chunk.split("label")[-1])
        sample_class = 1 if (label > 0) else 0
        # stateless prediction
        # setup.model.reset_states()
        # F, 32, 32 -> 1, F, 32, 32, 1
        inputs = [np.expand_dims(a, [0, -1]) for a in arrays]
        # append only positive prediction
        predictions_dict[prefix] = setup.model.predict(inputs)[0, :, 1]
        sample_classes_dict[prefix] = sample_class
        # ! max, not avg timestamp at each timestep:
        # you can't have a future frame
        timestamps_dict[prefix] = np.array(timestamps).max(axis=0)
        labels_dict[prefix] = label
    # now we have:
    # predictions_dict, timestamps_dict, labels_dict, sample_classes_dict, 

    # Metrics: #XXX +ROC!
    # Calculated in this step: TP, TN, FP, FN, TT_sum, DTP, DTN, DFP, DFN < D = detection
    # ANTICIPATION
    TP, TN, FP, FN, PT_sum = 0, 0, 0, 0, 0
    # DETECTION
    DTP, DTN, DFP, DFN, TTA_sum = 0, 0, 0, 0, 0
    # calculate
    assert len(prefixes)
    for prefix in prefixes:
        sample_class = sample_classes_dict[prefix]
        label = labels_dict[prefix]
        pred = predictions_dict[prefix]
        timestamps = timestamps_dict[prefix]
        thresh = pred > optimal_threshold
        # if any(thresh) <<< if threshold exceeded at any timestep
        # if P
        if sample_class:
            # TP
            if any(thresh):
                DTP += 1
                # ! TP or FN depends if action predicted before or after the onset
                # ! TP if tta > 0 else FN
                # time-to-action at each timestep
                tte = timestamps[label]-timestamps
                # tte * thresh <<< time-to-action at timesteps at which threshold is exceeded
                TTA = (tte * thresh).max()
                TTA_sum += TTA
                # if TTA positive count as TP because the model PREDICTED the action [before its onset]
                if TTA > 0:
                    TP += 1
                    PT_sum += TTA
                # if TTA negative count as FN! because the model FAILED to PREDICT the action
                else:
                    FN += 1
            # FN
            else:
                DFN += 1
                FN += 1
                # FN will not contribute in any way to ATTA
        # if N
        if not sample_class:
            # TN
            if not any(thresh):
                DTN += 1
                TN += 1
            # FP
            else:
                DFP += 1
                FP += 1

    assert(TP + TN + FP + FN == len(prefixes))
    assert(DTP + DTN + DFP + DFN == len(prefixes))

    # Metrics
    ATTA = TTA_sum/DTP if DTP else np.NaN
    APT = PT_sum/TP if TP else np.NaN
    precision = TP/(TP+FP) if TP+FP else np.NaN
    recall = TP/(TP+FN) if TP+FN else np.NaN
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    detection_precision = DTP/(DTP+DFP) if DTP+DFP else np.NaN
    detection_recall = DTP/(DTP+DFN) if DTP+DFN else np.NaN
    detection_accuracy = (DTP+DTN)/(DTP+DTN+DFP+DFN)
    plots_dict = {}
    # 3A Precision-recall curves
    prc_pre_precisions, prc_pre_recalls, prc_pre_thresholds = plots.prediction_pr_curve(labels_dict, predictions_dict)
    prc_det_precisions, prc_det_recalls, prc_det_thresholds = plots.detection_pr_curve(labels_dict, predictions_dict)
    lim_eps = 0.1
    plots_dict["precision_recall_curve"] = {"precision":prc_pre_precisions, "recall":prc_pre_recalls, "thresholds":prc_pre_thresholds}
    plots_dict["detection_precision_recall_curve"] = {"precision":prc_det_precisions, "recall":prc_det_recalls, "thresholds":prc_det_thresholds}

    # prediction precision-recall curve
    plt.plot(prc_pre_recalls[:-1], prc_pre_precisions[:-1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve for Prediction, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "prediction_precision_recall_curve.png"))
    plt.close()

    # detection precision-recall curve
    plt.plot(prc_det_recalls[:-1], prc_det_precisions[:-1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve for Detection, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "detection_precision_recall_curve.png"))
    plt.close()

    # 3B ROC
    prc_pre_fpr, prc_pre_tpr, prc_pre_thresholds = plots.prediction_roc_curve(labels_dict, predictions_dict)
    prc_det_fpr, prc_det_tpr, prc_det_thresholds = plots.detection_roc_curve(labels_dict, predictions_dict)
    plots_dict["roc"] = {"fpr":prc_pre_fpr, "tpr":prc_pre_tpr, "thresholds":prc_pre_thresholds}
    plots_dict["detection_roc"] = {"fpr":prc_det_fpr, "tpr":prc_det_tpr, "thresholds":prc_det_thresholds}
    lim_eps = 0.1

    # prediction precision-recall curve
    plt.plot(prc_pre_fpr[:-1], prc_pre_tpr[:-1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC for Prediction, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "prediction_roc.png"))
    plt.close()

    # detection precision-recall curve
    plt.plot(prc_det_fpr[:-1], prc_det_tpr[:-1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC for Detection, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "detection_roc.png"))
    plt.close()

    # PRC, DPRC, ATTARC, APTRC 
    # precisions, recalls, thresholds = precision_recall_curve(...)
    # detection_precisions, detection_recalls, detection_thresholds = detection_precision_recall_curve(...)
    # attas, recalls, thresholds = detection_precision_recall_curve(...)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

    # FINAL: GENERATE REPORT (HTML)
    metrics_dict = {"ATTA": ATTA, "APT": APT, 
                    "precision": precision, "recall": recall,
                    "accuracy": accuracy,
                    "detection_precision": detection_precision, "detection_recall": detection_recall,
                    "detection_accuracy": detection_accuracy, 
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN, 
                    "DTP": DTP, "DTN": DTN, "DFP": DFP, "DFN": DFN}

    cfs_table = '''
    <table>
    <thead>
    <tr>
        <th>Task</th>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td rowspan="4">Prediction</td>
        <td>TP</td>
        <td>{TP}</td>
    </tr>
    <tr>
        <td>FP</td>
        <td>{FP}</td>
    </tr>
    <tr>
        <td>TN</td>
        <td>{TN}</td>
    </tr>
    <tr>
        <td>FN</td>
        <td>{FN}</td>
    </tr>
    <tr>
        <td rowspan="4">Detection</td>
        <td>DTP</td>
        <td>{DTP}</td>
    </tr>
    <tr>
        <td>DFP</td>
        <td>{DFP}</td>
    </tr>
    <tr>
        <td>DTN</td>
        <td>{DTN}</td>
    </tr>
    <tr>
        <td>DFN</td>
        <td>{DFN}</td>
    </tr>
    </tbody>
    </table>
    '''.format(**metrics_dict)

    metrics_table = '''
    <table>
    <thead>
    <tr>
        <th>Task</th>
        <th>Metric</th>
        <th>Value</th>
        <th>Remarks</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td rowspan="4">Prediction</td>
        <td>APT</td>
        <td>{APT:.2f} s</td>
        <td>prediction time, always positive</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>{precision:.2f}</td>
        <td>TP/(TP+FP)</td>
    </tr>
    <tr>
        <td>Recall</td>
        <td>{recall:.2f}</td>
        <td>TP/(TP+FN)</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>{accuracy:.2f}</td>
        <td>(TP+TN)/(TP+TN+FP+FN)</td>
    </tr>
    <tr>
        <td rowspan="4">Detection</td>
        <td>ATTA</td>
        <td>{ATTA:.2f} s</td>
        <td>prediction and detection time</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>{detection_precision:.2f}</td>
        <td>DTP/(DTP+DFP)</td>
    </tr>
    <tr>
        <td>Recall</td>
        <td>{detection_recall:.2f}</td>
        <td>DTP/(DTP+DFN)</td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td>{detection_accuracy:.2f}</td>
        <td>(DTP+DTN)/(DTP+DTN+DFP+DFN)</td>
    </tr>
    </tbody>
    </table>
    '''.format(**metrics_dict)

    ################################################################
    dt = datetime.datetime.now()
    report_data = {"config_json_name": config_json_name, "date" : dt.date(), "time" : dt.time(), "optimal_threshold" : optimal_threshold,
                   "cfs_table": cfs_table, "metrics_table": metrics_table, "samples_n": len(prefixes), "pos_n": DTP+DFN, "neg_n": DTN + DFP}

    report_html = '''
    <html>
        <head>

            <title>Evaluation report: {config_json_name}</title>
            <style type = text/css>
                html {{
                    font-family: sans-serif;
                }}
                
                table {{
                    border-collapse: collapse;
                    border: 2px solid rgb(200,200,200);
                    letter-spacing: 1px;
                    font-size: 0.8rem;
                }}
                
                td, th {{
                    border: 1px solid rgb(190,190,190);
                    padding: 10px 20px;
                }}
                
                th {{
                    background-color: rgb(235,235,235);
                }}
                
                td {{
                    text-align: center;
                }}
                
                tr:nth-child(even) td {{
                    background-color: rgb(250,250,250);
                }}
                
                tr:nth-child(odd) td {{
                    background-color: rgb(245,245,245);
                }}
                
                caption {{
                    padding: 10px;
                }}
            </style>

        </head>

        <body>

            <h1>Evaluation report: {config_json_name}</h1>
            <!-- *** General info *** --->
            Tested on {samples_n} samples: {pos_n} positive and {neg_n} negative. <br> Report generated on {date}, {time}.
            <!-- *** Table: metrics  *** --->
            <h2>Performance in numbers</h2>
            Calculations correspond to threshold T={optimal_threshold:.2f}
            <h3>Confusion matrix</h3>
            {cfs_table}
            <h3>Metricsx</h3>
            {metrics_table}
            <h3>Precision-recall curves</h3>
            <img src="img/prediction_precision_recall_curve.png"><br>
            <img src="img/detection_precision_recall_curve.png"><br>

            <h3>ROC</h3>
            <img src="img/prediction_roc.png"><br>
            <img src="img/detection_roc.png"><br>

        </body>
    </html> 
    '''.format(**report_data)
    with open(html_path, 'w') as f:
        f.write(report_html)

    #5 
    data_dict = {"name": config_json_name, "metrics_dict":metrics_dict, "plots_dict":plots_dict}
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)



