'''
0. Read in predictions
3. Calculate metrics.
    A Generate precision-recall curves
4. Generate report data.
5. Save data to compare models.
'''
import matplotlib.pyplot as plt
import pickle
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
import shutil

if __name__ == "__main__":
    ###
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


    #0
    name = config_json_name
    data_models_model_path = os.path.join(project.DATA_MODELS_PATH, name)
    data_models_output_model_path = os.path.join(
        project.DATA_MODELS_OUTPUT_PATH, name)
    with open(os.path.join(data_models_model_path, "testing_results.pkl"), "rb") as f:
        testing_results_dict = pickle.load(f)
    prefixes = testing_results_dict["prefixes"] 
    sample_classes_dict = testing_results_dict["sample_classes_dict"] 
    labels_dict = testing_results_dict["labels_dict"]
    predictions_dict = testing_results_dict["predictions_dict"]  
    timestamps_dict = testing_results_dict["timestamps_dict"]  
    optimal_threshold = testing_results_dict["optimal_threshold"] 


    # Metrics:
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

         
    metrics_dict = {"ATTA": ATTA, "APT": APT,
                    "precision": precision, "recall": recall,
                    "accuracy": accuracy,
                    "detection_precision": detection_precision, "detection_recall": detection_recall,
                    "detection_accuracy": detection_accuracy,
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                    "DTP": DTP, "DTN": DTN, "DFP": DFP, "DFN": DFN}


    # 3A Precision-recall curves
    prc_pre_precisions, prc_pre_recalls, prc_pre_thresholds = plots.prediction_pr_curve(
        labels_dict, predictions_dict)
    prc_det_precisions, prc_det_recalls, prc_det_thresholds = plots.detection_pr_curve(
        labels_dict, predictions_dict)
    lim_eps = 0.1
    plots_dict["precision_recall_curve"] = {
        "precision": prc_pre_precisions, "recall": prc_pre_recalls, "thresholds": prc_pre_thresholds}
    plots_dict["detection_precision_recall_curve"] = {
        "precision": prc_det_precisions, "recall": prc_det_recalls, "thresholds": prc_det_thresholds}

    # prediction precision-recall curve
    plt.plot(prc_pre_recalls[:-1], prc_pre_precisions[:-1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve for Prediction, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir,
                             "prediction_precision_recall_curve.png"))
    plt.close()

    # detection precision-recall curve
    plt.plot(prc_det_recalls[:-1], prc_det_precisions[:-1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve for Detection, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.ylim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir,
                             "detection_precision_recall_curve.png"))
    plt.close()

    # 3B ROC
    prc_pre_fpr, prc_pre_tpr, prc_pre_thresholds = plots.prediction_roc_curve(
        labels_dict, predictions_dict)
    prc_det_fpr, prc_det_tpr, prc_det_thresholds = plots.detection_roc_curve(
        labels_dict, predictions_dict)
    prediction_auc = plots.auc(prc_pre_fpr, prc_pre_tpr)
    detection_auc = plots.auc(prc_det_fpr, prc_det_tpr)
    metrics_dict["prediction_auc"] = prediction_auc
    metrics_dict["detection_auc"] = detection_auc
    plots_dict["roc"] = {"fpr": prc_pre_fpr,
                         "tpr": prc_pre_tpr, "thresholds": prc_pre_thresholds}
    plots_dict["detection_roc"] = {
        "fpr": prc_det_fpr, "tpr": prc_det_tpr, "thresholds": prc_det_thresholds}
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

    #3C ATTA-recall
    arc_atta_pre, arc_recall_pre, arc_threshold_pre = plots.prediction_atta_recall_curve(
        labels_dict, predictions_dict, timestamps_dict)
    arc_atta_det, arc_recall_det, arc_threshold_det = plots.detection_atta_recall_curve(
        labels_dict, predictions_dict, timestamps_dict)
    plots_dict["apt_recall_curve"] = {"apt": arc_atta_pre,
                         "recall": arc_recall_pre, "thresholds": arc_threshold_pre}
    plots_dict["adt_recall_curve"] = {
        "adt": arc_atta_det, "recall": arc_recall_det, "thresholds": arc_threshold_det}
    # prediction atta-recall curve
    plt.plot(arc_recall_pre, arc_atta_pre)
    plt.xlabel('Recall')
    plt.ylabel('APT [s]')
    plt.title("APT-Recall for Prediction, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "prediction_atta_recall_curve.png"))
    plt.close()
    # detection atta-recall curve
    plt.plot(arc_recall_det, arc_atta_det)
    plt.xlabel('Recall')
    plt.ylabel('ADT [s]')
    plt.title("ADT-Recall for Detection, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "detection_atta_recall_curve.png"))
    plt.close()

    #3D ATTA-precision
    apc_atta_pre, apc_precision_pre, apc_threshold_pre = plots.prediction_atta_precision_curve(
        labels_dict, predictions_dict, timestamps_dict)
    apc_atta_det, apc_precision_det, apc_threshold_det = plots.detection_atta_precision_curve(
        labels_dict, predictions_dict, timestamps_dict)
    plots_dict["apt_precision_curve"] = {"apt": apc_atta_pre,
                         "precision": apc_precision_pre, "thresholds": apc_threshold_pre}
    plots_dict["adt_precision_curve"] = {
        "adt": apc_atta_det, "precision": apc_precision_det, "thresholds": apc_threshold_det}
    # prediction atta-precision curve
    plt.plot(apc_precision_pre, apc_atta_pre)
    plt.xlabel('precision')
    plt.ylabel('APT [s]')
    plt.title("APT-precision for Prediction, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "prediction_atta_precision_curve.png"))
    plt.close()
    # detection atta-precision curve
    plt.plot(apc_precision_det, apc_atta_det)
    plt.xlabel('precision')
    plt.ylabel('ADT [s]')
    plt.title("ADT-precision for Detection, {}".format(config_json_name))
    plt.xlim(0, 1+lim_eps)
    plt.savefig(os.path.join(reporting_img_dir, "detection_atta_precision_curve.png"))
    plt.close()

    # get 100% TPR 0%FPR bound for ATP and ADP:
    min_avg_t, max_avg_t = plots.compute_max_min_average_time_bounds(labels_dict, timestamps_dict) 
    plots_dict['min_avg_t'], plots_dict['max_avg_t'] =  min_avg_t, max_avg_t

    
    # PRC, DPRC, ATTARC, APTRC
    # precisions, recalls, thresholds = precision_recall_curve(...)
    # detection_precisions, detection_recalls, detection_thresholds = detection_precision_recall_curve(...)
    # attas, recalls, thresholds = detection_precision_recall_curve(...)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

    # FINAL: GENERATE REPORT (HTML)
    summary = '''
    <table>
    <thead>
    <tr>prediction_atta_recall_curve
    </thead>
    <tbody>
    <tr>
        <td>Epochs</td>
        <td>{epochs}</td>
    </tr>
    <tr>
        <td>Batch size</td>
        <td>{batch_size}</td>
    </tr>
    <tr>
        <td>Train set size</td>
        <td>{train_size}</td>
    </tr>
    <tr>
        <td>Loss fnc.</td>
        <td>{loss_function}</td>
    </tr>
    <tr>
        <td>Frames</td>
        <td>{frames}</td>
    </tr>
    <tr>
        <td>Frame shift</td>
        <td>{frame_shift}</td>
    </tr>
    <tr>
        <td>Views</td>
        <td>{view_IDs}</td>
    </tr>
    </tbody>
    </table>
    {description}
    '''.format(**{"config_json_name": config_json_name}, **model_config)

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
    <tr>l一
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
    shutil.copy2(src=os.path.join("data", "04_models", config_json_name, config_json_name+".png"), dst=os.path.join(reporting_img_dir, "model.png"))
    dt = datetime.datetime.now()
    report_data = {"summary": summary, "config_json_name": config_json_name, "date": dt.date(), "time": dt.time(), "optimal_threshold": optimal_threshold,
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
            <b>
            Tested on {samples_n} samples: {pos_n} positive and {neg_n} negative. <br> Report generated on {date}, {time}.
            </b>
            {summary}metrics_dict
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

            <h3>ATT-recall curves</h3>
            <img src="img/prediction_atta_recall_curve.png"><br>
            <img src="img/detection_atta_recall_curve.png"><br>

            <h3>ATT-precision curves</h3>
            <img src="img/prediction_atta_precision_curve.png"><br>
            <img src="img/detection_atta_precision_curve.png"><br>

            <h3>Architecture</h3>
            <img src="img/model.png"><br>
        </body>
    </html> 
    '''.format(**report_data)
    with open(html_path, 'w') as f:
        f.write(report_html)

    #5
    data_dict = {"name": config_json_name,
                 "metrics_dict": metrics_dict, "plots_dict": plots_dict}
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)

