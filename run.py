# os-like
import os
import glob
import argparse
import numpy as np
import json
import pickle
import datetime
import copy

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import HTPA32x32d
import MFIRAP.d00_utils.project as project
import MFIRAP.d00_utils.io as io
from MFIRAP.d04_modelling.training import TXT_Train_Validation_Generators
from MFIRAP.d04_modelling.losses import Losses_Keras
from MFIRAP.d04_modelling.models import SETUP_DIC, Model_Evaluation
from MFIRAP.d04_modelling.metrics import AUC_AP, Precision_AP, Recall_AP, PrecisionAtRecall_AP, Accuracy_AP
import MFIRAP.d05_model_evaluation.plots as plots

from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint

import experiments
from experiments import PROTOCOL_FIXED, PROTOCOL_CROSS_SUBJ, PROTOCOL_DICT
from experiments import configure_experiments

# computations

# /media/igor/DATA/D01_MFIR-AP-Dataset

# TODO
# CALLBACKS


LIM_EPS = 0
# project-specific


def compute_mean_std(dataset_path, subjects):
    fps = []
    for subject in subjects:
        for c in [0, 1]:
            fps.extend(glob.glob(os.path.join(
                dataset_path, subject, str(c), "*ID*.TXT")))
    value, _ = HTPA32x32d.tools.txt2np(fps[0], array_size=32)
    values = np.array([value.mean()], dtype=np.float32)
    for f in fps:
        a, _ = HTPA32x32d.tools.txt2np(f, array_size=32)
        values = np.hstack([values, a.flatten()])
    return values.mean(), values.std()


def store_mean_std(key, mean, std):
    if not os.path.exists(project.MEAN_STD_JSON):
        data = {}
    else:
        data = io.read_json(project.MEAN_STD_JSON)
    data[key] = str("{}, {}".format(mean, std))
    with open(project.MEAN_STD_JSON, "w") as f:
        json.dump(data, f)
    return True


def _preprocess_fold(l):
    result = list(set(l))
    result.sort()
    return result


def _get_key(l):
    result = _preprocess_fold(l)
    return str(result)


def get_mean_std(dataset_path, subjects):
    key = _get_key(subjects)
    try:
        value = io.read_json_key(project.MEAN_STD_JSON, key=key)
        cached = value
    except FileNotFoundError:
        cached = False
    except json.JSONDecodeError:
        cached = False
    except KeyError:
        cached = False
    if not cached:
        print("Calculating mu, std for: {}".format(subjects))
        mu, sigma = compute_mean_std(dataset_path, subjects)
        store_mean_std(key, mu, sigma)
    if cached:
        mu, sigma = [float(e) for e in value.split(",")]
    return mu, sigma


def _get_fn_excluded_subj(dict, fold_subjects, all_subjects):
    diff = list(set(all_subjects) - set(fold_subjects))
    diff.sort()
    return "{}_".format(dict["name"])+"_".join(diff).replace("subject", "")


def _get_fn_included_subj(dict, fold_subjects):
    result = list(set(fold_subjects))
    result.sort()
    return "{}_".format(dict["name"])+"_".join(result).replace("subject", "")


def combine(experiment_setup_obj):
    plots_dir = os.path.join(project.DATA_REPORTING_PATH, "comparison_plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    plots_fps = []
    models_data_dict = {}
    rocs, det_rocs, prc, det_prc = [{} for i in range(4)]
    aptpc, adtpc, aptrc, adtrc = [{} for i in range(4)]
    min_avg_ts, max_avg_ts = {}, {}

    for (model, protocol) in experiment_setup_obj.model_protocol_pairs:
        model_dict = model.dict()
        data_models_model_path = os.path.join(
            project.DATA_MODELS_PATH, model_dict['name'])
        report_models_model_path = os.path.join(
            project.DATA_REPORTING_PATH, model_dict['name'])
        with open(os.path.join(data_models_model_path, project.REPORT_DATA_FILE_PATTERN.format(PROTOCOL_DICT[protocol])), 'rb') as f:
            d = pickle.load(f)
        models_data_dict[(model.dict()["name"], protocol)] = d
        models_data_dict[(model.dict()["name"], protocol)
                         ]["metrics_dict"]["name"] = model.dict()["name"]
        # ROC, PRC, ATPC, ATRC
        if protocol == PROTOCOL_FIXED:
            rocs[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['roc']
            det_rocs[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['detection_roc']
            prc[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['precision_recall_curve']
            det_prc[(model.dict()["name"], protocol)] = models_data_dict[(model.dict()[
                "name"], protocol)]['plots_dict']['detection_precision_recall_curve']
            aptpc[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['apt_precision_curve']
            adtpc[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['adt_precision_curve']
            aptrc[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['apt_recall_curve']
            adtrc[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['adt_recall_curve']
            min_avg_ts[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['min_avg_t']
            max_avg_ts[(model.dict()["name"], protocol)] = models_data_dict[(
                model.dict()["name"], protocol)]['plots_dict']['max_avg_t']

    # plot settings
    fontP = FontProperties()
    fontP.set_size('small')
    loc_args = {'loc': (1.05, 0.5), 'ncol': 2, 'prop': fontP}
    plot_args = {'lw': 2}

    GUIDELINES_KWARGS = {"color": 'gray', "linestyle": '--'}

    def draw_hline(y=0.5):
        plt.axhline(y=y, **GUIDELINES_KWARGS)
        return True

    def draw_iden_line():
        plt.plot([0, 1], [0, 1], **GUIDELINES_KWARGS)

    def set_lims():
        plt.xlim(0, 1+LIM_EPS)
        plt.ylim(0, 1+LIM_EPS)

    # plots
    # ROC, PRC, ATPC, ATRC
    for ablation in experiment_setup_obj.ablations:
        if ablation.protocol != PROTOCOL_FIXED:
            continue
        models = ablation.models
        protocol = ablation.protocol
        plt.title(
            'ROC - prediction {}'.format("(ablation {})".format(ablation.name)))
        for model in models:
            d = rocs[(model.dict()["name"], protocol)]
            fpr, tpr = d['fpr'], d['tpr']
            plt.plot(fpr, tpr, label=model.name+", AUC = {:.2f}".format(models_data_dict[(
                model.dict()["name"], protocol)]["metrics_dict"]["prediction_auc"]), **plot_args)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(**loc_args)
        set_lims()
        draw_iden_line()
        fp = os.path.join(
            plots_dir, "prediction_roc{}.svg".format(ablation.name))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
        plt.close()
        plt.title(
            'ROC - detection {}'.format("(ablation {})".format(ablation.name)))
        for model in models:
            d = det_rocs[(model.dict()["name"], protocol)]
            fpr, tpr = d['fpr'], d['tpr']
            plt.plot(fpr, tpr, label=model.name+", AUC = {:.4f}".format(models_data_dict[(
                model.dict()["name"], protocol)]["metrics_dict"]["detection_auc"]), **plot_args)
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.legend(**loc_args)
            set_lims()
            draw_iden_line()
            fp = os.path.join(
                plots_dir, "detection_roc{}.svg".format(ablation.name))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            plt.title(
                'Precision-recall curve - prediction {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = prc[(model.dict()["name"], protocol)]
                precision, recall = d['precision'], d['recall']
                plt.plot(recall, precision, label=model.name, **plot_args)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(**loc_args)
            set_lims()
            draw_hline()
            fp = os.path.join(plots_dir, "prediction_prc{}.svg".format(
                "(ablation {})".format(ablation.name)))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            plt.title(
                'Precision-recall curve - detection {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = det_prc[(model.dict()["name"], protocol)]
                precision, recall = d['precision'], d['recall']
                plt.plot(recall, precision, label=model.name, **plot_args)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(**loc_args)
            set_lims()
            draw_hline()
            fp = os.path.join(plots_dir, "detection_prc{}.svg".format(
                "(ablation {})".format(ablation.name)))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            # APT-Recall
            plt.title(
                'APT-recall curve - prediction {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = aptrc[(model.dict()["name"], protocol)]
                apt, recall = d['apt'], d['recall']
                plt.plot(recall, apt, label=model.name, **plot_args)
            plt.xlabel('Recall')
            plt.ylabel('APT')
            plt.legend(**loc_args)
            fp = os.path.join(plots_dir, "apt_recall_curve{}.svg".format(
                "(ablation {})".format(ablation.name)))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            # ADT-Recall
            plt.title(
                'ADT-recall curve - detection {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = adtrc[(model.dict()["name"], protocol)]
                adt, recall = d['adt'], d['recall']
                plt.plot(recall, adt, label=model.name, **plot_args)
            plt.xlabel('Recall')
            plt.ylabel('ADT')
            plt.legend(**loc_args)
            fp = os.path.join(
                plots_dir, "adt_recall_curve{}.svg".format(ablation.name))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            # APT-Precision
            plt.title(
                'APT-precision curve - prediction {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = aptpc[(model.dict()["name"], protocol)]
                apt, precision = d['apt'], d['precision']
                plt.plot(precision, apt, label=model.name, **plot_args)
            plt.xlabel('Precision')
            plt.ylabel('APT')
            plt.legend(**loc_args)
            fp = os.path.join(
                plots_dir, "apt_precision_curve{}.svg".format(ablation.name))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()

            # ADT-Precision
            plt.title(
                'ADT-precision curve - detection {}'.format("(ablation {})".format(ablation.name)))
            for model in models:
                d = adtpc[(model.dict()["name"], protocol)]
                adt, precision = d['adt'], d['precision']
                plt.plot(precision, adt, label=model.name, **plot_args)
            plt.xlabel('Precision')
            plt.ylabel('ADT')
            plt.legend(**loc_args)
            fp = os.path.join(
                plots_dir, "adt_precision_curve{}.svg".format(ablation.name))
            plt.savefig(fp, bbox_inches='tight')
            plots_fps.append(os.path.relpath(fp, project.DATA_REPORTING_PATH))
            plt.close()
        plots_fps = list(set(plots_fps))
        _plots_fps = plots_fps.copy()
        excluded_plots = ["adt", "detection"]
        plots_fps = list(filter(lambda x: all(
            [e not in x for e in excluded_plots]), _plots_fps))
        plots_fps.sort()
    # /plots

    report_data = {}
    plots_html = str(
        "".join(['''<img src="{}"><br>Figure {}<br>'''.format(e, i) for i, e in enumerate(plots_fps)]))
    report_data["plots_html"] = plots_html
    links_html = str("".join(['''<li> <a href="{}">{}</a></li>'''.format(os.path.relpath(os.path.join(project.DATA_REPORTING_PATH, model.dict()["name"],
                                                                                                      PROTOCOL_DICT[protocol]+".html"), project.DATA_REPORTING_PATH),  model.dict()["name"]) for (model, protocol) in experiment_setup_obj.model_protocol_pairs]))
    report_data["links_html"] = links_html

    sortable_header_keys = ['name', 'APT', 'ATTA',  'precision', 'recall',
                            'accuracy', 'detection_precision', 'detection_recall', 'detection_accuracy', "prediction_auc", "detection_auc"]
    row_pattern = "<tr>{}</tr>"
    cell_pattern = '<td width="{w}%">{brackets}</td>'.format(
        w=100//len(sortable_header_keys), brackets="{}")
    metric_cell_pattern = cell_pattern.format("{:.4f}")
    header_cell_pattern = "<th>{}</th>"

    def cell_formatter(v):
        if v == np.NaN:
            x = -1
        elif str(v) == str(np.NaN):
            x = -1
        else:
            x = v
        try:
            return metric_cell_pattern.format(x)
        except ValueError:
            return cell_pattern.format(x)

    sortable_header = "".join(header_cell_pattern.format(h)
                              for h in sortable_header_keys)
    sortable_table_content = "".join([(row_pattern.format("".join(cell_formatter(
        models_data_dict[(model.dict()["name"], protocol)]["metrics_dict"][key]) for key in sortable_header_keys))) for (model, protocol) in experiment_setup_obj.model_protocol_pairs])

    sortable_header = sortable_header.replace("_", " ")

    terminology_html = '''
    <hr>
    <h2>Appendix A. Metrics and terms.</h2>
    <h3>General</h3>
    <ul>
        <li>Precision = TP/(TP+FP)</li>
        <li>Recall = TP/(TP+FN)</li>
        <li>Accuracy = (TP+TN)/(TP+TN+FP+FN)</li>
        <li>ROC - receiver operating characteristic curve.</li>
        <li>AUC - area under curve - is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.</li>
    </ul>
    <h3>Research-specific</h3>
    <ul>
        <li>APT - prediction time, always a positive number.</li>
        <li>ATTA - prediction and detection time, any real number.</li>
    </ul>
    '''
    report_data["terminology_html"] = terminology_html

    ablations_list_html = "".join("<li>{} - {}</li>".format(ablation.name,
                                                            ablation.description) for ablation in experiment_setup_obj.ablations)
    dataset_and_ablations_html = '''
    <h2>Appendix B. Dataset and ablation variations.</h2>
    <ul>
        <li>Training set subjects: 9</li>
            <ul>  
                <li> 30 positive samples (each),</li>
                <li> negative samples augmented by extracting patches in temporal dimension, positive:negative ratio balanced during training.
            </ul>
        <li>Testing set subjects: 3</li>
            <ul>  
                <li> 30 positive samples (each), around 1-1.5 min. long,</li>
                <li> 30 negative smaples (each), around 1-1.5 min. long.
            </ul>
    </ul>
    <h3>Ablations</h3>
    {ablations_list_html}
    '''.format(**{"ablations_list_html": ablations_list_html})
    report_data["dataset_and_ablations_html"] = dataset_and_ablations_html

    sortable_html = '''
    <table class="sortable" width="100%">
    <thead>
    <tr>{sortable_header}</tr>
    </thead>
    <tbody>
    {sortable_table_content}
    </tbody>
    </table>
    '''.format(**{"sortable_header": sortable_header, "sortable_table_content": sortable_table_content})
    report_data["sortable_html"] = sortable_html

    report_html = '''
    <html>
        <head>
            <script src="sorttable.js"></script>

            <title>Evaluation report: models comparison</title>
            <style type = text/css>
                /* Sortable tables */
                table.sortable thead {{
                    background-color:#eee;
                    color:#666666;
                    font-weight: bold;
                    cursor: default;
                }}

                table.sortable th:not(.sorttable_sorted):not(.sorttable_sorted_reverse):not(.sorttable_nosort):after {{ 
                    content: " \\25B4\\25BE" 
                }}

                /* Regular CSS */
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
            <h1>Evaluation report: models comparison</h1>
            {sortable_html}
            {links_html}
            <br>
            {plots_html}
            {terminology_html}
            {dataset_and_ablations_html}
        </body>
    </html> 
    '''.format(**report_data)
    with open(os.path.join(project.DATA_REPORTING_PATH, "comparison.html"), 'w') as f:
        f.write(report_html)
    return True


def get_model_metrics(testing_results_dict):
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
    plots_dict["precision_recall_curve"] = {
        "precision": prc_pre_precisions, "recall": prc_pre_recalls, "thresholds": prc_pre_thresholds}
    plots_dict["detection_precision_recall_curve"] = {
        "precision": prc_det_precisions, "recall": prc_det_recalls, "thresholds": prc_det_thresholds}

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

    #3C ATTA-recall
    arc_atta_pre, arc_recall_pre, arc_threshold_pre = plots.prediction_atta_recall_curve(
        labels_dict, predictions_dict, timestamps_dict)
    arc_atta_det, arc_recall_det, arc_threshold_det = plots.detection_atta_recall_curve(
        labels_dict, predictions_dict, timestamps_dict)
    plots_dict["apt_recall_curve"] = {"apt": arc_atta_pre,
                                      "recall": arc_recall_pre, "thresholds": arc_threshold_pre}
    plots_dict["adt_recall_curve"] = {
        "adt": arc_atta_det, "recall": arc_recall_det, "thresholds": arc_threshold_det}

    #3D ATTA-precision
    apc_atta_pre, apc_precision_pre, apc_threshold_pre = plots.prediction_atta_precision_curve(
        labels_dict, predictions_dict, timestamps_dict)
    apc_atta_det, apc_precision_det, apc_threshold_det = plots.detection_atta_precision_curve(
        labels_dict, predictions_dict, timestamps_dict)
    plots_dict["apt_precision_curve"] = {"apt": apc_atta_pre,
                                         "precision": apc_precision_pre, "thresholds": apc_threshold_pre}
    plots_dict["adt_precision_curve"] = {
        "adt": apc_atta_det, "precision": apc_precision_det, "thresholds": apc_threshold_det}

    min_avg_t, max_avg_t = plots.compute_max_min_average_time_bounds(
        labels_dict, timestamps_dict)
    plots_dict['min_avg_t'], plots_dict['max_avg_t'] = min_avg_t, max_avg_t

    data_dict = {"name": testing_results_dict["name"],
                 "metrics_dict": metrics_dict, "plots_dict": plots_dict, "testing_results_dict": testing_results_dict}
    return data_dict


def combine_all_metrics(all_metrics_dict_list):
    result = {}
    sum_dict = {}
    sum_keys = ["ATTA", "APT",
                "precision", "recall",
                "accuracy",
                "detection_precision", "detection_recall",
                "detection_accuracy",
                "TP", "TN", "FP", "FN",
                "DTP", "DTN", "DFP", "DFN"]
    for key in sum_keys:
        sum_dict[key] = 0
    for data_dict in all_metrics_dict_list:
        metrics_dict = data_dict["metrics_dict"].copy()
        plots_dict = data_dict["plots_dict"].copy()
        name = data_dict["name"]
        testing_results_dict = data_dict["testing_results_dict"].copy()
        prefixes = testing_results_dict["prefixes"]
        sample_classes_dict = testing_results_dict["sample_classes_dict"].copy(
        )
        labels_dict = testing_results_dict["labels_dict"].copy()
        predictions_dict = testing_results_dict["predictions_dict"].copy()
        timestamps_dict = testing_results_dict["timestamps_dict"].copy()
        optimal_threshold = testing_results_dict["optimal_threshold"]
        for key in sum_keys:
            sum_dict[key] += metrics_dict[key]
    comp_avg_keys = ["ATTA", "APT",
                     "precision", "recall",
                     "accuracy",
                     "detection_precision", "detection_recall",
                     "detection_accuracy"]
    for key in comp_avg_keys:
        sum_dict[key] /= len(all_metrics_dict_list)
    result["plots_dict"] = ["Not implemented"]
    result["metrics_dict"] = sum_dict
    result["metrics_dict"]["prediction_auc"] = np.NaN
    result["metrics_dict"]["detection_auc"] = np.NaN
    result["testing_results_dict"] = {}
    result["testing_results_dict"]["optimal_threshold"] = np.NaN
    result["testing_results_dict"]["sample_classes_dict"] = ["Not implemented"]
    result["testing_results_dict"]["labels_dict"] = ["Not implemented"]
    result["testing_results_dict"]["predictions_dict"] = ["Not implemented"]
    result["testing_results_dict"]["timestamps_dict"] = ["Not implemented"]
    result["testing_results_dict"]["prefixes"] = []
    [result["testing_results_dict"]["prefixes"].extend(
        data_dict["testing_results_dict"]["prefixes"]) for data_dict in all_metrics_dict_list]
    result["name"] = name
    return result


def generate_report_html_data(data_dict, model_dict):
    metrics_dict = data_dict["metrics_dict"].copy()
    plots_dict = data_dict["plots_dict"].copy()
    name = data_dict["name"]
    testing_results_dict = data_dict["testing_results_dict"].copy()
    prefixes = testing_results_dict["prefixes"]
    sample_classes_dict = testing_results_dict["sample_classes_dict"].copy()
    labels_dict = testing_results_dict["labels_dict"].copy()
    predictions_dict = testing_results_dict["predictions_dict"].copy()
    timestamps_dict = testing_results_dict["timestamps_dict"].copy()
    optimal_threshold = testing_results_dict["optimal_threshold"]
    DTP, DFN, DTN, DFP = metrics_dict["DTP"], metrics_dict["DFN"], metrics_dict["DTN"], metrics_dict["DFP"]

    summary_dict = {}
    summary_dict["epochs"] = model_dict["epochs"]
    summary_dict["batch_size"] = model_dict["batch_size"]
    summary_dict["train_size"] = model_dict["train_set_ratio"]
    summary_dict["loss_function"] = model_dict["loss_function"]
    summary_dict["frames"] = model_dict["frames"]
    summary_dict["frame_shift"] = model_dict["frame_shift"]
    summary_dict["view_IDs"] = model_dict["view_IDs"]
    summary = '''
    <table>
    <thead>
    <tr>
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
    '''.format(**summary_dict)

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
        <td>TP/(TP+FP)</td>
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

    dt = datetime.datetime.now()
    report_data = {"summary": summary, "date": dt.date(), "time": dt.time(), "optimal_threshold": optimal_threshold,
                   "cfs_table": cfs_table, "metrics_table": metrics_table, "samples_n": len(prefixes), "pos_n": DTP+DFN, "neg_n": DTN + DFP}

    report_html = '''
    <html>
        <head>
            
            <title>Evaluation report: {name}</title>
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
            <h1>Evaluation report: {name}</h1>
            <!-- *** General info *** --->
            <b>
            Tested on {samples_n} samples: {pos_n} positive and {neg_n} negative. <br> Report generated on {date}, {time}.
            </b>
            {summary}
            <!-- *** Table: metrics  *** --->
            <h2>Performance in numbers</h2>
            Calculations correspond to threshold T={optimal_threshold:.2f}
            <h3>Confusion matrix</h3>
            {cfs_table}
            <h3>Metrics</h3>
            {metrics_table}
        </body>
    </html> 
    '''.format(**report_data, name=name)
    return report_html


def report(experiment_setup_obj):
    for (model, protocol) in experiment_setup_obj.model_protocol_pairs:
        model_dict = model.dict()
        dataset_manager = experiment_setup_obj.dataset_manager
        data_models_model_path = os.path.join(
            project.DATA_MODELS_PATH, model_dict['name'])
        report_models_model_path = os.path.join(
            project.DATA_REPORTING_PATH, model_dict['name'])
        if not os.path.exists(report_models_model_path):
            os.mkdir(report_models_model_path)
        print("Reporting {}, protocol {} ({})...".format(
            model_dict["name"], protocol, PROTOCOL_DICT[protocol]))
        model_dir = os.path.join(project.DATA_MODELS_PATH, model_dict["name"])
        if not os.path.exists(model_dir):
            raise FileNotFoundError
        if protocol == PROTOCOL_FIXED:
            folds_metrics = []
            testing_subject_folds = [dataset_manager.dataset_testing_subj]
            assert len(testing_subject_folds) == 1
            for testing_fold in testing_subject_folds:
                fold_name = _get_fn_included_subj(model_dict, testing_fold)
                if not os.path.exists(os.path.join(data_models_model_path,
                                                   project.TESTING_RESULTS_FILE_PATTERN.format(fold_name))):
                    raise FileNotFoundError
                testing_subject_folds = [dataset_manager.dataset_testing_subj]
                # in protocol fixed there is only one fold so no need for any looping or compiling, just renaming
                with open(os.path.join(data_models_model_path, project.TESTING_RESULTS_FILE_PATTERN.format(fold_name)), "rb") as f:
                    data = pickle.load(f)
                data["name"] = "{}_{}".format(
                    model_dict['name'], PROTOCOL_DICT[protocol])
                assert set(data.keys()) == set(
                    project.TESTING_RESULTS_DICT_KEYS)
                data["protocol"] = protocol
            # Get metrics and plots
            print("Getting report data for {}, protocol {} ({})...".format(
                model_dict["name"], protocol, PROTOCOL_DICT[protocol]))
            model_metrics = get_model_metrics(data.copy())
            html = generate_report_html_data(model_metrics, model_dict)
            with open(os.path.join(data_models_model_path, project.REPORT_DATA_FILE_PATTERN.format(PROTOCOL_DICT[protocol])), "wb") as f:
                pickle.dump(model_metrics, f)
            with open(os.path.join(report_models_model_path, project.REPORT_HTML_FILE_PATTERN.format(PROTOCOL_DICT[protocol])), "w") as f:
                f.write(html)
        if protocol == PROTOCOL_CROSS_SUBJ:
            testing_subject_folds = [[s] for s in dataset_manager.subjects]
            all_metrics = []
            for testing_fold in testing_subject_folds:
                fold_name = _get_fn_included_subj(model_dict, testing_fold)
                assert len(testing_subject_folds) != 1
                fold_name = _get_fn_included_subj(model_dict, testing_fold)
                if not os.path.exists(os.path.join(data_models_model_path,
                                                   project.TESTING_RESULTS_FILE_PATTERN.format(fold_name))):
                    raise FileNotFoundError
                with open(os.path.join(data_models_model_path, project.TESTING_RESULTS_FILE_PATTERN.format(fold_name)), "rb") as f:
                    data = pickle.load(f)
                data["name"] = "{}_{}".format(
                    model_dict['name'], PROTOCOL_DICT[protocol])
                assert set(data.keys()) == set(
                    project.TESTING_RESULTS_DICT_KEYS)
                data["protocol"] = protocol
                all_metrics.append(get_model_metrics(data.copy()))
            model_metrics = combine_all_metrics(all_metrics)
            html = generate_report_html_data(model_metrics, model_dict)
            with open(os.path.join(data_models_model_path, project.REPORT_DATA_FILE_PATTERN.format(PROTOCOL_DICT[protocol])), "wb") as f:
                pickle.dump(model_metrics, f)
            with open(os.path.join(report_models_model_path, project.REPORT_HTML_FILE_PATTERN.format(PROTOCOL_DICT[protocol])), "w") as f:
                f.write(html)
    return True


def model_test(fold_name, model_dir, model_dict, dataset_path, testing_subj, mu, sigma):
    #1. Read in the trained model, as well as optimal threshold, get file list
    data_models_model_path = os.path.join(
        project.DATA_MODELS_PATH, model_dict['name'])
    data_models_output_model_path = os.path.join(
        project.DATA_MODELS_OUTPUT_PATH, model_dict['name'])
    setup = Model_Evaluation(data_models_model_path, fold_name=fold_name,
                             stateful=False, weights_ext="hdf5", load_scaling=False)
    # https://support.sas.com/en/books/reference-books/analyzing-receiver-operating-characteristic-curves-with-sas/review.html
    # Gonen, Mithat. 2007. Analyzing Receiver Operating Characteristic Curves with SAS. Cary, NC: SAS Institute Inc.
    with open(os.path.join(data_models_model_path, project.THRESHOLD_FILE_PATTERN.format(fold_name)), "rb") as f:
        optimal_threshold = pickle.load(f)
    #1A. File list
    clips_fps = []
    for subj in testing_subj:
        clips_fps.extend(glob.glob(os.path.join(
            dataset_path, subj, "*", "*ID*.TXT")))
    prefixes = list(set([fp.split("ID")[0] for fp in clips_fps]))
    if len(clips_fps)/len(prefixes) != 3:
        raise Exception(
            "It seems that there are some missing/additional views in your test set")
    #2. Loop! (:
    print("Testing {}...".format(fold_name))
    predictions_dict,  sample_classes_dict, labels_dict, timestamps_dict = [
        dict() for i in range(4)]
    for prefix in prefixes:
        arrays = []
        timestamps = []
        for id in model_dict['view_IDs']:
            # + Z-score!
            a, ts = HTPA32x32d.tools.txt2np(prefix+"ID"+id+".TXT")
            arrays.append((a-mu)/sigma)
            timestamps.append(ts)
        header = HTPA32x32d.tools.read_txt_header(
            prefix+"ID"+model_dict['view_IDs'][0]+".TXT")
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
    testing_results_dict = {}
    testing_results_dict["name"] = fold_name
    testing_results_dict["prefixes"] = prefixes
    testing_results_dict["sample_classes_dict"] = sample_classes_dict
    testing_results_dict["labels_dict"] = labels_dict
    testing_results_dict["predictions_dict"] = predictions_dict
    testing_results_dict["timestamps_dict"] = timestamps_dict
    testing_results_dict["optimal_threshold"] = optimal_threshold
    assert set(testing_results_dict.keys()) == set(
        project.TESTING_RESULTS_DICT_KEYS)
    with open(os.path.join(data_models_model_path, project.TESTING_RESULTS_FILE_PATTERN.format(fold_name)), "wb") as f:
        pickle.dump(testing_results_dict, f)
    return True


def test(experiment_setup_obj):
    for (model, protocol) in experiment_setup_obj.model_protocol_pairs:
        model_dict = model.dict()
        dataset_manager = experiment_setup_obj.dataset_manager
        data_models_model_path = os.path.join(
            project.DATA_MODELS_PATH, model_dict['name'])
        print("Testing {}, protocol {} ({})...".format(
            model_dict["name"], protocol, PROTOCOL_DICT[protocol]))
        model_dir = os.path.join(project.DATA_MODELS_PATH, model_dict["name"])
        if not os.path.exists(model_dir):
            raise FileNotFoundError
        if protocol == PROTOCOL_FIXED:
            testing_subject_folds = [dataset_manager.dataset_testing_subj]
        if protocol == PROTOCOL_CROSS_SUBJ:
            testing_subject_folds = [[s] for s in dataset_manager.subjects]
        for testing_fold in testing_subject_folds:
            fold_name = _get_fn_included_subj(model_dict, testing_fold)
            if not os.path.exists(os.path.join(model_dir, fold_name+".hdf5")):
                raise FileNotFoundError(
                    "Weights for fold {} not found".format(fold_name))
            if not os.path.exists(os.path.join(data_models_model_path, project.TESTING_RESULTS_FILE_PATTERN.format(fold_name))):
                corresponding_testing_fold = list(
                    set(dataset_manager.subjects)-set(testing_fold))
                mu, sigma = get_mean_std(
                    dataset_path=dataset_manager.dataset_path, subjects=corresponding_testing_fold)
                result = model_test(fold_name=fold_name, model_dir=model_dir, model_dict=model_dict, dataset_path=dataset_manager.dataset_path,
                                    testing_subj=testing_fold, mu=mu, sigma=sigma)
            else:
                print("Skipping model {}, use a flag to retest or delete files manually if necessery.".format(
                    fold_name))
    return True


def model_train(fold_name, model_dir, model_dict, dataset_path, development_subj, mu, sigma):
    """1 Setup_dic > model 
    2 Data generators OK
    3 ATTA callback IGNORE
    3 losses OK
    5 metrics : IGNORE ATTA, callbacks OK, optimizer
    6 setup"""
    generators = TXT_Train_Validation_Generators(dataset_path=dataset_path, subject_list=development_subj, train_size=model_dict["train_set_ratio"], frames_before=model_dict[
                                                 "frames"]-model_dict["frame_shift"], frames_after=model_dict["frame_shift"], view_IDs=model_dict["view_IDs"], batch_size=model_dict["batch_size"], mu=mu, sigma=sigma, shuffle=True)
    train_gen, valid_gen = generators.get_train(), generators.get_valid()
    losses = Losses_Keras(
        frames=model_dict['frames'], frame_shift=model_dict['frame_shift'])
    loss_fnc = losses.get_by_name(model_dict["loss_function"])
    ap_metrics = [AUC_AP(), Accuracy_AP(), Precision_AP(),
                  Recall_AP(), PrecisionAtRecall_AP(0.8)]
    fp_hdf5 = os.path.join(model_dir, fold_name+".hdf5")
    mcp = ModelCheckpoint(fp_hdf5, monitor='val_loss', verbose=True,
                          save_best_only=False, save_weights_only=True)
    metrics = ap_metrics
    callbacks = [mcp]
    optimizer = "adam"
    epochs = model_dict["epochs"]
    #### 1
    compile_kwargs = {"loss": loss_fnc,
                      "optimizer": optimizer, "metrics": metrics}
    fit_kwargs = {"x": train_gen, "epochs": epochs,
                  "validation_data": valid_gen, "callbacks": callbacks}
    Setup = SETUP_DIC[model_dict["architecture"]]
    setup = Setup(name=model_dict["name"], compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs,
                  TPA_view_IDs=model_dict['view_IDs'])
    # setup.delete_existing_model_data_and_output()
    print(setup.model.summary())
    setup.train()
    setup.write_architecture()
    # setup.plot_metrics(plot_val_metrics=valid_gen)
    #### /1
    #### 2
    # Get optimal threshold.
    print("Getting optimal threshold...")
    # https://support.sas.com/en/books/reference-books/analyzing-receiver-operating-characteristic-curves-with-sas/review.html
    # Gonen, Mithat. 2007. Analyzing Receiver Operating Characteristic Curves with SAS. Cary, NC: SAS Institute Inc.
    preds_list, trues_list = [], []
    generators = [train_gen, valid_gen] if valid_gen else [train_gen]
    for generator in generators:
        for i in range(len(generator)):
            x, y = generator[i]
            preds_list.append(setup.model.predict(x))
            trues_list.append(y)
    preds = np.vstack(preds_list)
    trues = np.vstack(trues_list)
    labels_dict, predictions_dict = {}, {}
    for idx, l in enumerate(zip(preds, trues)):
        pred, true = l
        predictions_dict[idx] = pred[:, 1]
        sample_class = true[-1][-1]
        labels_dict[idx] = model_dict["frames"] - \
            model_dict["frame_shift"] if sample_class else -1
    prc_pre_fpr, prc_pre_tpr, prc_pre_thresholds = plots.prediction_pr_curve(
        labels_dict, predictions_dict)
    # get optimal threshold
    fpr, tpr, thresh = prc_pre_fpr[:-1], prc_pre_tpr[:-1], prc_pre_thresholds
    xy = np.stack([fpr, tpr]).T
    ideal = np.array([1, 1])
    d = ideal-xy
    D = (d*d).sum(axis=-1)
    optimal_threshold = thresh[D.argmin()]
    with open(os.path.join(setup.data_models_model_path, project.THRESHOLD_FILE_PATTERN.format(fold_name)), "wb") as f:
        pickle.dump(optimal_threshold, f)
    #### /2
    print("Trained {}".format(model_dict["name"]))
    clear_session()
    return True


def train(experiment_setup_obj):
    for (model, protocol) in experiment_setup_obj.model_protocol_pairs:
        model_dict = model.dict()
        dataset_manager = experiment_setup_obj.dataset_manager
        print("Training {}, protocol {}".format(model_dict["name"], protocol))
        model_dir = os.path.join(project.DATA_MODELS_PATH, model_dict["name"])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if protocol == PROTOCOL_FIXED:
            devel_subject_folds = [dataset_manager.dataset_development_subj]
        if protocol == PROTOCOL_CROSS_SUBJ:
            _l = dataset_manager.subjects
            devel_subject_folds = [_l[:i]+_l[i+1:] for i in range(len(_l))]
        for fold in devel_subject_folds:
            fold_name = _get_fn_excluded_subj(
                model_dict, fold, dataset_manager.subjects)
            if not os.path.exists(os.path.join(model_dir, fold_name+".hdf5")):
                print("Fold: {}".format(fold_name))
                mu, sigma = get_mean_std(
                    dataset_path=dataset_manager.dataset_path, subjects=fold)
                result = model_train(fold_name=fold_name, model_dir=model_dir, model_dict=model_dict, dataset_path=dataset_manager.dataset_path,
                                     development_subj=fold, mu=mu, sigma=sigma)
            else:
                print("Skipping model {}, use a flag to retrain or delete files manually if necessery.".format(
                    fold_name))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--all', action='store_true')
    FLAGS = parser.parse_args()
    experiment_setup = configure_experiments(
        dataset_path="/media/igor/DATA/D01_MFIR-AP-Dataset")
    experiment_setup.print_summary()
    model = experiment_setup.models[0]
    model_dict = model.dict()
    if FLAGS.train or FLAGS.all:
        train(experiment_setup)
    if FLAGS.test or FLAGS.all:
        test(experiment_setup)
    if FLAGS.report or FLAGS.all:
        report(experiment_setup)
    if FLAGS.combine or FLAGS.all:
        combine(experiment_setup)
