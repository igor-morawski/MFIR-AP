import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# https://stackoverflow.com/questions/40865645/confusion-about-precision-recall-curve-and-average-precision

lim_eps = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()
    models = args.input

    reporting_dir = os.path.join("data", "06_reporting", "comparison")
    if not os.path.exists(reporting_dir):
        os.mkdir(reporting_dir)

    plots_dir = os.path.join(reporting_dir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plots2rm = glob.glob(os.path.join(plots_dir, "*")) # no subdirectories
    [os.remove(f) for f in plots2rm]

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

    loc_args = {'loc':'lower right'}
    plot_args = {'lw':2}
    
    GUIDELINES_KWARGS = {"color":'gray', "linestyle":'--'}
    def draw_hline(y=0.5):   
        plt.axhline(y=y, **GUIDELINES_KWARGS)
        return True
        
    def draw_iden_line():
        plt.plot([0, 1], [0, 1], **GUIDELINES_KWARGS)

    def set_lims():
        plt.xlim(0, 1+lim_eps)
        plt.ylim(0, 1+lim_eps)


    plt.title('ROC - prediction')
    for model in models:
        d = rocs[model]
        fpr, tpr = d['fpr'], d['tpr']
        plt.plot(fpr, tpr, label=model, **plot_args)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(**loc_args)
    set_lims()
    draw_iden_line()
    plt.savefig(os.path.join(plots_dir, "prediction_roc.png"))
    plt.close()


    plt.title('ROC - detection')
    for model in models:
        d = det_rocs[model]
        fpr, tpr = d['fpr'], d['tpr']
        plt.plot(fpr, tpr, label=model, **plot_args)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(**loc_args)
    set_lims()
    draw_iden_line()
    plt.savefig(os.path.join(plots_dir, "detection_roc.png"))
    plt.close()


    plt.title('Precision-recall curve - prediction')
    for model in models:
        d = prc[model]
        precision, recall = d['precision'], d['recall']
        plt.plot(recall, precision, label=model, **plot_args)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(**loc_args)
    set_lims()
    draw_hline()
    plt.savefig(os.path.join(plots_dir, "prediction_prc.png"))
    plt.close()


    plt.title('Precision-recall curve - detection')
    for model in models:
        d = det_prc[model]
        precision, recall = d['precision'], d['recall']
        plt.plot(recall, precision, label=model, **plot_args)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(**loc_args)
    set_lims()
    draw_hline()
    plt.savefig(os.path.join(plots_dir, "detection_prc.png"))
    plt.close()


    