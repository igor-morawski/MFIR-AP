import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# https://stackoverflow.com/questions/40865645/confusion-about-precision-recall-curve-and-average-precision

# vvv sortable table vvv
# https://kryogenix.org/code/browser/sorttable/sorttable.js

lim_eps = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()
    models = args.input

    # ablation info
    ablation_dict = {}

    ablation_dict[""] = "No ablation, all models included."
    ablation_dict["1"] = "Frame number and shift ablation."
    ablation_dict["A2"] = "View ablation, frame number and shift A."
    ablation_dict["B2"] = "View ablation, frame number and shift B."
    ablation_dict["C2"] = "View ablation, frame number and shift C."
    ablation_dict["D2"] = "View ablation, frame number and shift D."

    ablations = list(ablation_dict.keys())
    ablations.sort()
    # /ablation info

    reporting_dir = os.path.join("data", "06_reporting", "comparison")
    html_path = os.path.join(reporting_dir, "report.html")
    plots_fps = []

    reporting_dir = os.path.join("data", "06_reporting", "comparison")
    if not os.path.exists(reporting_dir):
        os.mkdir(reporting_dir)

    plots_dir = os.path.join(reporting_dir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    plots2rm = glob.glob(os.path.join(plots_dir, "*"))  # no subdirectories
    [os.remove(f) for f in plots2rm]

    models_data_dict = {}
    for model in models:
        with open(os.path.join("data", "06_reporting", model, model+".pkl"), 'rb') as f:
            d = pickle.load(f)
        models_data_dict[model] = copy.deepcopy(d)
        models_data_dict[model]["metrics_dict"]["name"] = model
        models_data_dict[model]["is_alias"] = False
        # add ablation *1x -> *2
        if model[-1] == "1":
            alias = model[:-1]+"2crl"
            models_data_dict[alias] = copy.deepcopy(d)
            models_data_dict[alias]["metrics_dict"]["name"] = alias
            models_data_dict[alias]["is_alias"] = True
    for key in models_data_dict.keys():
        if key not in models:
            models.append(key)

    

    # ROC, PRC, ATPC, ATRC
    rocs, det_rocs, prc, det_prc = [{} for i in range(4)]
    aptpc, adtpc, aptrc, adtrc = [{} for i in range(4)]
    min_avg_ts, max_avg_ts = {}, {}
    for model in models:
        rocs[model] = models_data_dict[model]['plots_dict']['roc']
        det_rocs[model] = models_data_dict[model]['plots_dict']['detection_roc']
        prc[model] = models_data_dict[model]['plots_dict']['precision_recall_curve']
        det_prc[model] = models_data_dict[model]['plots_dict']['detection_precision_recall_curve']
        aptpc[model] = models_data_dict[model]['plots_dict']['apt_precision_curve']
        adtpc[model] = models_data_dict[model]['plots_dict']['adt_precision_curve']
        aptrc[model] = models_data_dict[model]['plots_dict']['apt_recall_curve']
        adtrc[model] = models_data_dict[model]['plots_dict']['adt_recall_curve']
        min_avg_ts[model] = models_data_dict[model]['plots_dict']['min_avg_t']
        max_avg_ts[model] = models_data_dict[model]['plots_dict']['max_avg_t']

    # assert all ts are equal, tolerate roundoff errors etc
    _minl = np.array([v for v in min_avg_ts.values()])
    _maxl = np.array([v for v in max_avg_ts.values()])
    assert_d = 1
    assert round(_minl.min(), assert_d) == round(_minl.max(), assert_d)
    assert round(_maxl.min(), assert_d) == round(_maxl.max(), assert_d)
    min_avg_ts, max_avg_t = _minl.min(), _maxl.min()


    # plot settings
    from matplotlib.font_manager import FontProperties
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
        plt.xlim(0, 1+lim_eps)
        plt.ylim(0, 1+lim_eps)

    for ablation in ablations:
        plt.title('ROC - prediction {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = rocs[model]
            fpr, tpr = d['fpr'], d['tpr']
            plt.plot(fpr, tpr, label=model+", AUC = {:.2f}".format(models_data_dict[model]["metrics_dict"]["prediction_auc"]), **plot_args)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(**loc_args)
        set_lims()
        draw_iden_line()
        fp = os.path.join(plots_dir, "prediction_roc{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()

        plt.title('ROC - detection {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = det_rocs[model]
            fpr, tpr = d['fpr'], d['tpr']
            plt.plot(fpr, tpr, label=model+", AUC = {:.2f}".format(models_data_dict[model]["metrics_dict"]["detection_auc"]), **plot_args)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(**loc_args)
        set_lims()
        draw_iden_line()
        fp = os.path.join(plots_dir, "detection_roc{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()

        plt.title('Precision-recall curve - prediction {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = prc[model]
            precision, recall = d['precision'], d['recall']
            plt.plot(recall, precision, label=model, **plot_args)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(**loc_args)
        set_lims()
        draw_hline()
        fp = os.path.join(plots_dir, "prediction_prc{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()

        plt.title('Precision-recall curve - detection {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = det_prc[model]
            precision, recall = d['precision'], d['recall']
            plt.plot(recall, precision, label=model, **plot_args)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(**loc_args)
        set_lims()
        draw_hline()
        fp = os.path.join(plots_dir, "detection_prc{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()
    
        # APT-Recall
        plt.title('APT-recall curve - prediction {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = aptrc[model]
            apt, recall = d['apt'], d['recall']
            plt.plot(recall, apt, label=model, **plot_args)
        plt.xlabel('Recall')
        plt.ylabel('APT')
        plt.legend(**loc_args)
        draw_hline(y=max_avg_t)
        fp = os.path.join(plots_dir, "apt_recall_curve{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()
            
        # ADT-Recall
        plt.title('ADT-recall curve - detection {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = adtrc[model]
            adt, recall = d['adt'], d['recall']
            plt.plot(recall, adt, label=model, **plot_args)
        plt.xlabel('Recall')
        plt.ylabel('ADT')
        plt.legend(**loc_args)
        draw_hline(y=max_avg_t)
        fp = os.path.join(plots_dir, "adt_recall_curve{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()
    
        # APT-Precision
        plt.title('APT-recall curve - prediction {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = aptpc[model]
            apt, precision = d['apt'], d['precision']
            plt.plot(precision, apt, label=model, **plot_args)
        plt.xlabel('Precision')
        plt.ylabel('APT')
        plt.legend(**loc_args)
        draw_hline(y=max_avg_t)
        fp = os.path.join(plots_dir, "apt_precision_curve{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()
            
        # ADT-Precision
        plt.title('ADT-precision curve - detection {}'.format("(ablation {})".format(ablation) if ablation else "(all models)"))
        for model in models:
            if ablation not in model:
                continue
            d = adtpc[model]
            adt, precision = d['adt'], d['precision']
            plt.plot(precision, adt, label=model, **plot_args)
        plt.xlabel('Precision')
        plt.ylabel('ADT')
        plt.legend(**loc_args)
        draw_hline(y=max_avg_t)
        fp = os.path.join(plots_dir, "adt_precision_curve{}.png".format(ablation))
        plt.savefig(fp, bbox_inches='tight')
        plots_fps.append(os.path.relpath(fp, reporting_dir))
        plt.close()

        
    plots_fps.sort()
    # /plots

    report_data = {}
    plots_html = str(
        "".join(['''<img src="{}"><br>Figure {}<br>'''.format(e, i) for i, e in enumerate(plots_fps)]))
    report_data["plots_html"] = plots_html
    _links_fps = [os.path.join(
        "data", "06_reporting", model, "report.html") for model in models]
    links_fps = [os.path.relpath(l, reporting_dir) for l in _links_fps]
    links_html = str("".join(
        ['''<li> <a href="{}">{}</a></li>'''.format(l, m) for l, m in zip(links_fps, models)]))
    report_data["links_html"] = links_html

    sortable_header_keys = ['name', 'APT', 'ATTA',  'precision', 'recall',
                            'accuracy', 'detection_precision', 'detection_recall', 'detection_accuracy', "prediction_auc", "detection_auc"]
    row_pattern = "<tr>{}</tr>"
    cell_pattern = '<td width="{w}%">{brackets}</td>'.format(
        w=100//len(sortable_header_keys), brackets="{}")
    metric_cell_pattern = cell_pattern.format("{:.3f}")
    header_cell_pattern = "<th>{}</th>"

    def cell_formatter(x):
        try:
            return metric_cell_pattern.format(x)
        except ValueError:
            return cell_pattern.format(x)
    sortable_header = "".join(header_cell_pattern.format(h)
                              for h in sortable_header_keys)
    sortable_table_content = "".join([(row_pattern.format("".join(cell_formatter(
        models_data_dict[model]["metrics_dict"][key]) for key in sortable_header_keys))) if not models_data_dict[model]["is_alias"] else "" for model in models])

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

    ablations_list_html = "".join("<li>{} - {}</li>".format(ablation, ablation_dict[ablation]) if ablation else "" for ablation in ablations)
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
    with open(html_path, 'w') as f:
        f.write(report_html)
