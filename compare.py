import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

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
        models_data_dict[model] = d
        models_data_dict[model]["metrics_dict"]["name"] = model

    # ROC and PRC
    rocs, det_rocs, prc, det_prc = [{} for i in range(4)]
    for model in models:
        rocs[model] = models_data_dict[model]['plots_dict']['roc']
        det_rocs[model] = models_data_dict[model]['plots_dict']['detection_roc']
        prc[model] = models_data_dict[model]['plots_dict']['precision_recall_curve']
        det_prc[model] = models_data_dict[model]['plots_dict']['detection_precision_recall_curve']

    loc_args = {'loc': 'lower right'}
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
    fp = os.path.join(plots_dir, "prediction_roc.png")
    plt.savefig(fp)
    plots_fps.append(os.path.relpath(fp, reporting_dir))
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
    fp = os.path.join(plots_dir, "detection_roc.png")
    plt.savefig(fp)
    plots_fps.append(os.path.relpath(fp, reporting_dir))
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
    fp = os.path.join(plots_dir, "prediction_prc.png")
    plt.savefig(fp)
    plots_fps.append(os.path.relpath(fp, reporting_dir))
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
    fp = os.path.join(plots_dir, "detection_prc.png")
    plt.savefig(fp)
    plots_fps.append(os.path.relpath(fp, reporting_dir))
    plt.close()

    report_data = {}
    plots_html = str(
        "".join(['''<img src="{}"><br>'''.format(e) for e in plots_fps]))
    report_data["plots_html"] = plots_html
    _links_fps = [os.path.join(
        "data", "06_reporting", model, "report.html") for model in models]
    links_fps = [os.path.relpath(l, reporting_dir) for l in _links_fps]
    links_html = str("".join(
        ['''<li> <a href="{}">{}</a></li>'''.format(l, m) for l, m in zip(links_fps, models)]))
    report_data["links_html"] = links_html

    sortable_header_keys = ['name', 'ATTA', 'APT', 'precision', 'recall',
                            'accuracy', 'detection_precision', 'detection_recall', 'detection_accuracy', "prediction_auc", "detection_auc"]
    row_pattern = "<tr>{}</tr>"
    cell_pattern = '<td width="{w}%">{brackets}</td>'.format(w=100//len(sortable_header_keys), brackets="{}")
    metric_cell_pattern = cell_pattern.format("{:.3f}")
    header_cell_pattern = "<th>{}</th>"

    def cell_formatter(x):
        try:
            return metric_cell_pattern.format(x)
        except ValueError:
            return cell_pattern.format(x)
    sortable_header = "".join(header_cell_pattern.format(h) for h in sortable_header_keys)
    sortable_table_content = "".join([(row_pattern.format("".join(cell_formatter(
        models_data_dict[model]["metrics_dict"][key]) for key in sortable_header_keys))) for model in models])

    sortable_header = sortable_header.replace("_", " ")
    
    sortable_html = '''
    <table class="sortable" width="100%">
    <thead>
    <tr>{sortable_header}</tr>
    </thead>
    <tbody>
    {sortable_table_content}
    </tbody>
    </table>
    '''.format(**{"sortable_header":sortable_header, "sortable_table_content":sortable_table_content})
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
            
        </body>
    </html> 
    '''.format(**report_data)
    with open(html_path, 'w') as f:
        f.write(report_html)
