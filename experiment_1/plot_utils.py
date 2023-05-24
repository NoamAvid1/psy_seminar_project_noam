import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import stat_utils


def plot_rdm(paths_dict, nets_titles, datasets_titles, results_path):
    """
    :param paths_dict: dictionary, where keys are tuples of strings (net_domain,dataset_name) and value is a list of
    paths to csv file in both states (upright\inverted)
    :param nets_titles: list of titles
    :param datasets_titles: list of titles
    :param results_path: path to save the image of plots
    :return: None
    """
    rdm_size = 300
    num_rows = len(nets_titles)
    num_cols = len(datasets_titles) * 2
    subplots_titles = []
    for i in list(paths_dict.keys()):
        title = f'{str(i[0])} Net, {str(i[1])}'
        subplots_titles.append(title + " Upright")
        subplots_titles.append(title + " Inverted")
    rdms = make_subplots(rows=num_rows, cols=num_cols,
                         subplot_titles=subplots_titles,
                         vertical_spacing=0.05,
                         row_heights=num_rows * [100])
    for domain_index, domain in enumerate(nets_titles):
        for dataset_index, dataset in enumerate(datasets_titles):
            for state_index, state_path in enumerate(paths_dict[(domain, dataset)]):  # iterate over upright and inverted
                df = pd.read_csv(state_path)
                rdms.add_trace(go.Heatmap(z=df),
                               row=1 + domain_index, col=1 + 2 * dataset_index + state_index)
    rdms.update_layout(height=rdm_size * num_rows, width=rdm_size * num_cols, template='plotly_white')
    rdms.update_annotations(font_size=13)
    rdms.write_image(results_path)
    print(f"wrote RDMs to path {results_path}")


def plot_pairs_list_roc(net_to_csv: dict, datasets_names, results_path):
    """
        :param net_to_csv: list of paths of dists csv files, each csv file contains dists
        for the net for all
        :param nets_titles: list of titles
        :param datasets_names: list of Dataset names
        :param results_path: path to save the image of plots
        :return: None
        """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
    ]
    nets_titles = list(net_to_csv.keys())
    num_nets = len(nets_titles)
    num_datasets = len(datasets_names)
    rocs = make_subplots(rows=num_nets, cols=num_datasets,
                         x_title='Dataset', y_title='Model domain',
                         row_titles=nets_titles,
                         column_titles=datasets_names,
                         vertical_spacing=0.05,
                         row_heights=num_nets * [100])
    for domain_index, domain in enumerate(list(net_to_csv.keys())):
        df = pd.read_csv(net_to_csv[domain])
        for dataset_index, dataset in enumerate(datasets_names):
            for rotation_state in ["upright", "inverted"]:
                dataset_and_rotation = f"{dataset}_{rotation_state}"
                verification_dist_list = stat_utils.pairs_list_to_dist_list(df, dataset_and_rotation)
                fpr, tpr, thresh, roc_auc = stat_utils.calc_graph_measurements(verification_dist_list, 'same', 'cos')
                rocs.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{domain} net, {dataset.title} dataset. AUC={roc_auc}', mode='lines',
                               line=line_designs[0 if rotation_state == "upright" else 1]),
                    row=1 + domain_index, col=1 + dataset_index)
    for i in range(num_nets):
        for j in range(num_datasets):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1 + i, col=1 + j)
    rocs.update_layout(height=2 * 750, width=5 * 750 // 2, template='plotly_white', showlegend=False)
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.show()
    rocs.write_image(results_path)
    print(f"wrote ROCs to path {results_path}")


def plot_dist_mat_roc(paths_dict, nets_titles, datasets, results_path):
    """
    :param paths_dict: dictionary, where keys are tuples of strings (net_domain,dataset_name) and value is a list of
    paths to csv files in both states (upright\inverted)
    :param nets_titles: list of titles
    :param datasets: list of Dataset objects
    :param results_path: path to save the image of plots
    :return: None
    """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
    ]
    num_nets = len(nets_titles)
    num_datasets = len(datasets)
    datasets_titles = list(map(lambda x: x.title, datasets))
    subplots_titles = [str(i) for i in list(paths_dict.keys())]
    rocs = make_subplots(rows=num_nets, cols=num_datasets,
                         # subplot_titles=subplots_titles,
                         x_title='Dataset', y_title='Model domain',
                         row_titles=nets_titles,
                         column_titles=datasets_titles,
                         vertical_spacing=0.05,
                         row_heights=num_nets *  [100])
    for domain_index, domain in enumerate(nets_titles):
        for dataset_index, dataset in enumerate(datasets):
            for state_index, state_path in enumerate(paths_dict[(domain, dataset.title)]):  # iterate over upright and inverted
                df = pd.read_csv(state_path)
                verification_dist_list = stat_utils.rdm_to_dist_list(df, dataset.image_to_class_map_path)
                fpr, tpr, thresh, roc_auc = stat_utils.calc_graph_measurements(verification_dist_list, 'same', 'cos')
                rocs.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{domain} net, {dataset.title} dataset. AUC={roc_auc}', mode='lines',
                               line=line_designs[state_index % 2]),
                    row=1 + domain_index, col=1 + dataset_index)
    for i in range(num_nets):
        for j in range(num_datasets):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1 + i, col=1 + j)
    rocs.update_layout(height=2 * 750, width=5 * 750 // 2, template='plotly_white', showlegend=False)
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.show()
    rocs.write_image(results_path)
    print(f"wrote ROCs to path {results_path}")


def plot_single_roc(csv_paths, results_path,image_to_class_map_path ):
    """
    This function was created to add multiple curves on the same figure
    :param csv_paths: list of RDM dist mat paths as strings
    :param results_path: were to save results
    :param image_to_class_map_path: path of csv file that maps image to class name
    :return:
    """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
    ]
    rocs = make_subplots(rows=1, cols=1,
                         # subplot_titles=subplots_titles,
                         x_title='Dataset', y_title='Model domain',
                         vertical_spacing=0.05,
                         row_heights=1 * [100])
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        verification_dist_list = stat_utils.rdm_to_dist_list(df, image_to_class_map_path)
        fpr, tpr, thresh, roc_auc = stat_utils.calc_graph_measurements(verification_dist_list, 'same', 'cos')
        rocs.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines',
                       line=line_designs[0]),
            row=1, col=1)
    rocs.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1, row=1, col=1)
    rocs.update_layout(height=2 * 750, width=5 * 750 // 2, template='plotly_white', showlegend=False)
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.write_image(results_path)
    print(f"wrote ROCs to path {results_path}")
