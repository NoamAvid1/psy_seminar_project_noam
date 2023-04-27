import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import stat_utils


def get_roc_fig(file_data, state, dataset_title_name, net_type):
    # Setting up data arrays for plotting:
    data_arr = np.array(file_data)[:, 1:]  # array of the csv data (not including title)
    data_vector_shape = (data_arr.shape[0] * data_arr.shape[1] - data_arr.shape[
        0],)  # desired vector size is (160X160) - 160, Since we will delete the main diagonal
    # create vectors for roc plotting:
    same_pair_mat = np.ones((2, 2))
    y_test = np.kron(np.eye((data_arr.shape[0] // 2), dtype=int),
                     same_pair_mat)  # y_test = block diagonal matrix, each block is 2X2 matrix of "1"s.
    # delete main diagonal (not comparing a picture to itself!):
    y_test = y_test[~np.eye(y_test.shape[0], dtype=bool)].reshape(y_test.shape[0], -1)
    # reshape y_test into a vector:
    y_test = y_test.reshape(*data_vector_shape)
    arr_norm = np.linalg.norm(data_arr)  # compute std
    # noramlized the distances and get the "similarity" score:
    probs = (1 - data_arr / arr_norm)
    probs = probs[~np.eye(probs.shape[0], dtype=bool)].reshape(probs.shape[0], -1)
    # reshape probabilities into a vector:
    probs = probs.reshape(*data_vector_shape)

    # Compute ROC curve and area the curve:
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ### print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve:
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve for {state} {dataset_title_name} <br>net type = {net_type}, AUC={auc(fpr, tpr):.4f}',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig


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
    rdms = make_subplots(rows=num_rows, cols=num_cols,
                         subplot_titles=paths_dict.keys(),
                         x_title='Dataset', y_title='Model domain',
                         row_titles=nets_titles,
                         column_titles=datasets_titles,
                         vertical_spacing=0.05,
                         row_heights=num_cols * [100])
    for indx, domain_and_dataset in enumerate(list(paths_dict.keys())):
        for state_index, path in enumerate(paths_dict[domain_and_dataset]):  # iterate over upright and inverted
            df = pd.read_csv(path)
            rdms.add_trace(go.Heatmap(z=df),
                           row=1 + (indx // num_rows ), col=1 + (indx % num_rows) + state_index)
            rdms.update_layout(height=rdm_size * num_rows, width=rdm_size * num_cols, template='plotly_white')
    rdms.write_image(results_path)
    print(f"wrote RDMs to path {results_path}")


def plot_roc(paths_dict, nets_titles, datasets, results_path):
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
    datasets_titles = map(lambda x: x.dataset_title, datasets)
    rocs = make_subplots(rows=num_nets, cols=num_datasets,
                         subplot_titles=paths_dict.keys(),
                         x_title='Dataset', y_title='Model domain',
                         row_titles=nets_titles,
                         column_titles=datasets_titles,
                         vertical_spacing=0.05,
                         row_heights=num_datasets * [100])
    for i, domain in enumerate(nets_titles):
        for j, dataset in enumerate(datasets):
            for k, path in enumerate(paths_dict[(domain, dataset.title)]):  # iterate over upright and inverted
                df = pd.read_csv(path)
                verification_dist_list = stat_utils.rdm_to_dist_list(df, dataset.image_to_class_map_path)
                fpr, tpr, thresh, roc_auc = stat_utils.calc_graph_measurements(verification_dist_list, 'same', 'cos')
                rocs.add_trace(
                    go.Scatter(x=fpr, y=tpr, name=f'{domain} net, {dataset.title} dataset. AUC={roc_auc}', mode='lines',
                               line=line_designs[k % 2]),
                    row=1 + i, col=1 + j)
    for i in range(num_nets):
        for j in range(num_datasets):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1 + j, col=1 + i)
    rocs.update_layout(height=2 * 750, width=5 * 750 // 2, template='plotly_white', showlegend=False)
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.show()
    rocs.write_image(results_path)
    print(f"wrote ROCs to path {results_path}")
