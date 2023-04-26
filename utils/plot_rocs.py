import mlflow
import const
import plotly as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score

NET_NAME = "faces"

def get_rdm_locs():
    # map net type to RDM locations for different datasets
    locs = {
        "faces":[
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/faces_upright/faces_upright_38.csv",
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/faces_inverted/faces_inverted_38.csv",
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/objects_upright/objects_upright_38.csv",
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/objects_inverted/objects_inverted_38.csv",
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/dorsal_right_hands_upright/dorsal_right_hands_upright_38.csv",
        "/galitylab/students/Noam/Seminar_2022/RDM/faces/vgg16/results/dorsal_right_hands_inverted/dorsal_right_hands_inverted_38.csv"]
    }
    return locs


def plot_rocs(rdm_paths):
    """ Plot the ROCs for the RDMs """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
        dict(color='royalblue', width=2, dash='dot'),
        dict(color='firebrick', width=2, dash='dot'),
    ]
    titles = []
    n_cols = 6
    n_rows = 1
    for i in range(n_cols):
        titles.append(f'{key}, {rdm_paths.index[i]} ROC')

    rocs = make_subplots(rows=1, cols=n_cols,
                         x_title='dataset type', y_title='',
                         row_titles=['Individual birds verification', 'Bird species verification'], column_titles=list(rdm_paths.index))
                         # vertical_spacing=0.5)

    for i, idx in enumerate(rdm_paths.index):
        for j, col in enumerate(rdm_paths.columns):
            curr_path = rdm_paths.iloc[i][col]
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            verification_dist_list = rdm_to_verification_dist_list(df)
            fpr, tpr, thresh, roc_auc = calc_graph_measurements(verification_dist_list, 'same', 'cos')
            name = f'{col}, {idx}'
            rocs.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{name}. AUC={roc_auc}', mode='lines', line=line_designs[j % 2]),
                row=1 + (j // 2), col=1 + i)
    for i in range(9):
        for j in range(2):
            rocs.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1, row=1 + j, col=1 + i)

    rocs.update_layout(height=750, width=9*750//2, template='plotly_white')
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.update_yaxes(range=[0.0, 1.0])
    rocs.show()
    rocs.write_html(f'/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/roc_for_{faces}_net.html')

def plot_aucs(rdm_paths):
    """ Create line plot of AUC as function of finetuning depth """
    line_designs = [
        dict(color='royalblue', width=2),
        dict(color='firebrick', width=2),
        dict(color='royalblue', width=2, dash='dot'),
        dict(color='firebrick', width=2, dash='dot'),
    ]
    aucs = {}
    for j, col in enumerate(rdm_paths.columns):
        aucs[col] = []
        for i, idx in enumerate(rdm_paths.index):
            curr_path = rdm_paths.iloc[i][col]
            df = pd.read_csv(curr_path, index_col='Unnamed: 0')
            verification_dist_list = rdm_to_verification_dist_list(df)
            fpr, tpr, thresh, roc_auc = calc_graph_measurements(verification_dist_list, 'same', 'cos')
            aucs[col].append(roc_auc)
    aucs = pd.DataFrame(aucs, index = rdm_paths.index)
    fig = go.Figure()
    for i, col in enumerate(aucs.columns):
        fig.add_trace(go.Scatter(x=aucs.index, y=aucs[col],
                                 mode='lines',
                                 name=col, line=line_designs[i]))

    fig.update_layout(
        title="Verification AUC as function of finetuning depth",
        # xaxis_title="Finetune depth",
        # yaxis_title="Verification AUC",
        legend_title="Legend",
        template='plotly_white'
    )
    fig.update_yaxes(range=[0.0, 1.0])

    fig.show()
    aucs.to_csv(f'/galitylab/students/Noam/Seminar_2022/AUC/verification_AUC_for_')
    fig.write_html('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/verification_AUC_over_finetune_depth_range01_16022022.html')

if __name__ == '__main__':
    locations = get_rdm_locs()
    plot_rocs(locations)
    #plot_aucs(locations)
