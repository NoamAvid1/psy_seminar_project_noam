import os
import numpy as np
import pandas as pd
from plot_utils import plot_rdm, plot_dist_mat_roc, plot_single_roc, plot_pairs_list_roc, plot_aucs_bar_chart
from collections import namedtuple
from datetime import datetime
import stat_utils

DATASET_NAME_IN_DIST_LIST = "hands_both_bg_unbalanced_upright"
NUM_BATCHES = 30
LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "pretrained"]


def get_results_path():
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    results_folder_path = fr'/galitylab/students/Noam/psy_seminar_project_noam/experiments_code/experiment2_results/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    bar_chart_file_path = os.path.join(results_folder_path, dt_string + "_experiment2.png")
    return bar_chart_file_path


def get_pretrained_domain_dists_path(domain):
    if domain == "faces":
        return "/galitylab/students/Noam/Seminar_2022/RDM/100_faces_resnet/resnet50/results/dists.csv"
    if domain == "objects":
        return "/galitylab/students/Noam/Seminar_2022/RDM/100_objects_resnet/resnet50/results/dists.csv"
    print("could not find pretrained file path")
    exit(0)


def get_pairs_lists_paths(domains: list[str] = ["objects", "faces"]):
    # layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]
    layers_paths = {}
    # add layers paths
    for domain in domains:
        layers_paths_list = list(map(lambda x: f"/galitylab/students/Noam/Seminar_2022/finetuning/hands_both_finetuning_{domain}_{x}/resnet50/results/dists.csv", LAYERS[:-1]))
        layers_paths[domain] = layers_paths_list
        layers_paths[domain].append(get_pretrained_domain_dists_path(domain))
    # layers paths is a dictionary such that
    # layers_paths["faces"] = [<path_to_conv1_dists>, ... <path_to_pretrained_dists>]
    # etc...
    return layers_paths


def get_dists_df(path):
    df = pd.read_csv(path)
    return stat_utils.pairs_list_to_dist_lis(df, DATASET_NAME_IN_DIST_LIST)


def get_precalculated_auc():
    domain_to_auc_mean = {'objects': [0.9960625000000001, 0.9966875000000001, 0.9916666666666668, 0.9856666666666666, 0.5730833333333333, 0.5426249999999999],
                          'faces': [0.9851041666666666, 0.9704166666666666, 0.931375, 0.8688541666666668, 0.5325833333333333, 0.5115624999999999]
                          }
    domain_to_auc_std = {'objects': [0.007310362935586723, 0.0073900799217599825, 0.009907558842733272, 0.01374804278999581, 0.07690547047440051, 0.0587622770860127],
                         'faces': [0.011352637511707255, 0.01712799041205815, 0.024225718083612433, 0.03533916151814521, 0.06415633033631382, 0.06068483945421076]
                         }
    return domain_to_auc_mean, domain_to_auc_std


def precalculated_main():
    domain_to_auc_mean, domain_to_auc_std = get_precalculated_auc()
    results_path =get_results_path()
    plot_aucs_bar_chart(LAYERS, domain_to_auc_mean, domain_to_auc_std, results_path)


def main():
    domains = ["objects", "faces"]
    dists_paths = get_pairs_lists_paths(domains)
    print(dists_paths)
    domain_to_auc_mean = {}
    domain_to_auc_std = {}
    for domain in domains:
        domain_dfs = list(map(get_dists_df, dists_paths[domain]))
        domain_to_auc_mean[domain] = []
        domain_to_auc_std[domain] = []
        for df in domain_dfs:
            auc_mean, auc_std = stat_utils.get_df_auc_mean_std(df)
            domain_to_auc_mean[domain].append(auc_mean)
            domain_to_auc_std[domain].append(auc_std)
    print("means: ", domain_to_auc_mean)
    print("aucs:", domain_to_auc_std)
    results_path = get_results_path()
    plot_aucs_bar_chart(LAYERS, domain_to_auc_mean, domain_to_auc_std, results_path)


if __name__ == '__main__':
    # precalculated_main()
    main()




