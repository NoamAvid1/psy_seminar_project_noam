import os
import pandas as pd
from plot_utils import plot_rdm, plot_dist_mat_roc, plot_single_roc, plot_pairs_list_roc, plot_single_roc_pairs_list
from collections import namedtuple
from datetime import datetime

ROTATION_STATES = ["Upright", "Inverted"]
PAIRS_CSV_TYPE = "pairs_list"  # "pairs_list" or "dist_mat".


def get_dists_mat_nets():
    # define named tuples for networks:
    Net = namedtuple("Net", ["title", "dir_name", "architecture", "results_layer"])
    faces_net = Net("Faces", "faces_sampled_by_ratio","vgg16", "fc7")
    objects_net = Net("Objects", "objects_sampled_by_ratio", "vgg16", "fc7")
    dorsal_right_hands_net = Net("Dorsal Right Hands", "dorsal_right_hands", "resnet50", "resnet_last_fc")
    hands_both_net = Net("Both Hands", "hands_both", "vgg16", "fc7")
    net_types = [faces_net, objects_net, dorsal_right_hands_net, hands_both_net]
    return net_types


def get_dists_mat_datasets():
    # define named tuples for datasets:
    Dataset = namedtuple("Dataset", ["title", "dir_name", "image_to_class_map_path"])
    faces_dataset = Dataset("Faces", "faces", "/galitylab/students/Noam/Datasets/100_faces_sampled_by_ratio/faces_val_img_to_class.csv")
    objects_dataset = Dataset("Objects", "objects",  "/galitylab/students/Noam/Datasets/100_objects_sampled_by_ratio/objects_val_img_to_class.csv")
    dorsal_right_hands_dataset = Dataset("Dorsal Right Hands","dorsal_right_hands", "/galitylab/students/Noam/Datasets/100_dorsal_right_hands/dorsal_right_hands_val_img_to_class.csv")
    hands_both_dataset = Dataset("Both hands","hands_both", "/galitylab/students/Noam/Datasets/100_hands_both/both_hands_val_img_to_class.csv")
    hands_both_bg_dataset = Dataset("Both hands","hands_both", "/galitylab/students/Noam/Datasets/100_hands_both/both_hands_val_img_to_class.csv")
    # datasets = [faces_dataset, objects_dataset, dorsal_right_hands_dataset, hands_both_dataset]
    datasets = [faces_dataset, objects_dataset, dorsal_right_hands_dataset, hands_both_bg_dataset]
    return datasets


def get_pairs_list_nets_and_datasets():
    ### experiment 1.1 ###
    # nets_to_csvs = {"faces": "/galitylab/students/Noam/Seminar_2022/RDM/100_faces_resnet/resnet50/results/dists.csv",
    #                 "objects": "/galitylab/students/Noam/Seminar_2022/RDM/100_objects_resnet/resnet50/results/dists.csv",
    #                 "hands both": "/galitylab/students/Noam/Seminar_2022/RDM/hands_both_bg/resnet50/results/dists.csv"}
    # datasets_names = ["100_faces_100_each", "100_objects_100_each", "hands_both_bg_unbalanced"]
    #### experiment 1.2: ###
    nets_to_csvs = {"DR hands": "/galitylab/students/Noam/Seminar_2022/RDM/50_dr_hands_bg/resnet50/results/dists.csv",
                    "Both hands": "/galitylab/students/Noam/Seminar_2022/RDM/50_hands_both_bg/resnet50/results/dists.csv"
                    }
    datasets_names = ["50_dr_hands_bg", "50_hands_both_bg"]
    return nets_to_csvs, datasets_names


def get_results_path():
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    results_folder_path = fr'/galitylab/students/Noam/psy_seminar_project_noam/experiments_code/experiment1_results/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    rdm_file_path = os.path.join(results_folder_path, dt_string + "_RDM.png")
    roc_file_path = os.path.join(results_folder_path, dt_string + "_ROC.png")
    aucs_file_path = os.path.join(results_folder_path, dt_string + "_AUCs.csv")
    return rdm_file_path, roc_file_path, aucs_file_path


def get_paths_dict(datasets, net_types):
    paths_dict = {}  # list to hold the path of upright and inverted
    for dataset in datasets:
        for net in net_types:
            fig_name = (net.title, dataset.title)
            paths_dict[fig_name] = []
            for state in ROTATION_STATES:
                old_folder_path = fr'/galitylab/students/Noam/Seminar_2022/RDM/{net.dir_name}/{net.architecture}/results/{dataset.dir_name}_{state.lower()}'
                result_files = [f for f in os.listdir(old_folder_path)]
                try:
                    is_results_layer = lambda x: x.endswith(net.results_layer + ".csv")
                    results_file = list(filter(is_results_layer, result_files))[0]
                except:
                    print(f"Error-dataset no '{net.results_layer}' file in directory {old_folder_path}")
                    exit(1)
                old_file_path = os.path.join(old_folder_path, results_file)
                paths_dict[fig_name].append(old_file_path)
    return paths_dict


def dist_mat_main():
    datasets = get_dists_mat_datasets()
    net_types = get_dists_mat_nets()
    nets_titles = list(map(lambda x: x.title, net_types))
    datasets_titles = list(map(lambda x: x.title, datasets))
    rdm_results_path, roc_results_path, aucs_results_path = get_results_path()
    csv_paths_dict = get_paths_dict(datasets, net_types)
    plot_dist_mat_roc(csv_paths_dict, nets_titles, datasets, roc_results_path)
    # plot_rdm(csv_paths_dict, nets_titles, datasets_titles, rdm_results_path)


def pairs_list_main():
    net_to_csv, dataset_names = get_pairs_list_nets_and_datasets()
    rdm_results_path, roc_results_path, aucs_results_path = get_results_path()
    plot_pairs_list_roc(net_to_csv, dataset_names, roc_results_path, aucs_results_path)


def single_roc_main():
    csv_paths = ["/galitylab/students/Noam/Seminar_2022/RDM/faces_sampled_by_ratio/vgg16/results/100_dr_hands_no_bg/dists.csv",
                 "/galitylab/students/Noam/Seminar_2022/RDM/faces_sampled_by_ratio/vgg16/results/50_dr_hands_bg/dists.csv"]
    dataset_names = ["100_dr_hands_no_bg", "50_dr_hands_bg"]
    _, results_path, _ = get_results_path()
    print("results path: ", results_path)
    plot_single_roc_pairs_list(csv_paths, dataset_names, results_path)


    

if __name__ == '__main__':
    # single_roc_main()
    if PAIRS_CSV_TYPE == "dist_mat":
        dist_mat_main()
    else:
        pairs_list_main()




