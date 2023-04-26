import os
import pandas as pd
from plot_utils import plot_rdm,plot_roc
from collections import namedtuple

RESULTS_LAYER = "fc7"

# define named tuples for networks:
Net = namedtuple("Net", ["net_name", "architecture", "results_layer"])
faces_net = Net("faces", "vgg16", "fc7")
hands_both_net = Net("hands_both", "vgg16", "fc7")
objects_net = Net("objects", "vgg16", "fc7")
dorsal_right_hands_net = Net("dorsal_right_hands", "resnet50", "resnet_last_fc")
net_types = [faces_net, objects_net, dorsal_right_hands_net, hands_both_net]

rotation_states = ["Upright", "Inverted"]
datasets_titles = {"faces": "Faces", "objects": "Objects", "dorsal_right_hands": "Dorsal Right Hands"}
# datasets_titles = {"faces": "Faces", "objects": "Objects", "dorsal_right_hands": "Dorsal Right Hands",
# "hands_both": "Hands Both"}


def get_results_path():
    results_folder_path = fr'/galitylab/students/Noam/Seminar_2022/results/experiment_1'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    rdm_file_path = os.path.join(results_folder_path, RESULTS_LAYER + "_RDM.png")
    roc_file_path = os.path.join(results_folder_path, RESULTS_LAYER + "_ROC.png")
    return rdm_file_path, roc_file_path


def get_paths_dict():
    paths_dict = {}  # list to hold the path of upright and inverted
    for dataset in datasets_titles:
        dataset_title_name = datasets_titles[dataset]
        for net in net_types:
            fig_name = (net.net_name,dataset_title_name)
            paths_dict[fig_name] = []
            for state in rotation_states:
                old_folder_path = fr'/galitylab/students/Noam/Seminar_2022/RDM/{net.net_name}/{net.architecture}/results/{dataset}_{state.lower()}'
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


def main():
    rdm_file_path, roc_file_path = get_results_path()
    paths_dict = get_paths_dict()
    plot_roc(paths_dict, list(map(lambda x: x.net_name, net_types)), list(datasets_titles.values()), roc_file_path)
    plot_rdm(paths_dict, list(map(lambda x: x.net_name, net_types)), list(datasets_titles.values()), rdm_file_path)


if __name__ == '__main__':
    main()


