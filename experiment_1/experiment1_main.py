import os
import pandas as pd
from plot_utils import plot_rdm,plot_roc
from collections import namedtuple

# define named tuples for networks:
Net = namedtuple("Net", ["title", "dir_name", "architecture", "results_layer"])
faces_net = Net("Faces", "faces_sampled_by_ratio","vgg16", "fc7")
objects_net = Net("Objects", "objects_sampled_by_ratio", "vgg16", "fc7")
dorsal_right_hands_net = Net("Dorsal Right Hands", "dorsal_right_hands", "resnet50", "resnet_last_fc")
hands_both_net = Net("Hands both", "hands_both", "vgg16", "fc7")
net_types = [faces_net, objects_net, dorsal_right_hands_net]
# net_types = [faces_net, objects_net, dorsal_right_hands_net, hands_both_net]

# define named tuples for datasets:
Dataset = namedtuple("Dataset", ["title", "dir_name", "image_to_class_map_path"])
faces_dataset = Dataset("Faces", "faces", "/galitylab/students/Noam/Datasets/100_faces_sampled_by_ratio/faces_val_img_to_class.csv")
objects_dataset = Dataset("Objects", "objects",  "/galitylab/students/Noam/Datasets/100_objects_sampled_by_ratio/objects_val_img_to_class.csv")
dorsal_right_hands_dataset = Dataset("Dorsal Right Hands","dorsal_right_hands", "/galitylab/students/Noam/Datasets/100_dorsal_right_hands/dorsal_right_hands_val_img_to_class.csv")
hands_both_dataset = Dataset("Hands both","hands_both", "/galitylab/students/Noam/Datasets/100_hands_both/both_hands_val_img_to_class.csv")
# datasets = [faces_dataset, objects_dataset, dorsal_right_hands_dataset, hands_both_dataset]
datasets = [faces_dataset, objects_dataset, dorsal_right_hands_dataset]

rotation_states = ["Upright", "Inverted"]
nets_titles = list(map(lambda x: x.title, net_types))
datasets_titles = list(map(lambda x: x.title, datasets))


def get_results_path():
    results_folder_path = fr'/galitylab/students/Noam/psy_seminar_project_noam/experiment_1/results/'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    rdm_file_path = os.path.join(results_folder_path, "RDM.png")
    roc_file_path = os.path.join(results_folder_path, "ROC.png")
    return rdm_file_path, roc_file_path


def get_paths_dict():
    paths_dict = {}  # list to hold the path of upright and inverted
    for dataset in datasets:
        for net in net_types:
            fig_name = (net.title, dataset.title)
            paths_dict[fig_name] = []
            for state in rotation_states:
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


def main():
    rdm_results_path, roc_results_path = get_results_path()
    csv_paths_dict = get_paths_dict()
    plot_roc(csv_paths_dict, nets_titles, datasets, roc_results_path)
    plot_rdm(csv_paths_dict, nets_titles, datasets_titles, rdm_results_path)


if __name__ == '__main__':
    main()


