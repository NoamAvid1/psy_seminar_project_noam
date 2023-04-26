import os
import pandas as pd
from plot_utils import get_rdm_fig,get_rocs_new_style
from collections import namedtuple

RESULTS_LAYER = "fc7"

# define named tuples for networks:
Net = namedtuple("Net", ["net_name", "architecture", "results_layer"])
faces_net = Net("faces", "vgg16", "fc7")
hands_both_net = Net("hands_both", "vgg16", "fc7")
objects_net = Net("objects", "vgg16", "fc7")
dorsal_right_hands_net = Net("dorsal_right_hands", "resnet50", "resnet_last_fc")
net_types = [faces_net, dorsal_right_hands_net, hands_both_net]

rotation_state = ["Upright", "Inverted"]
datasets_titles = {"dorsal_right_hands": "Dorsal Right Hands", "objects": "Objects",
                   "faces": "Faces"}
# datasets_titles = {"dorsal_right_hands": "Dorsal Right Hands", "both_hands": "Both Hands",
#                    "faces": "Faces", "objects": "Objects"}
rdm_figures = []
roc_figures = []
paths_for_rocs = {} # list to hold the path of upright and inverted
results_folder_path = fr'/galitylab/students/Noam/Seminar_2022/results/experiment_1'


def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w+')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


for dataset in datasets_titles:
    dataset_title_name = datasets_titles[dataset]
    for net in net_types:
        fig_name = f"{net.net_name,dataset_title_name}"
        paths_for_rocs[fig_name] = []
        for state in rotation_state:
            old_folder_path = fr'/galitylab/students/Noam/Seminar_2022/RDM/{net.net_name}/{net.architecture}/results/{dataset}_{state.lower()}'
            result_files = [f for f in os.listdir(old_folder_path)]
            try:
                is_results_layer = lambda x: x.endswith(net.results_layer + ".csv")
                results_file = list(filter(is_results_layer, result_files))[0]
            except:
                print(f"Error-dataset no '{net.results_layer}' file in directory {old_folder_path}")
            old_file_path = os.path.join(old_folder_path, results_file)
            paths_for_rocs[fig_name].append(old_file_path)
            file_data = pd.read_csv(old_file_path)
            # rdm_fig = get_rdm_fig(file_data, state, dataset_title_name, net.net_name)
            # rdm_figures.append(rdm_fig)
            # roc_fig = get_roc_fig(file_data, state, dataset_title_name, net.net_name)
            # roc_figures.append(roc_fig)

if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)
rdm_file_path = os.path.join(results_folder_path, RESULTS_LAYER + "_RDM.html")
figures_to_html(rdm_figures, rdm_file_path)
# roc_file_path = os.path.join(results_folder_path, RESULTS_LAYER + "_ROC.html")
# figures_to_html(roc_figures, roc_file_path)
rocs_file_path = os.path.join(results_folder_path, RESULTS_LAYER + "_roc.png")
get_rocs_new_style(paths_for_rocs, list(map(lambda x: x.net_name, net_types)), list(datasets_titles.values())  , rocs_file_path)





