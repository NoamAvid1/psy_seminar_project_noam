import os
from PIL import Image

# This files takes

ANGLE = 180
# dictionary of dataset name to [<upright_path>, <inverted_path>]
# datasets_names = {"faces": ["/galitylab/students/Noam/Datasets/100_faces_300_each/val",
#                         "/galitylab/students/Noam/Datasets/inverted/100_faces_300_each/val"],
#               "objects":["/galitylab/students/Noam/Datasets/100_objects_300_each/val",
#                          "/galitylab/students/Noam/Datasets/inverted/100_objects_300_each/val"],
#               "dorsal_right_hands":["/galitylab/students/Noam/Datasets/100_dorsal_right_hands/val",
#                                     "/galitylab/students/Noam/Datasets/inverted/100_dorsal_right_hands/val"]
#                   }
# datasets_names = {"faces_sampled": [
#     "/galitylab/students/Noam/Datasets/100_faces_sampled_by_ratio/val",
#     "/galitylab/students/Noam/Datasets/inverted/100_faces_sampled_by_ratio/val"
# ], "objects_sampled": ["/galitylab/students/Noam/Datasets/100_objects_sampled_by_ratio/val/",
#                        "/galitylab/students/Noam/Datasets/inverted/100_objects_sampled_by_ratio/val/"]}
datasets_names = {
    "hands_both": ["/galitylab/students/Noam/Datasets/100_hands_both/val/",
                   "/galitylab/students/Noam/Datasets/inverted/100_hands_both/val/"]
}
for dataset_name in datasets_names:
    # change old_folder_path:
    old_folder_path = datasets_names[dataset_name][0]
    new_folder_path = datasets_names[dataset_name][1]
    result_folders = [f for f in os.listdir(old_folder_path)]
    for folder in result_folders:
        old_class_path = os.path.join(old_folder_path, str(folder))
        new_class_path = os.path.join(new_folder_path, str(folder))
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path)
        result_files = [fi for fi in os.listdir(old_class_path)]
        for file_name in result_files:
            old_file_path = os.path.join(old_class_path, file_name)
            new_file_name = file_name
            new_file_path = os.path.join(new_class_path, new_file_name)
            new_image = Image.open(old_file_path)
            out = new_image.rotate(ANGLE)
            out.save(new_file_path)
            new_image.close()
