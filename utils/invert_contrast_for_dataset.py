import os
from PIL import Image, ImageOps

# This files takes a dataset and create a new dataset for which
# each image in the original dataset appears in reversed contrast


datasets_names = {
    "faces_train": ["/galitylab/students/Noam/Datasets/100_faces_100_each/train/",
                    "/galitylab/students/Noam/Datasets/contrast_reversal/100_faces_100_each/train/"],
    "faces_val": ["/galitylab/students/Noam/Datasets/100_faces_100_each/val/",
                  "/galitylab/students/Noam/Datasets/contrast_reversal/100_faces_100_each/val/"]
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
            out = ImageOps.invert(new_image)
            out.save(new_file_path)
            new_image.close()
