import os, os.path, shutil
import pandas as pd

folders_and_quantities = {
    "100_hands_both": 100,
    "hands_both_unseen": 36
}
# dorsal_dataset_path = "/galitylab/students/Noam/Datasets/100_dorsal_right_hands"
original_dataset_path = "/galitylab/students/Noam/Datasets/189_hands_both/"
dataset_classes = os.listdir(os.path.join(original_dataset_path, "val"))
datasets_path = "/galitylab/students/Noam/Datasets"
metadata_path = "/galitylab/students/Noam/Datasets/100_dorsal_right_hands/metadata/100_dorsal_right_metadata.csv"
df = pd.read_csv(metadata_path)
df = df.set_index("id")
df_row_count = 0

for folder in list(folders_and_quantities.keys()):
    path = os.path.join(datasets_path, folder)
    for i in range(folders_and_quantities[folder]):
        curr_class = dataset_classes[df_row_count]
        while curr_class in df.index:
            df_row_count += 1
            curr_class = dataset_classes[df_row_count]
        df_row_count += 1
        old_train_path = os.path.join(original_dataset_path, "train", curr_class)
        old_val_path = os.path.join(original_dataset_path, "val", curr_class)
        new_train_path = os.path.join(path, "train", curr_class)
        new_val_path = os.path.join(path, "val", curr_class)
        shutil.copytree(old_train_path, new_train_path)
        shutil.copytree(old_val_path, new_val_path)
    print(f"copied {folders_and_quantities[folder]} classes to {new_train_path}")
