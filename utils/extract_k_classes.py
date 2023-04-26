import os, os.path, shutil
import pandas as pd

# This program gets a path to a directory of dataset is P classes
# and some number k < P
# and creates a new dataset with the same k values for train and val
# also: rights to csv file what

K = 189
OLD_DATASET_PATH = "/galitylab/students/Noam/Datasets/faces/home/administrator/datasets/images_faces/faces_only_300/"
NEW_DATASET_PATH = "/galitylab/students/Noam/Datasets/faces/189_faces_only_300/"
CSV_PATH = f"{K}_faces_class_names.csv"

subject_ids = []
for subject in os.listdir(os.path.join(OLD_DATASET_PATH, "train")):
    if len(subject_ids) == K:
        break
    train_path = os.path.join(OLD_DATASET_PATH, "train", subject)
    val_path = os.path.join(OLD_DATASET_PATH, "val", subject)
    new_train_path = os.path.join(NEW_DATASET_PATH, "train", subject)
    new_val_path = os.path.join(NEW_DATASET_PATH, "val", subject)
    # check if class exists for both train and val:
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        continue
    # copy selected dirs
    shutil.copytree(train_path, new_train_path)
    shutil.copytree(val_path, new_val_path)
    # update subject ids
    subject_ids.append(subject)
    print(f"Copied {len(subject_ids)} classes")

print(f"Finished selecting classes. copied: {len(subject_ids)} classes")
# write selected ids to csv:
df = pd.DataFrame({"subject_id": subject_ids})
df.to_csv(CSV_PATH)



