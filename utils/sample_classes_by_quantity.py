import pandas as pd
import os, os.path, shutil
import math

"""
This code takes a dataset and samples K classes from it (K is a parameter), 
and for each class samples specific number of images according to given quantities 
"""

# constants
quantities_file = "/galitylab/students/Noam/Datasets/100_dorsal_right_metadata/100_dorsal_right_metadata.csv"
num_classes = 100
train_val_ratio = 0.8 # train: 0.8, val: 0.2
datasets_to_sample = {"/galitylab/students/Noam/Datasets/189_faces/189_faces_only_300" : "/galitylab/students/Noam/Datasets/100_faces",
                      "/galitylab/students/Noam/Datasets/189_objects/": "/galitylab/students/Noam/Datasets/100_objects/"}


def copy_x_images(x, old_train_path, old_val_path, new_train_path, new_val_path):
    old_train_images = os.listdir(old_train_path)
    old_val_images = os.listdir(old_val_path)
    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)
        os.makedirs(new_val_path)
    num_train = math.ceil(train_val_ratio * x)
    for i in range(num_train):
        old_image_path = os.path.join(old_train_path, old_train_images[i])
        new_image_path = os.path.join(new_train_path, old_train_images[i])
        shutil.copy(old_image_path, new_image_path)
    for index, i in enumerate(range(num_train, x)):
        old_image_path = os.path.join(old_val_path, old_val_images[index])
        new_image_path = os.path.join(new_val_path, old_val_images[index])
        shutil.copy(old_image_path, new_image_path)
    print(f"num train: {num_train}, num val:{x - num_train}  \n")


df = pd.read_csv(quantities_file)
df = df.loc[(df["in final dataset?"] != "X")]
df.reset_index(inplace=True)

if df.shape[0] < num_classes:
    print("Error: less rows than classes")
    exit(0)

for dataset in datasets_to_sample.keys():
    print(f"copying for dataset:{dataset}")
    new_dataset = datasets_to_sample[dataset]
    classes = os.listdir(os.path.join(dataset, "train"))
    for i in range(num_classes):
        sampled_class = classes[i]
        old_train_path = os.path.join(dataset, "train", sampled_class)
        new_train_path = os.path.join(new_dataset, "train", sampled_class)
        old_val_path = os.path.join(dataset, "val", sampled_class)
        new_val_path = os.path.join(new_dataset, "val", sampled_class)
        num_images = df.iloc[i]["count"]
        num_images = num_images if num_images > 20 else 20  # number of images should be at least 20
        print(f"copying for class {sampled_class}, num images: {num_images}")
        copy_x_images(num_images, old_train_path, old_val_path, new_train_path, new_val_path)







        


