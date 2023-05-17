import os, shutil, random

NUM_TRAIN_TO_COPY = 100
NUM_VAL_TO_COPY = 25
old_paths = ["/galitylab/students/Noam/Datasets/100_objects_300_each/",
             "/galitylab/students/Noam/Datasets/100_faces_300_each/"]
new_paths = ["/galitylab/students/Noam/Datasets/100_objects_100_each/",
             "/galitylab/students/Noam/Datasets/100_faces_100_each/"]


def copy_quantity(old_folder_path, new_folder_path, num_to_copy):
    os.makedirs(new_folder_path, exist_ok=True)
    copied = 0
    images = os.listdir(old_folder_path)
    random.shuffle(images)
    for image in images:
        if copied >= num_to_copy:
            break
        old_image_path = os.path.join(old_folder_path, image)
        new_image_path = os.path.join(new_folder_path, image)
        shutil.copy(old_image_path, new_image_path)
        copied += 1


def main():
    for old_path, new_path in list(zip(old_paths, new_paths)):
        classes = os.listdir(os.path.join(old_path, "train"))
        for folder in classes:
            old_train_path = os.path.join(old_path, "train", folder)
            new_train_path = os.path.join(new_path, "train", folder)
            copy_quantity(old_train_path, new_train_path, NUM_TRAIN_TO_COPY)
            old_val_path = os.path.join(old_path, "val", folder)
            new_val_path = os.path.join(new_path, "val", folder)
            copy_quantity(old_val_path, new_val_path, NUM_VAL_TO_COPY)


if __name__ == '__main__':
    main()
