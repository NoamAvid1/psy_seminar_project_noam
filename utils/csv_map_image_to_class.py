import pandas as pd
import os
from collections import namedtuple

Dataset = namedtuple("Dataset", ["title", "path"])

# DATASETS_PATHS = [
#             Dataset("faces", "/galitylab/students/Noam/Datasets/100_faces_sampled_by_ratio"),
#             Dataset("objects", "/galitylab/students/Noam/Datasets/100_objects_sampled_by_ratio"),
#             Dataset("dorsal_right_hands", "/galitylab/students/Noam/Datasets/100_dorsal_right_hands"),
#             Dataset("both_hands", "/galitylab/students/Noam/Datasets/100_hands_both")
# ]
DATASETS_PATHS = [Dataset("both_hands_bg_5", "/galitylab/students/Noam/Datasets/hands_both_bg_5_each/")]


def map_image_to_class():
    for dataset in DATASETS_PATHS:
        df = pd.DataFrame(columns=["class"])
        validation_path = os.path.join(dataset.path, "val")
        classes = os.listdir(validation_path)
        for cls in classes:
            images = os.listdir(os.path.join(validation_path, cls))
            for img in images:
                df.loc[img] = cls
        results_path = os.path.join(dataset.path, f'{dataset.title}_val_img_to_class.csv')
        if os.path.exists(results_path):
            os.remove(results_path)
        df.to_csv(results_path)


if __name__ == '__main__':
    map_image_to_class()


