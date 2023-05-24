from PIL import Image
import os

NEW_IMAGE_SIZE = (480, 360)


def change_dataset_image_size(old_dataset_path, new_dataset_path):
    for fold in ["train", "val"]:
        print(f"started {fold}")
        old_dir_path = os.path.join(old_dataset_path, fold)
        for cls_num, cls in enumerate(os.listdir(old_dir_path)):
            old_class_path = os.path.join(old_dir_path, cls)
            new_class_path = os.path.join(new_dataset_path, fold, cls)
            if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)
            images = os.listdir(old_class_path)
            for image in images:
                old_image_path = os.path.join(old_class_path, image)
                new_image_path = os.path.join(new_class_path, image[:-4] + ".png")
                new_image = Image.open(old_image_path)
                new_image = new_image.resize((480, 360))
                new_image.save(new_image_path)
        print(f"* {cls_num + 1}/{len(os.listdir(old_dir_path))} classes")


if __name__ == '__main__':
    change_dataset_background("/galitylab/students/Noam/Datasets/hands_both_bg",
                              "/galitylab/students/Noam/Datasets/hands_both_bg_small")
