from randimage import get_random_image
from PIL import Image
import matplotlib
import os
import random
import numpy as np

BACKGROUND_WIDTH, BACKGROUND_HEIGHT = (1600, 1200) # size of new image in the dataset
NUM_TRAIN = 100 # number of images per class in train folder of the new dataset
NUM_VAL = 25    # number of images per class in val folder
BRIGHTNESS_CRITERIA = [215, 230, 235, 245, 250]  # how strictly we crop corners of image
BACKGROUND_IMAGE_PATH = '/galitylab/students/Noam/psy_seminar_project_noam/data_preprocessing/random_background.jpg'


def get_random_background():
    """
    generates random background image and saves
    """
    image = get_random_image((BACKGROUND_HEIGHT // 16, BACKGROUND_WIDTH // 16))
    image = np.repeat(image, 16, axis=0) # for efficiency - generate small image and then rescale it
    image = np.repeat(image, 16, axis=1)
    matplotlib.image.imsave(BACKGROUND_IMAGE_PATH, image)


def get_random_object_size(width, height):
    ratio = random.uniform(0.4,1)
    return int(width * ratio), int(height * ratio)


def get_random_object_location(background_image):
    width_loc = random.randint(0,4)
    height_loc = random.randint(0,3)
    width = background_image.width // 10 * width_loc
    height = background_image.height // 10 * height_loc
    return width, height


def change_image_background(old_image_path, background_path, new_image_path, brightness_criterion):
    """
        Takes image of object with white background
        crops the object from the background
        randomly resizes the objects
         and pastes them on the background
         :param brightness_criterion: value of pixels (between 235-250) from which we crop the bright edges of the object
        """
    # process front image:
    front_image = Image.open(old_image_path)
    front_image_size = get_random_object_size(width=front_image.width, height=front_image.height)
    front_image.thumbnail(front_image_size)
    front_image = front_image.convert("RGBA")
    # process background:
    background_image = Image.open(background_path)
    background_image.thumbnail((BACKGROUND_WIDTH, BACKGROUND_HEIGHT))
    background_image = background_image.convert("RGBA")

    datas = front_image.getdata()
    new_data = []
    for item in datas:
        if item[0] > brightness_criterion and item[1] > brightness_criterion and item[2] > brightness_criterion:  # finding white color by its RGB value
            # storing a transparent value when we find a white colour
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)  # other colours remain unchanged

    front_image.putdata(new_data)

    width, height = get_random_object_location(background_image)

    background_image.paste(front_image, (width, height), front_image)
    background_image.save(new_image_path, format="png")


def change_dataset_background(old_dataset_path, new_dataset_path):
    for fold in ["train", "val"]:
        print(f"started {fold}")
        num_to_sample = NUM_TRAIN if fold == "train" else NUM_VAL
        old_dir_path = os.path.join(old_dataset_path, fold)
        for cls_num, cls in enumerate(os.listdir(old_dir_path)):
            old_class_path = os.path.join(old_dir_path, cls)
            new_class_path = os.path.join(new_dataset_path, fold, cls)
            if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)
            images = os.listdir(old_class_path)
            image_count = len(os.listdir(new_class_path))
            iter_count = 1
            while image_count < num_to_sample:
                for image in images:
                    if image_count >= num_to_sample:
                        # print(f"cls {cls} now has {len(os.listdir(new_class_path))} images")
                        break
                    old_image_path = os.path.join(old_class_path, image)
                    new_image_path = os.path.join(new_class_path, image[:-4] + "_" + str(iter_count) + ".png")
                    while image[:-4] + "_" + str(iter_count) + ".png" in os.listdir(new_class_path):
                        iter_count += 1
                        new_image_path = os.path.join(new_class_path, image[:-4] + "_" + str(iter_count) + ".png")
                    brightness_criterion = random.choice(BRIGHTNESS_CRITERIA)
                    get_random_background()
                    change_image_background(old_image_path, BACKGROUND_IMAGE_PATH, new_image_path,
                                            brightness_criterion)
                    image_count += 1
                iter_count += 1

            print(f"* {cls_num + 1}/{len(os.listdir(old_dir_path))} classes")
    if os.path.exists(BACKGROUND_IMAGE_PATH):
        os.remove(BACKGROUND_IMAGE_PATH)    


if __name__ == '__main__':
    print("started main")
    change_dataset_background("/galitylab/students/Noam/Datasets/50_dr_hands_white/",
                              "/galitylab/students/Noam/Datasets/50_dr_hands_bg/")
