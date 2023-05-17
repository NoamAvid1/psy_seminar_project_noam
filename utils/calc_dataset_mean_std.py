# code taken from kaggle : https://www.kaggle.com/code/aliabdin1/calculate-mean-std-of-images

import numpy as np
import cv2
import os

train_img_root = '/galitylab/students/Noam/Datasets/hands_both_bg/train/'

def calc_avg_mean_std(folder_names, img_root, size):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])
    n_images = 0
    n_folders = len(folder_names)
    print(f'size = {size}')
    print(f'n_folders = {n_folders}')
    for folder in folder_names:
      img_names = os.listdir(train_img_root + folder)
      for img_name in img_names:
          #print(f'img_name = {img_name}')
          try:
            #print(f'path = {img_root + os.path.join(folder ,img_name)}')
            img = cv2.imread(img_root + os.path.join(folder ,img_name))
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean, std = cv2.meanStdDev(img)
            mean_sum += np.squeeze(mean)
            std_sum += np.squeeze(std)
            n_images+= 1
          except Exception as e:
            print(str(e))
    print(n_images)
    return (mean_sum / n_images, std_sum / n_images)

print("started run")
folder_names = os.listdir(train_img_root)
train_mean, train_std = calc_avg_mean_std(folder_names, train_img_root, (1600,1200))
print(f'calculating mean and std for dataset {train_img_root}')
print(f'train mean: {train_mean}, train std: {train_std}')
print("NORMALIZED:")
print()
# print(f'train mean: {train_mean}, NORMALIZED: {train_mean / 255}')
# print(f'train std: {train_std}, NORMALIZED: {train_std / 255}')

