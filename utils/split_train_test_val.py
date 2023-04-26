import splitfolders

RATIO  = (.8, .2)    # to add test- change to (.8, .1, .1)
INPUT_FOLDER = "/galitylab/students/Noam/Datasets/100_dorsal_right_hands/"
OUTPUT_FOLDER = "/galitylab/students/Noam/Datasets/100_dorsal_right_hands/"


splitfolders.ratio(INPUT_FOLDER, output=OUTPUT_FOLDER,
    seed=1337, ratio=RATIO, group_prefix=None)
    