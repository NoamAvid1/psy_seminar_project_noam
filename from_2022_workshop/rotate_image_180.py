import os
from PIL import Image

ANGLE = 180
#data_types = {"objects":["/home/ssd_storage/datasets/students/Noam/objects_upright/","/home/ssd_storage/datasets/students/Noam/objects_inverted/"],
#              "faces":["/home/ssd_storage/datasets/students/Noam/faces_upright/","/home/ssd_storage/datasets/students/Noam/faces_inverted/"],
#              "birds_species":["/home/ssd_storage/datasets/students/Noam/bird_species_upright/","/home/ssd_storage/datasets/students/Noam/bird_species_inverted/"],
#              "birds_indiv": ["/home/ssd_storage/datasets/students/Noam/bird_indiv_upright/", "/home/ssd_storage/datasets/students/Noam/bird_indiv_inverted/"]}  
# data_types = {"faces":["/home/ssd_storage/datasets/students/Noam/faces_upright/","/home/ssd_storage/datasets/students/Noam/faces_inverted/"]}
data_types = {"faces":["/home/ssd_storage/datasets/students/Noam/Hands_inverted/","/home/ssd_storage/datasets/students/Noam/Hands_upright/"]}  

              
for data_type in data_types:
    # change old_folder_path:
    old_folder_path = data_types[data_type][0]
    new_folder_path = data_types[data_type][1]
    result_folders= [f for f in os.listdir(old_folder_path)]
    for folder in result_folders:
        old_class_path = os.path.join(old_folder_path, str(folder))
        new_class_path = os.path.join(new_folder_path, str(folder))
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path)
        result_files = [fi for fi in os.listdir(old_class_path)]
        for file_name in result_files:
            old_file_path = os.path.join(old_class_path, file_name)
            # new_file_name = file_name[:-4] + '_inverted' + '.jpg'
            new_file_name = file_name
            new_file_path = os.path.join(new_class_path, new_file_name)
            new_image = Image.open(old_file_path)
            out = new_image.rotate(ANGLE)
            out.save(new_file_path)
            new_image.close()
