"""
Created on Wed May  4 12:53:53 2022

@author: Stefano Talamona
"""

import os

def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            temp = (image_name.lstrip('data/obj/'))
            names_list.append(temp.rstrip('.txt "Wrong annotation: x = 0 or y = 0" '))
    return names_list


com_path = r"D:\UNIVERSITA\Magistrale\SecondoAnno\VisualInformationProcessingAndManagement\ProgettoVISUAL\yolov4\darknet"
list_path = com_path + "\\bad_label.list"
obj_path = com_path + "\\data\obj\\"

names_list = load_filenames(list_path)

jpg = 0
txt = 0

for file in names_list:
    
    # Remove image
    if os.path.exists(obj_path + file + ".jpg") :
        os.remove(obj_path + file + ".jpg")
        jpg += 1
        
    # Remove annotation file
    if os.path.exists(obj_path + file + ".txt") :
        os.remove(obj_path + file + ".txt")
        txt += 1
        
print(f"\n Images removed: {jpg} \n Annotation files removed: {txt}")
