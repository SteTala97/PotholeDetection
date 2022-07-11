import os
import shutil
import cv2 as cv


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name.lstrip('data/obj/'))
    return names_list


# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
    
# Load filenames of images to copy in the validation set (the ones in 'test.txt')
imgs_to_copy = load_filenames('./test.txt')

# Populate validation set
for img in imgs_to_copy:
    # save images
    path_to_image = "../data/original_dataset/" + img
    image = cv.imread(path_to_image)
    path_save = r'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/code/PROVA/' + img
    cv.imwrite(path_save, image)
    # save annotations
    annotation_og = path_to_image.rstrip('.jpg') + '.txt'
    annotation_target = path_save.rstrip('.jpg') + '.txt'
    shutil.copyfile(annotation_og, annotation_target)