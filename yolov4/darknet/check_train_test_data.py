"""
Created on Wed Jun 22 13:03:03 2022

@author: Stefano Talamona
"""

import glob, os


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name.lstrip('data/obj/'))
    return names_list


current_dir = 'data/obj'
    
# Load filenames of test images (the ones in 'test.txt')
test_imgs = load_filenames('data/test.txt')
# Load filenames of training images (the ones in 'train.txt')
train_imgs = load_filenames('data/train.txt')

# Check that there are no correspondences between train and test images
wrong_imgs = []
for test_img in test_imgs:
    if test_img in train_imgs:
        wrong_imgs.append(test_img)
        
if len(wrong_imgs) == 0:
    print("Everything alright - there are no test images in the training set")
else:
    print("WARNING! - the following test images are also in the training set:\n")
    for wrong_img in wrong_imgs:
        print(f"{wrong_img}")