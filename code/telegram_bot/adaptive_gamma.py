"""
Created on Wed Jul 06 15:52:02 2022

@author: StefanoTalamona
"""

import cv2 as cv
import sys
import os
import glob
import numpy as np
sys.path.insert(0, 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/code/darknet_scripts')
from darknet_helper_functions import detection_image, load_networks


DATA_FOLDER = 'new_data_PROBLEM'
DATA_FORMAT = 'jpg'
CHOSEN_MODEL = 'yolov4'
DESTINATION_FOLDER = ''
SAVE_RESULTS = False
SHOW_RESULTS = True
SHOW_CONFIDENCE = True
ANNOTATION_FORMAT = 'txt'


def init():
    # Load YOLOV4 and YOLOV4-tiny models
    YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS = load_networks()
    # Get folders paths
    cwd = os.getcwd() # Current Working Directory
    cwd = cwd[:-18]
    data_folder = cwd + '/data/' + DATA_FOLDER
    destination_folder = cwd + '/data/' + DESTINATION_FOLDER + '/'
    
    return YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, data_folder, destination_folder


def draw_bbox(img, bbox_pts):
    for i in range(len(bbox_pts) // 4):
        pt1 = (bbox_pts[i * 4], bbox_pts[i * 4 + 1])
        pt2 = (bbox_pts[i * 4 + 2], bbox_pts[i * 4 + 3])
        cv.rectangle(img, pt1, pt2, [50, 50, 255], 2)
    return img


def get_bbox_from_file(path, annotation_format, r, c):
    
    coordinates = []
    # .txt file, YOLO annotation format
    if annotation_format == 'txt':
        with open(path, 'r') as f:
            annotations = [line.rstrip() for line in f]
            for annotation in annotations:
                if annotation != '':
                    values = annotation.split(" ")
                    # annotation values
                    x =      float(values[1])
                    y =      float(values[2])
                    width =  float(values[3])
                    height = float(values[4])
                    # conversion in (x_min, y_min) and (x_max, y_max)
                    x_min = (x * c) - (width  * c) / 2
                    y_min = (y * r) - (height * r) / 2
                    x_max = (x * c) + (width  * c) / 2
                    y_max = (y * r) + (height * r) / 2
                    # append coordinates to "coordinates" list
                    coordinates.append(int(x_min))
                    coordinates.append(int(y_min))
                    coordinates.append(int(x_max))
                    coordinates.append(int(y_max))
    # non-supported format
    else:
        sys.exit(f"The format \"{annotation_format}\" is NOT supported!")
    return coordinates


def show_img(winname, img, img_name):
    # cv.namedWindow(winname, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(winname, img)
    k = 0xFF & cv.waitKey()
    if k == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    elif k == ord('s'):
        print(img_name)

    cv.destroyAllWindows()


def image_agcwd(img, a=0.25, truncated_cdf=False):
    h,w = img.shape[:2]
    hist,bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    intensity_max = unique_intensity.max()
    intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0]**a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
    
    if truncated_cdf: 
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd
    
    img_new = img.copy()
    for i in unique_intensity:
        img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
   
    return img_new


def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed = 255 - agcwd
    return reversed


def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd


def correrct_gamma(img):
    YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]
    # Determine whether image is bright or dimmed
    threshold = 0.3
    exp_in = 112 # Expected global average intensity 
    M,N = img.shape[:2]
    mean_in = np.sum(Y / (M * N)) 
    t = (mean_in - exp_in) / exp_in
    
    # Process image for gamma correction
    img_output = None
    if t < -threshold: # Dimmed Image
        result = process_dimmed(Y)
        YCrCb[:, :, 0] = result
        img_output = cv.cvtColor(YCrCb, cv.COLOR_YCrCb2BGR)
    elif t > threshold: # Bright Image
        result = process_bright(Y)
        YCrCb[:, :, 0] = result
        img_output = cv.cvtColor(YCrCb, cv.COLOR_YCrCb2BGR)
    else:
        img_output = img
    return img_output


def main():
    if DATA_FORMAT == 'jpg':
        # Load models and get folders path
        YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, data_folder, destination_folder = init()
        image_path = data_folder + '/*.' + DATA_FORMAT # generic path for each image
        
        # Get each image in 'DATA_FOLDER'
        image_names = [file for file in glob.glob(image_path)]
        n_images = len(image_names)

        for img_name in image_names:
            img = cv.imread(img_name)
            original = cv.imread(img_name)
            original = cv.resize(original, (512, 512))
            img = cv.resize(img, (512, 512))
            image_name = img_name.replace(data_folder, '')
            image_name = image_name.replace('.' + DATA_FORMAT, '')
            img_gammacorr = correrct_gamma(img)

            # Perform object (potholes) detection 
            if CHOSEN_MODEL == 'yolov4':
                result, n_potholes, detection_time = detection_image(cv.cvtColor(img_gammacorr, cv.COLOR_BGR2RGB), YOLOV4_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.5, bbox_thickness=2)
                result_original, n_potholes, detection_time = detection_image(cv.cvtColor(original, cv.COLOR_BGR2RGB), YOLOV4_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.5, bbox_thickness=2)
            elif CHOSEN_MODEL == 'yolov4-tiny':
                result, n_potholes, detection_time = detection_image(cv.cvtColor(img_gammacorr, cv.COLOR_BGR2RGB), YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.5, bbox_thickness=2)
                result_original, n_potholes, detection_time = detection_image(cv.cvtColor(original, cv.COLOR_BGR2RGB), YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.5, bbox_thickness=2)
            else:
                print(f"\nERROR! Model '{CHOSEN_MODEL}' is not defined")
                sys.exit(0)

            # Show results
            if SHOW_RESULTS:
                r, c = img.shape[:-1]
                annotation_path = img_name.rstrip('jpg') + ANNOTATION_FORMAT
                bboxes = get_bbox_from_file(annotation_path, ANNOTATION_FORMAT, r, c)
                result = draw_bbox(result, bboxes)
                result_original = draw_bbox(result_original, bboxes)
                final_result = np.zeros((1024, 1024, 3), np.uint8)
                final_result[:512, :512, :] = original
                final_result[:512, 512:, :] = result_original
                final_result[512:, :512, :] = img_gammacorr
                final_result[512:, 512:, :] = result
                show_img('original - gamma correction', final_result, image_name)
            
            # # Save results to memory
            # if SAVE_RESULTS:
            #     saving_path = destination_folder + image_name + '-gammcorr.' + DATA_FORMAT
            #     cv.imwrite(saving_path, result)
              
    # Wrong or unsupported data format
    else:
        print(f"\nERROR! Data format '{DATA_FORMAT}' is not defined")
        sys.exit(0)
        

if __name__ == "__main__":
    main()