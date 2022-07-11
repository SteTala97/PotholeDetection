"""
Created on Mon Jun 27 09:21:59 2022

@author: StefanoTalamona
"""

import cv2 as cv
import sys
import os
import glob
from darknet_helper_functions import detection_image, load_networks #, detection_video
sys.path.insert(0, 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/code/telegram_bot')
from iqa import isImageOk


DATA_FOLDER = 'test_videos/video-8'
DATA_FORMAT = 'jpg'
CHOSEN_MODEL = 'yolov4-tiny'
ANNOTATION_FORMAT = 'txt'
DESTINATION_FOLDER = 'detection_results/' + CHOSEN_MODEL + '/' + DATA_FOLDER
SHOW_CONFIDENCE = True
SHOW_ANNOTATIONS = False
SAVE_RESULTS = False
SHOW_RESULTS = True



def show_img(winname, img):
    # cv.namedWindow(winname, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(winname, img)
    k = 0xFF & cv.waitKey()
    if k == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    cv.destroyAllWindows()


def draw_bbox(img, bbox_pts):
    bbox_thickness = 2
    for i in range(len(bbox_pts) // 4):
        pt1 = (bbox_pts[i * 4], bbox_pts[i * 4 + 1])
        pt2 = (bbox_pts[i * 4 + 2], bbox_pts[i * 4 + 3])
        cv.rectangle(img, pt1, pt2, [50, 50, 255], bbox_thickness)
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

    
def init():
    # Load YOLOV4 and YOLOV4-tiny models
    YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS = load_networks()
    # Get folders paths
    cwd = os.getcwd() # Current Working Directory
    cwd = cwd[:-21] # Remove the substring "/code/darknet_scripts" from the cwd (I know... but I'm lazy)
    data_folder = cwd + '/data/' + DATA_FOLDER
    destination_folder = cwd + '/data/' + DESTINATION_FOLDER + '/'
    
    return YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, data_folder, destination_folder
    

def main():
    images_done = 0
    bad_quality_imgs = 0
    bad_resolution_imgs = 0

    if SHOW_CONFIDENCE:
        conf = "with confidence"
    else:
        conf = "without confidence"

    # Pothole Detection on image data
    if DATA_FORMAT == 'jpg':
        # Load models and get folders path
        YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, data_folder, destination_folder = init()
        image_path = data_folder + '/*.' + DATA_FORMAT # generic path for each image
        
        total_detection_time = 0
        # Get each image in 'DATA_FOLDER'
        image_names = [file for file in glob.glob(image_path)]
        n_images = len(image_names)


        for img_name in image_names:
            # Get original image and save its name
            img = cv.imread(img_name)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            dims = img.shape[:-1]
            img = cv.resize(img, (512, 512))
            image_name = img_name.replace(data_folder, '')
            image_name = image_name.replace('.' + DATA_FORMAT, '')

            # # Check image quality and resolution
            # if not isImageOk(img):
            #     print(f"/Image {image_name} did not pass the quality check!")
            #     bad_quality_imgs += 1
            #     problematic_results = show_img(image_name, img, problematic_results)
            #     # continue
            # if (max(dims) / min(dims)) >= 1.8:
            #     print(f"/Image {image_name} has an invalid resolution!")
            #     bad_resolution_imgs += 1
            #     problematic_results = show_img(image_name, img, problematic_results)
            #     # continue
            
            # Perform object (potholes) detection 
            if CHOSEN_MODEL == 'yolov4':
                result, n_potholes, detection_time = detection_image(img, YOLOV4_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.75, bbox_thickness=2)
            elif CHOSEN_MODEL == 'yolov4-tiny':
                result, n_potholes, detection_time = detection_image(img, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS,
                                                                     SHOW_CONFIDENCE, font_size=0.75, bbox_thickness=2)
            else:
                print(f"\nERROR! Model '{CHOSEN_MODEL}' is not defined")
                sys.exit(0)
            total_detection_time += detection_time

            # draw the annotations on the image
            if SHOW_ANNOTATIONS:
                r, c = img.shape[:-1]
                annotation_path = img_name.rstrip('jpg') + ANNOTATION_FORMAT
                bboxes = get_bbox_from_file(annotation_path, ANNOTATION_FORMAT, r, c)
                result = draw_bbox(result, bboxes)
                
            # Show detection results
            if SHOW_RESULTS:
                if (n_potholes == 1):
                    print(f"\n{n_potholes} pothole detected in the image '{image_name}'")
                else:
                    print(f"\n{n_potholes} potholes detected in the image '{image_name}'")
                show_img(image_name, result)
            
            # Save results to memory
            if SAVE_RESULTS:
                if SHOW_CONFIDENCE:
                    # with confidence
                    saving_path = destination_folder + 'with_confidence' + image_name + '-result-conf.' + DATA_FORMAT
                else:
                    # without confidence
                    saving_path = destination_folder + 'without_confidence' + image_name + '-result.' + DATA_FORMAT          
                cv.imwrite(saving_path, result)
            
            images_done += 1
            print(f"{images_done} of {n_images} images done ({conf})")
            
                        
    # Wrong or unsupported data format
    else:
        print(f"\nERROR! Data format '{DATA_FORMAT}' is not defined")
        sys.exit(0)
        
    # Get detection time per image (detection speed expressed in fps)
    fps = n_images / total_detection_time
    print(f"\nDetection completed! The model kept an average detection speed of {round(fps,2)} fps")

    # Number of images that did not pass the quality check control
    print(f"\n{bad_quality_imgs} images did not pass the quality check")
    print(f"\n{bad_resolution_imgs} images have a bad resolution")
 

if __name__ == "__main__":
    main()
  