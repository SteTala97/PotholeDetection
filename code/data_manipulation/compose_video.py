"""
Created on Mon Jun 27 14:39:33 2022

@author: Stefano Talamona
"""

import cv2 as cv
import glob
import re


VIDEO_NAME = 'video-5'
VIDEO_FORMAT = 'mp4'
MODEL_USED = 'yolov4-tiny'
CONF_NO_CONF = 'without_confidence'


# Read images ordered by number (which is used as part of the filename, e.g. "img-123.jpg")
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

common_path = "D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/data/detection_results/"
images_path = common_path + MODEL_USED + '/test_videos/' + VIDEO_NAME + '/' #+ CONF_NO_CONF
saving_path = images_path[:-1] + '.' + VIDEO_FORMAT
# get the size of the frames
width, height = cv.imread(glob.glob(images_path + '/*.jpg')[0]).shape[:-1]
video = cv.VideoWriter(saving_path, cv.VideoWriter_fourcc(*'mp4v'), 30, (height, width))

# Read images in batches and store them as video frames
for filename in sorted(glob.glob(images_path + '/*.jpg'), key=numericalSort):
    img = cv.imread(filename)
    video.write(img)


video.release()
