"""
Created on Sat Jul  2 18:08:46 2022

@author: Stefano Talamona
"""


import cv2 as cv
import numpy as np
from math import isnan
import sys
sys.path.insert(0, r'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/code/image_quality_assessment')
from performIQA import calculate_image_quality_score, calculate_brisque_features


BRISQUE_THRESH = 90
UPPER_BOUND = 190
LOWER_BOUND = -10
KERNEL_SIZE = 7
SIGMA = 7/6



def scale_score(score):
    return (((score - LOWER_BOUND) * 100) / UPPER_BOUND)
    
    
def calculateImageScore(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brisque_features = calculate_brisque_features(img, kernel_size=KERNEL_SIZE, sigma=SIGMA)
    downscaled_image = cv.resize(img, None, fx=1/2, fy=1/2, interpolation=cv.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=KERNEL_SIZE, sigma=SIGMA)
    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
    score = calculate_image_quality_score(brisque_features)
    if isnan(score):
        return UPPER_BOUND
    
    return scale_score(score)


def isImageOk(img):
    if calculateImageScore(img) > BRISQUE_THRESH:
        return False
    return True


def isVideoOk(video):
    capture = cv.VideoCapture(video)
    while capture.isOpened:
        ret, frame = capture.read()
        if ret:
            if calculateImageScore(frame) > BRISQUE_THRESH:
                return False
        else:
            break
    capture.release()
    return True


# # Test on image
# img_path = r'D:\UNIVERSITA\Magistrale\SecondoAnno\VisualInformationProcessingAndManagement\ProgettoVISUAL\data\validation_set'
# img_path = img_path + '/img-658.jpg'
# img = cv.imread(img_path)
# cv.imshow('originale', img)
# print(calculateImageScore(img))
# print(isImageOk(img), "\n")
# img = cv.GaussianBlur(img, (5, 5), 0)
# # cv.imshow('dopo blurring', img)
# # cv.waitKey()
# # cv.destroyAllWindows()
# print(calculateImageScore(img))
# print(isImageOk(img), "\n")
# img = cv.GaussianBlur(img, (25, 25), 0)
# print(calculateImageScore(img))
# print(isImageOk(img), "\n")

## Test on video
# video = r'D:\UNIVERSITA\Magistrale\SecondoAnno\VisualInformationProcessingAndManagement\ProgettoVISUAL\data\test_videos'
# video = video + '/video2-pos.mov'
# print(isVideoOk(video))


