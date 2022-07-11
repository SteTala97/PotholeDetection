"""
Created on Tue Jun 14 15:01:24 2022

@author: Stefano Talamona
"""

import cv2 as cv

SAVE_IMAGE_FOLDER = 'video-8'
IMAGE_FORMAT = '.jpg'
VIDEO_NAME = 'video-8'
VIDEO_FORMAT = '.MOV'

path_to_video = '../../data/test_videos/' + VIDEO_NAME + VIDEO_FORMAT
video = cv.VideoCapture(path_to_video)

if not video.isOpened:
    print("Could not open selected file!")
        
i = 0
while video.isOpened :
    ret, frame = video.read()
    if ret:
       # cv.imshow("Frame", frame)
       
       image_path = '../../data/test_videos/' + SAVE_IMAGE_FOLDER + '/' + VIDEO_NAME + '_frame' + str(i) + IMAGE_FORMAT
       cv.imwrite(image_path, frame)
       i += 1
       
       # if cv.waitKey(30) & 0xFF == ord(' '): # press space to quit
       #     break  
    else:
        break
    
    
video.release()
cv.destroyAllWindows()