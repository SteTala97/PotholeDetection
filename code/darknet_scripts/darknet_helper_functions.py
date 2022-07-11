"""
Created on Wed Jun 22 09:14:31 2022

@author: Stefano Talamona
"""

import cv2 as cv
import sys
from math import floor, ceil
from time import perf_counter

sys.path.insert(0, r'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/yolov4/darknet')
import darknet


def load_networks():
    # yolov4
    config_file = './cfg/yolov4.cfg'
    data_file = 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/yolov4/darknet/data/obj.data'
    weights = './best_weights/yolov4.weights'
    yolov4_network, class_names, class_colors = darknet.load_network(config_file, data_file, weights)
    # yolov4-tiny
    config_file = './cfg/yolov4-tiny.cfg'
    data_file = 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/yolov4/darknet/data/obj.data'
    weights = './best_weights/yolov4-tiny.weights'
    yolov4tiny_network, class_names, class_colors = darknet.load_network(config_file, data_file, weights)
    # return the loaded models
    return yolov4_network, yolov4tiny_network, class_names, class_colors



def darknet_helper(img, width, height, network, class_names):
    # detection on image (or single video frame)
    darknet_img = darknet.make_image(width, height, 3)
    resized_img = cv.resize(img, (width, height))
    # image ratios are needed to convert bboxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height
    darknet.copy_image_from_bytes(darknet_img, resized_img.tobytes())
    # run darknet and get detections 
    start_time = perf_counter()
    detections = darknet.detect_image(network, class_names, darknet_img)
    detection_time = perf_counter() - start_time
    darknet.free_image(darknet_img)
    return detections, width_ratio, height_ratio, detection_time



def detection_image(image, network, class_names, class_colors, show_confidence, font_size=0.5, bbox_thickness=2):
    r, c = image.shape[:-1]
    # network size
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # perform detection and draw results on image
    bbox_color = (0, 255, 0)
    detections, width_ratio, height_ratio, detection_time = darknet_helper(image, width, height, network, class_names)
    n_detections = 0
    for label, confidence, bbox in detections:
        n_detections += 1
        # resize bbox
        left, top, right, bottom = darknet.bbox2points(bbox)
        left = int(floor(left * width_ratio)) if int(floor(left * width_ratio)) >= 0 else bbox_thickness//2
        top = int(floor(top * height_ratio)) if int(floor(top * height_ratio)) >= 0 else bbox_thickness//2
        right = int(floor(right * width_ratio)) if int(floor(right * width_ratio)) <= c else c-bbox_thickness//2
        bottom = int(floor(bottom * height_ratio)) if int(floor(bottom * height_ratio)) <= r else r-bbox_thickness//2
        # draw bbox
        cv.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thickness)
        # add bbox label
        if show_confidence:
            cv.putText(image, "{} {:.2f}%".format(label, float(confidence)), (left, top - 5),
                       cv.FONT_HERSHEY_SIMPLEX, font_size, bbox_color, 2, cv.LINE_8) #int(ceil(font_size)), cv.LINE_8)
            
    # # show result
    # for label, confidence, bbox in detections:
    #     print("\n", confidence, "%\n")
    # cv.namedWindow('Potholes detection', cv.WINDOW_GUI_EXPANDED)
    # cv.imshow('Potholes detection', image)
    # cv.waitKey()
    # cv.destroyAllWindows()
    return cv.cvtColor(image, cv.COLOR_BGR2RGB), n_detections, detection_time
    

       
def detection_video(video, network, class_names, class_colors, show_confidence, font_size=2, bbox_thickness=4): 
    # network size
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    
    # read video file
    video = 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/data/test_videos/video-3.MOV'
    capture = cv.VideoCapture(video)

    # perform detection and draw results on every frame of the input video
    bbox_color = (0, 255, 0)
    frames_list = []
    total_detections = []
    total_time = 0
    while capture.isOpened:
        frame_acquired, frame = capture.read()

        if frame_acquired:
            frame = cv.resize(frame, (512, 512))
            detections, width_ratio, height_ratio, detection_time = darknet_helper(frame, width, height, network, class_names)
            total_time += detection_time
            n_detections = 0

            for label, confidence, bbox in detections:
                n_detections += 1
                # resize bbox
                left, top, right, bottom = darknet.bbox2points(bbox)
                left = int(floor(left * width_ratio))
                top = int(floor(top * height_ratio))
                right = int(floor(right * width_ratio))
                bottom = int(floor(bottom * height_ratio))
                # draw bbox
                cv.rectangle(frame, (left, top), (right, bottom), bbox_color, bbox_thickness)
                # add bbox label
                if show_confidence:
                    cv.putText(frame, "{} {:.2f}%".format(label, float(confidence)), (left, top - 5),
                               cv.FONT_HERSHEY_SIMPLEX, font_size, bbox_color, int(ceil(font_size)), cv.LINE_8)

            total_detections.append(n_detections)
            frames_list.append(frame)
        else:
            break
    capture.release()

    # # compose the video form the sequence of frames
    # size = np.shape(frames_list[0])[:-1]
    # output_video = cv.VideoWriter('result.mp4', -1, 1, size)
    # for img in frames_list:
    #     output_video.write(img)

    # mean number of detections for each frame
    if len(total_detections) > 0:
        detections_per_frame = round((sum(total_detections) / len(total_detections)), 2)
    else:
        detections_per_frame = 0

    # calculate FPS
    fps = round(len(frames_list) / total_time, 2)

    return frames_list, detections_per_frame, fps
