"""
Created on Wed Jun 22 11:24:07 2022

@author: Stefano Talamona
"""

import glob, os

folder = "PUT_FOLDER_NAME_HERE"
path = "D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/data/test_videos/" + folder + "/"


for pathAndFilename in glob.iglob(os.path.join(path, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print(path+title+'.jpg')
    annotation_file = open(path+title+'.txt', 'w+')
    annotation_file.write("")
    annotation_file.close()
