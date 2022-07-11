"""
Created on Tue Jul 5 17:33:46 2022

@author: Stefano Talamona
"""


import glob
import os
from PIL import Image
import pillow_heif

CURRENT_FORMAT = '.heic'
NEW_FORMAT = '.jpg'
LOAD_FOLDER = r'C:\Users\\stefa\\Downloads\\new_data_TEMP-20220706T094935Z-001\\new_data_TEMP\\*'
SAVE_FOLDER = r'D:\\UNIVERSITA\\Magistrale\\SecondoAnno\\VisualInformationProcessingAndManagement\\ProgettoVISUAL\\data\\new_data_TEMP\\'

generic_path = LOAD_FOLDER + CURRENT_FORMAT
i = 0
# print(generic_path)
for file in glob.glob(generic_path):
	heif_file = pillow_heif.read_heif(file)
	image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
	new_file = file.lstrip(LOAD_FOLDER[:-2])
	new_file = SAVE_FOLDER + new_file.rstrip(CURRENT_FORMAT) + NEW_FORMAT
	image.save(new_file)
	print(i)
	i += 1

print("\nDONE!")