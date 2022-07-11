"""
Created on Wed Mar  9 08:53:05 2022

@author: Stefano Talamona
"""


"""#########################        IMPORT        #########################"""


import xml.etree.ElementTree as ET
import os
import sys
import cv2 as cv
import numpy as np


"""########################        FUNZIONI        ########################"""

da_modificare = []
da_eliminare = []

def get_bbox_from_xml(path):
    coord = []
    file = ET.parse(path)
    root = file.getroot()
    for bndbox in root.iter('bndbox'):
        coord.append(int(bndbox[0].text))
        coord.append(int(bndbox[1].text))
        coord.append(int(bndbox[2].text))
        coord.append(int(bndbox[3].text))
    return coord

def draw_bbox(img, bbox_pts):
    image = np.copy(img)
    for i in range(len(bbox_pts) // 4):
        pt1 = (bbox_pts[i * 4], bbox_pts[i * 4 + 1])
        pt2 = (bbox_pts[i * 4 + 2], bbox_pts[i * 4 + 3])
        cv.rectangle(image, pt1, pt2, [50, 50, 255], 2)
    return image

def show_img(winname, img, image_name):
    #cv.namedWindow(winname, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(winname, img)
    cv.moveWindow(winname, 0, 0)
    key = 0xFF & cv.waitKey(0)
    """
    'C' o 'c' = l'immagine non va bene, inseriscila tra quelle da eliminare
    'M' o 'm' = l'annotazione non va bene, inserisci l'immagine tra quelle da modificare
    'ESC' = termina l'esecuzione
    'QUALSIASI_ALTRO_TASTO' = prosegui oltre
    """
    if key == 27:
        recap_and_exit()
    elif key == 67 or key == 99:
        da_eliminare.append(image_name)
        print(f" Immagine segnalata per la CANCELLAZIONE! Finora: {len(da_eliminare)}")
    elif key == 77 or key == 109:
        da_modificare.append(image_name)
        print(f" Immagine segnalata per la MODIFICA! Finora: {len(da_modificare)}")
    cv.destroyAllWindows()


def recap_and_exit():
    cv.destroyAllWindows()
    print("\n", "=" * 40)
    print("\n", " " * 5, "Le immagini da cancellare sono: \n")
    for im in da_eliminare: print(" " * 15, im)
    print("\n\n", " " * 5, "Le immagini da modificare sono: \n")
    for im in da_modificare: print(" " * 15, im)
    print("\n", "=" * 40)
    sys.exit(0)


def save_list_to_file(list_to_save):
    file = os.getcwd() + r'\note_varie\immagini_da_ricontrollare_TEMP.txt'
    with open(file, 'w') as f:
        for name in list_to_save:
            f.write("%s\n" % name)


"""############################       MAIN        ##########################"""


def main():
    cwd = os.getcwd() # Current Working Directory
    print("\n\n", cwd.rstrip("code"), "\n\n")
    file = cwd.rstrip("code") + r'note_varie\immagini_da_eliminare.txt' # immagini_da_ricontrollare.txt
    
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            # Carica l'immagine originale  
            im_path = cwd.rstrip("code") + r'data/annotated-images/' + image_name + '.jpg'
            img = cv.imread(im_path)
            
            if img is not None:
                r, c, ch = img.shape
                # Carica il file .xml e leggi i dati relativi alle bounding box
                xml_path = cwd.rstrip("code") + r'data/annotated-images/' + image_name + '.xml'
                bbox_values = get_bbox_from_xml(xml_path)
                # Disegna le bounding box attorno alle buche
                img_bb = draw_bbox(img, bbox_values)
                n_bboxes = len(bbox_values) // 4
                print(f"\n Numero di buche in {image_name}: {n_bboxes}")
                
                # Giudica la qualità delle annotazioni
                result = np.zeros((r, c*2, ch), dtype=np.uint8)
                result[:, :c, :] = img
                result[:, c:, :] = img_bb
                show_img(image_name, result, image_name)
        
    recap_and_exit()


if __name__ == "__main__":
    main()
    
