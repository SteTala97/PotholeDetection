"""

 1 - Data una immagine, leggi file .xml ad essa associato ed estrai:
    • numero di bounding boxes;
    • vertici di ognuna delle bboxes presenti;
    
 2 - Visualizza l'immagine originale e la/le bounding box annotate
 
"""

import xml.etree.ElementTree as ET
import os
import cv2 as cv
import numpy as np



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
                                               
def show(winname, img):
    cv.imshow(winname, img)
    cv.waitKey()
    cv.destroyAllWindows()
 


start = 124 # Numero immagine da cui partire
end = 124   # Numero immagine a cui fermarsi
for n in range(start, end + 1):
    # Carica l'immagine originale  
    image_name = 'img-' + str(n)   
    im_path = os.getcwd() + r'/data/annotated-images/' + image_name + '.jpg'
    img = cv.imread(im_path)
    r, c, ch = img.shape
    #show("immagine", img)
    
    # Carica il file .xml e leggi i dati relativi alle bounding box
    xml_path = os.getcwd() + r'/data/annotated-images/' + image_name + '.xml'
    bbox_values = get_bbox_from_xml(xml_path)
    
    # Disegna le bounding box attorno alle buche presenti nell'immagine originale
    img_bb = np.copy(img)
    n_bboxes = len(bbox_values) // 4
    for i in range(n_bboxes):
        pt1 = (bbox_values[i * 4], bbox_values[i * 4 + 1])
        pt2 = (bbox_values[i * 4 + 2], bbox_values[i * 4 + 3])
        cv.rectangle(img_bb, pt1, pt2, [255, 255, 0], 1)
    
    # Mostra il risultato
    print(f"\n Bounding boxes presenti nell'immagine '{image_name}.jpg' : {n_bboxes}")
    result = np.zeros((r, c * 2, ch), dtype = np.uint8)
    result[:, : c, :] = img
    result[:, c :, :] = img_bb
    show("Buche", result)

