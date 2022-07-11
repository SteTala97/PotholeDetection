"""
Created on Thu Mar 10 21:45:04 2022

@author: Stefano Talamona

Converte i file di annotazione .xml in formato PascalVOC in file .txt in 
formato YOLO; una volta convertito, il file .xml viene eliminato

Da: xmin, ymin, xmax, ymax
A : class, xCenter, yCenter, width, height

NOTA: tutti i valori vanno normalizzati in [0, 1]

"""



import xml.etree.ElementTree as ET
import os
import glob



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


def get_size_from_xml(path):
    size = []
    file = ET.parse(path)
    root = file.getroot()
    for bndbox in root.iter('size'):
        size.append(int(bndbox[0].text))
        size.append(int(bndbox[1].text))
    return size




def main():
    directory = os.getcwd() + '/data/new_dataset/*.xml'
    annotazioni = [file for file in glob.glob(directory)]
    for annotazione in annotazioni:
        # Carica il file .xml e leggi i dati relativi alle bounding box
        bboxes = get_bbox_from_xml(annotazione)
        n_bboxes = len(bboxes) // 4
        
        # Per ogni bbox scrivi una riga di annotazione
        yolo_string = ''
        for i in range(n_bboxes):
            # Prendi i due vertici: in alto a sinistra e in basso a destra
            p_min = (bboxes[i * 4], bboxes[i * 4 + 1])
            p_max = (bboxes[i * 4 + 2], bboxes[i * 4 + 3])
            
            # Centro della bbox
            x_center = (p_min[0] + p_max[0]) / 2
            y_center = (p_min[1] + p_max[1]) / 2
            
            # Ampiezza della bbox
            width =  p_max[0] - p_min[0]
            height = p_max[1] - p_min[1]
            
            # Normalizza in [0, 1] rispetto alle dimensioni dell'immagine
            size = get_size_from_xml(annotazione)
            rows = size[0]
            cols = size[1]
            x_center /= cols
            y_center /= rows
            width /= cols
            height /= rows
            
            # Realizza la stringa inerente alla bbox corrente
            yolo_string += '0 ' + str(x_center) + ' ' + str(y_center) + ' ' + \
                                  str(width)    + ' ' + str(height)   + '\n'
            #print(yolo_string)
        
        # Salva l'annotazione su file
        file = annotazione.replace('.xml', '.txt')
        with open(file, 'w') as f:
            f.writelines(yolo_string)
            
        # Rimuovi l'annotazione in formato .xml
        os.remove(annotazione)



if __name__ == "__main__":
    main()
    


