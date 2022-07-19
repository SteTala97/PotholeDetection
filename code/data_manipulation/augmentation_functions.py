"""
Created on Sat Mar 12 15:47:06 2022

@author: Stefano Talamona
"""


import xml.etree.ElementTree as ET
import sys
import cv2 as cv
import numpy as np



# I formati supportati sono .xml (PascalVOC) e .txt (YOLO);
# annotation_format in ['xml', 'yolo'];
# Le annotazioni in formato YOLO vengono convertite in PascalVOC per poter 
# operare le trasformazioni geometriche di data augmentation
def get_bbox_from_file(path, annotation_format, r, c):
    
    coordinates = []
    
    # .xml file, annotazione in formato PascalVOC
    if annotation_format == 'xml':
        file = ET.parse(path)
        root = file.getroot()
        for bndbox in root.iter('bndbox'):
            coordinates.append(int(bndbox[0].text))
            coordinates.append(int(bndbox[1].text))
            coordinates.append(int(bndbox[2].text))
            coordinates.append(int(bndbox[3].text))
            
    # .txt file, annotazione in formato YOLO
    elif annotation_format == 'txt':
        with open(path, 'r') as f:
            annotations = [line.rstrip() for line in f]
            for annotation in annotations:
                if annotation != '':
                    values = annotation.split(" ")
                    # Valori originali presenti nell'annotazione
                    x =      float(values[1])
                    y =      float(values[2])
                    width =  float(values[3])
                    height = float(values[4])
                    # Converti in (x_min, y_min) e (x_max, y_max)
                    x_min = (x * c) - (width  * c) / 2
                    y_min = (y * r) - (height * r) / 2
                    x_max = (x * c) + (width  * c) / 2
                    y_max = (y * r) + (height * r) / 2
                    # Aggiungili alla lista
                    coordinates.append(int(x_min))
                    coordinates.append(int(y_min))
                    coordinates.append(int(x_max))
                    coordinates.append(int(y_max))
                    
    # formato non supportato
    else:
        sys.exit(f"The supported annotation formats are \"xml\" for PascalVOC and \"txt\" for YOLO, \
the specified format \"{annotation_format}\" is NOT supported!")
    return coordinates




# Non serve - Non ho mai da convertire da lista YOLO a lista PascalVOC
def yolo2pascalvoc(bboxes, r, c):
    
    n_bboxes = len(bboxes) // 4
    
    # Per ogni bbox scrivi una riga di annotazione
    pascalvoc_list = []
    for i in range(n_bboxes):
        # Prendi i valori: x_centro, y_centro, larghezza, altezza
        x = (bboxes[i * 4])
        y = (bboxes[i * 4 + 1])
        width = (bboxes[i * 4 + 2])
        height = (bboxes[i * 4 + 3])
        
        # Vertice in basso a sinistra
        x_min = (x * c) - (width  * c) / 2
        y_min = (y * r) - (height * r) / 2
        
        # Vertice in alto a destra
        x_max = (x * c) + (width  * c) / 2
        y_max = (y * r) + (height * r) / 2
        
        # Concatena i valori calcolati per ottenere la lista desiderata
        pascalvoc_list.append(int(x_max))
        pascalvoc_list.append(int(y_max))
        pascalvoc_list.append(int(x_min))
        pascalvoc_list.append(int(y_min))
        
    return pascalvoc_list
    
    


# Converti da lista PascalVOC a stringa YOLO per memorizzare su file .txt
# la nuova annotazione
def pascalvoc2yolo(bboxes, r, c):
    
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
        x_center /= c
        y_center /= r
        width  /= c
        height /= r
        
        # Realizza la stringa inerente alla bbox corrente
        yolo_string += '0 ' + str(x_center) + ' ' + str(y_center) + ' ' + \
                              str(width)    + ' ' + str(height)   + '\n'
                              
    return yolo_string




def write_xml(points, path):
    root = ET.Element("annotation")
    for i in range (len(points) // 4):
        bndbox = ET.SubElement(root, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(points[i * 4])
        ET.SubElement(bndbox, "ymin").text = str(points[i * 4 + 1])
        ET.SubElement(bndbox, "xmax").text = str(points[i * 4 + 2])
        ET.SubElement(bndbox, "ymax").text = str(points[i * 4 + 3])
    tree = ET.ElementTree(root)
    tree.write(path)




# L'annotazione contenuta nella variabile "points" arriva in input sempre in 
# formato PascalVOC, mai in formato YOLO; questo perchè le annotazioni in 
# formato YOLO vengono convertite in formato PascalVOC per poter operare le 
# operazioni di data augmentation
def write_annotation(path, points, save_annotation_format, r, c):
    
    # Salva in formato .txt
    if save_annotation_format == 'txt':
        yolo_string = pascalvoc2yolo(points, r, c)
        with open(path, 'w') as f:
            f.writelines(yolo_string)
            
    # Salva in formato .xml
    elif save_annotation_format == 'xml':
        write_xml(points, path)
    
    # formato non supportato
    else:
        sys.exit(f"The supported annotation formats are \"xml\" for PascalVOC and \"txt\" for YOLO, \
 the specified format \"{save_annotation_format}\" is NOT supported!")      
     



def rotate(img, bbox_pts, angle):
    
    if len(bbox_pts) == 0:  
        sys.exit("The bounding box points are invalid for function \"rotate\"!")
        
    # Ruota l'immagine preservandone il contenuto
    (r, c) = img.shape[:2]
    (cX, cY) = (c // 2, r // 2) # Coordinate del centro
    M = cv.getRotationMatrix2D((cX, cY), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Vengono calcolate le nuove dimensioni che avrà l'immagine ruotata
    newW = int((r * sin) + (c * cos))
    newH = int((r * cos) + (c * sin))
    # Viene modificata la matrice di rotazione per tener conto della traslazione
    # necessaria a non compromettere l'immagine originale
    M[0, 2] += (newW / 2) - cX
    M[1, 2] += (newH / 2) - cY
    
    
    # L'immagine è ruotata, vanno ora ruotati anche i vertici delle bbox
    n_bboxes = len(bbox_pts) // 4
    bboxes = np.zeros((n_bboxes, 4), dtype = int)
    for i in range(n_bboxes):
        bboxes[i, :] = bbox_pts[i * 4 : i * 4 + 4]
    # Devo avere i 4 vertici delle bbox per poter calcolare in questo modo le 
    # nuove coordinate (si potrebbe fare anche con due in qualche modo, ma poco
    # importa, basta che le coordinate siano giuste)
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)
    # Vertice 1
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)
    # Vertice 2
    x2 = x1 + width
    y2 = y1 
    # Vertice 3 
    x3 = x1
    y3 = y1 + height
    # Vertice 4
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)
    # Vettore dei vertici
    vertices = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    vertices = vertices.reshape(-1,2)
    
    # Trasformalo in coordinate omogenee per poter eseguire il prodotto 
    # matriciale che seguirà (metti un "1" in coda ad ogni punto)
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1), dtype = int)))
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((r * sin) + (c * cos))
    nH = int((r * cos) + (c * sin))
    # Modifica la matrice di roto-traslazione per tener conto della traslazione
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # Applica la trasformazione lineare
    transformed_pts = np.dot(M, vertices.T).T    
    transformed_pts = transformed_pts.reshape(-1, 8)
    
    # Ora bisogna tracciare la bbox di dimensione minima
    x_ = transformed_pts[:, [0, 2, 4, 6]]
    y_ = transformed_pts[:, [1, 3, 5, 7]]
    xmin = np.min(x_, 1).reshape(-1, 1).astype(int)
    ymin = np.min(y_, 1).reshape(-1, 1).astype(int)
    xmax = np.max(x_, 1).reshape(-1, 1).astype(int)
    ymax = np.max(y_, 1).reshape(-1, 1).astype(int)
    final_bbox_pts = np.hstack((xmin, ymin, xmax, ymax)).reshape(1, -1)
    
    return cv.warpAffine(img, M, (newW, newH)), final_bbox_pts[0, :].tolist(), newW, newH



# "op" in ['gamma', 'brightness', 'saturation', 'hue'] 
def jitter(img, op, gamma = 1, brightness = 0, saturation = 0, hue = 0):
    
    # Gamma correction
    if op == 'gamma':
        if gamma == 1.0: 
            return img
        table = np.array([((i / 255.0) ** gamma) * 255
		                  for i in np.arange(0, 256)]).astype("uint8")
        return cv.LUT(img, table)
    
    # Brightness    
    elif op == 'brightness':
        if brightness == 0:
            return img
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        v = img_hsv[:, :, 2]
        v = cv.add(v, brightness)
        v[v > 255] = 255
        v[v < 0] = 0
        img_hsv[:, :, 2] = v        
        return cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    
    # Saturation
    elif op == 'saturation':
        if saturation == 0:
            return img
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        s = img_hsv[:, :, 1]
        s = cv.add(s, saturation)
        s = cv.add(s, int(np.min(s)))
        if np.max(s) != 0:
            s = s / np.max(s) * 255
        else:
            s = s / 255
        img_hsv[:, :, 1] = s        
        return cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    
    # Hue
    elif op == 'hue':
        if hue == 0:
            return img
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h = img_hsv[:, :, 0]
        h = cv.add(h, hue)
        h[h > 179] = abs(hue)
        h[h < 0] = 0
        img_hsv[:, :, 0] = h      
        return cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    
    # Nome operazione non corretto: restituisci l'immagine originale
    else:
        print("\nERROR - The specified operation is not supported!")
        return img




def flip(img, bbox_pts, direction):
    
    if len(bbox_pts) == 0: 
        sys.exit("The bounding box points are invalid for function \"flip\"!")
        
    pts = np.copy(bbox_pts)
    
    # Flipping orizzontale
    if (direction == 'horizontal'):
        img = cv.flip(img, 1)
        ctr = img.shape[1] // 2
        # Ribalta ogni vertice di ciascuna annotazione rispetto al centro dell'immagine
        for i in range(0, len(pts), 2):
            dist = pts[i] - ctr
            pts[i] = pts[i] - dist * 2
        # Ora i vertici sono opposti, vanno scambiati
        for i in range(0, len(pts), 4):
            tmp = pts[i]
            pts[i] = pts[i+2]
            pts[i+2] = tmp
        return img, pts
    
    # Flipping orizzontale
    if (direction == 'vertical'):
        img = cv.flip(img, 0)
        ctr = img.shape[0] // 2
        # Ribalta ogni vertice di ciascuna annotazione rispetto al centro dell'immagine
        for i in range(1, len(pts), 2):
            dist = pts[i] - ctr
            pts[i] = pts[i] - dist * 2
        # Ora i vertici sono opposti, vanno scambiati
        for i in range(1, len(pts), 4):
            tmp = pts[i]
            pts[i] = pts[i+2]
            pts[i+2] = tmp
        return img, pts
        
    # Se la stringa "direction" non è valida, l'immagine ed i punti dei vertici
    # della bounding box vengono restituiti tali e quali ed un messaggio di 
    # errore viene stampato a console
    else:
        print("\nERROR - Direction should be 'horizontal' or 'vertical'!")
        return img, bbox_pts
    
    
    
# "blurring_factor" fa riferimento alle dimensioni dell'immagine in input, 
# significa: "prendi un 'tot'-esimo delle dimensioni dell'immagine come dimensioni
# per il kernel gaussiano con cui effettuare il blur"; ad esempio, con il valore
# standard fissato a 75, si intende che: dim_kernel = img.shape / 75
def blur(img, blurring_factor = 75):
    h, w, _ = np.asarray(img.shape) / blurring_factor
    h, w = int(h), int(w)
    # Il kernel deve avere dimensioni dispari
    if h % 2 == 0 : h += 1
    if w % 2 == 0 : w += 1
    
    return cv.GaussianBlur(img, (h, w), 0)
