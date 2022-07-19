"""
Created on Thu Feb 10 14:40:45 2022

@author: Stefano Talamona

"""


SAVE = False # [True, False]
AUGMENT = False # [True, False]
IMAGE_FORMAT = 'jpg' # ['jpg', 'png', 'JPEG']
ANNOTATION_FORMAT = 'txt' # ['txt', 'xml']
SAVE_ANNOTATION_FORMAT = 'txt' # ['txt', 'xml']
DATA_FOLDER = 'PROVA_TMP'
ANNOTATION_FOLDER = 'PROVA_TMP'
DESTINATION_FOLDER = 'PROVA_TMP'



"""#########################        IMPORTS        #########################"""


import os
import cv2 as cv
import numpy as np
import sys
import glob
from augmentation_functions import get_bbox_from_file, flip, blur,  \
                                   jitter, write_annotation, rotate


"""########################        FUNCTIONS        ########################"""



def show_img(winname, img):
    cv.namedWindow(winname, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(winname, img)
    k = 0xFF & cv.waitKey()
    if k == 27:
        cv.destroyAllWindows()
        sys.exit(0)
    cv.destroyAllWindows()
    
        
def draw_bbox(img, bbox_pts):
    image = np.copy(img)
    for i in range(len(bbox_pts) // 4):
        pt1 = (bbox_pts[i * 4], bbox_pts[i * 4 + 1])
        pt2 = (bbox_pts[i * 4 + 2], bbox_pts[i * 4 + 3])
        cv.rectangle(image, pt1, pt2, [50, 50, 255], 2)
    return image


def save(img, bbox_pts, path, rows, cols):
    if len(bbox_pts) > 0:        
        img_path = path + '.' + IMAGE_FORMAT
        annotation_path = path + '.' + SAVE_ANNOTATION_FORMAT
        cv.imwrite(img_path, img)
        write_annotation(annotation_path, bbox_pts, SAVE_ANNOTATION_FORMAT, rows, cols)
    else:
        print(f'\n Image: \n {path} \n has not been saved since it contains no objects')


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name.lstrip('data/obj/'))
    return names_list



"""############################      MAIN       ###########################"""



def main():
    cwd = os.getcwd() # Current Working Directory
    cwd = cwd[:-23] # Remove the substring "/code/data_manipulation" from the cwd
    destination_folder = cwd + '/data/' + DESTINATION_FOLDER + '/' # Directory del dataset aumentato
    generic_path = cwd + '/data/' + DATA_FOLDER 
    im_path = generic_path + '/*.' + IMAGE_FORMAT
    image_names = [file for file in glob.glob(im_path)] # Percorsi delle immagini originali

    for img_name in image_names: 
        # Carica l'immagine originale
        img = cv.imread(img_name)
        
        # Prendi il nome dell'immagine, servirà più avanti
        image_name = img_name.replace(generic_path, '')
        image_name = image_name.replace('.' + IMAGE_FORMAT, '')
        
        r, c, ch = img.shape      # L'immagine è un array con shape r*c*ch
        #show_img("immagine", img) # Mostra l'immagine appena caricata
        
        # Carica il file di annotazione e leggi i dati relativi alle bounding box
        annotation_path = cwd + '/data/' + ANNOTATION_FOLDER + '/' + \
                          image_name + '.' + ANNOTATION_FORMAT
        bbox_values = get_bbox_from_file(annotation_path, ANNOTATION_FORMAT, r, c)
        n_bboxes = len(bbox_values) // 4

        # img_bb = draw_bbox(img, bbox_values)
        # show_img("immagine", img_bb)
        
        if(AUGMENT): 
            
            # Rendi pari le dimensioni delle immagini in modo da evitare problemi
            # riscontrati durante alcune operazioni di data augmentation
            needs_resize = False
            if r % 2 != 0 :
                r -= 1
                needs_resize = True
            if c % 2 != 0 :
                c -= 1
                needs_resize = True
            if needs_resize :
                img = cv.resize(img, [c, r])
                needs_resize = False
            
            img_bb = draw_bbox(img, bbox_values)
            # Salva l'immagine originale
            if SAVE:
                s_path = destination_folder + image_name 
                save(img, bbox_values, s_path, r, c)
            
                
            """ ↓ Le operazioni di data augmentation iniziano da qui ↓ """
            
            ### FLIPPING ###
            
            # FLIP ORIZZONTTALE
            img_h, bbox_values_h = flip(img, bbox_values, 'horizontal')
            #img_h_bb = draw_bbox(img_h, bbox_values_h)
            # FLIP VERTICALE
            img_v, bbox_values_v = flip(img, bbox_values, 'vertical')
            #img_v_bb = draw_bbox(img_v, bbox_values_v)
            
            # Salva in memoria i risultati del FLIPPING
            s_path = destination_folder + image_name + 'hflip'
            if SAVE:
                save(img_h, bbox_values_h, s_path, r, c)
            s_path = destination_folder + image_name + 'vflip'
            if SAVE:
                save(img_v, bbox_values_v, s_path, r, c)
            
            # # Mostra il risultato del flipping
            # print(f"\n Bounding boxes presenti nell'immagine '{image_name}.jpg' : {n_bboxes}")
            # result = np.zeros((r * 3, c * 2, ch), dtype = np.uint8)
            # result[: r, : c, :] = img
            # result[: r, c :, :] = img_bb
            # result[r : r * 2, : c, :] = img_h
            # result[r : r * 2, c :, :] = img_h_bb
            # result[r * 2 :, : c, :] = img_v
            # result[r * 2 :, c :, :] = img_v_bb
            # show_img("Risultati flipping", result)
            
            
            ### JITTERING ###
            
            # GAMMA CORRECTION
            img_g_pos = jitter(img, 'gamma', 0.5) # incremento positivo
            # img_g_pos_bb = draw_bbox(img_g_pos, bbox_values)
            img_g_neg = jitter(img, 'gamma', 1.5)   # incremento negativo
            # img_g_neg_bb = draw_bbox(img_g_neg, bbox_values)
            
            # Salva in memoria i risultati della GAMMA CORRECTION
            s_path = destination_folder + image_name + 'gamma_pos'
            if SAVE:
                save(img_g_pos, bbox_values, s_path, r, c)
            s_path = destination_folder + image_name + 'gamma_neg'
            if SAVE:
                save(img_g_neg, bbox_values, s_path, r, c)
            
            # BRIGHTNESS
            img_b_pos = jitter(img, 'brightness', brightness = 50)  # incremento positivo
            # img_b_pos_bb = draw_bbox(img_b_pos, bbox_values)
            img_b_neg = jitter(img, 'brightness', brightness = -50) # incremento negativo
            # img_b_neg_bb = draw_bbox(img_b_neg, bbox_values)
            
            # Salva in memoria i risultati della BRIGHTNESS
            s_path = destination_folder + image_name + 'bright_pos'
            if SAVE:
                save(img_b_pos, bbox_values, s_path, r, c)
            s_path = destination_folder + image_name + 'bright_neg'
            if SAVE:
                save(img_b_neg, bbox_values, s_path, r, c)
            
            # SATURATION
            img_s_pos = jitter(img, 'saturation', saturation = 15)   # incremento positivo
            # img_s_pos_bb = draw_bbox(img_s_pos, bbox_values)
            img_s_neg = jitter(img, 'saturation', saturation = -100) # incremento negativo
            # img_s_neg_bb = draw_bbox(img_s_neg, bbox_values)
            
            # Salva in memoria i risultati della SATURATION
            s_path = destination_folder + image_name + 'satur_neg'
            if SAVE:
                save(img_s_neg, bbox_values, s_path, r, c)
            s_path = destination_folder + image_name + 'satur_pos'
            if SAVE:
                save(img_s_pos, bbox_values, s_path, r, c)
            
            # HUE
            img_h_pos = jitter(img, 'hue', hue = 15)  # incremento positivo
            # img_h_pos_bb = draw_bbox(img_h_pos, bbox_values)
            img_h_neg = jitter(img, 'hue', hue = -15) # incremento negativo
            # img_h_neg_bb = draw_bbox(img_h_neg, bbox_values)
            
            # Salva in memoria i risultati della SATURATION
            s_path = destination_folder + image_name + 'hue_neg'
            if SAVE:
                save(img_h_pos, bbox_values, s_path, r, c)
            s_path = destination_folder + image_name + 'hue_pos'
            if SAVE:
                save(img_h_neg, bbox_values, s_path, r, c)
            
            
            # # Mostra il risultato del jittering (solo una operazione viene mostrata,
            # # bisogna specificare quali immagini stampare a video)
            # print(f"\n Bounding boxes presenti nell'immagine '{image_name}.jpg' : {n_bboxes}")
            # result = np.zeros((r * 3, c * 2, ch), dtype = np.uint8)
            # result[: r, : c, :] = img
            # result[: r, c :, :] = img_bb
            # result[r : r * 2, : c, :] = img_g_pos
            # result[r : r * 2, c :, :] = img_g_pos_bb
            # result[r * 2 :, : c, :] = img_g_neg
            # result[r * 2 :, c :, :] = img_g_neg_bb
            # show_img("Risultati jitering", result)
            
            
            ### ROTAZIONE ###
            
            img_rot_pos, bbox_rot_pos, newC, newR = rotate(img, bbox_values, 5)
            # img_rot_pos_bb = draw_bbox(img_rot_pos, bbox_rot_pos)
            img_rot_neg, bbox_rot_neg,  newC, newR = rotate(img, bbox_values, -5)
            # img_rot_neg_bb = draw_bbox(img_rot_neg, bbox_rot_neg)
            
            # Salva in memoria i risultati della ROTAZIONE
            s_path = destination_folder + image_name + 'rot_pos_5deg'
            if SAVE:
                save(img_rot_pos_bb, bbox_rot_pos, s_path, newR, newC)
            s_path = destination_folder + image_name + 'rot_neg_5deg'
            if SAVE:
                save(img_rot_neg_bb, bbox_rot_neg, s_path, newR, newC)
            
            # # Mostra il risultato della rotazione 
            # print(f"\n Bounding boxes presenti nell'immagine '{image_name}.jpg' : {n_bboxes}")
            # rp, cp = img_rot_pos.shape[:2] # Dimensioni immagine ruotata con angolo positivo
            # rn, cn = img_rot_neg.shape[:2] # Dimensioni immagine ruotata con angolo negativo
            # result = np.zeros((r + rp + rn, max([c, cn, cp]) * 2, ch), dtype = np.uint8)
            # result[:r, :c, :] = img
            # result[:r, c : c * 2, :] = img_bb
            # result[r : r + rp, :cp, :] = img_rot_pos
            # result[r : r + rp, cp : cp * 2, :] = img_rot_pos_bb
            # result[r + rp:, :cn, :] = img_rot_neg
            # result[r + rp:, cn : cn * 2, :] = img_rot_neg_bb
            # show_img("Risultati rotazione 5", result)


            ### BLURRING ###
            
            blurred = blur(img, 75)
            # Salva in memoria i risultati del CROPPING
            s_path = destination_folder + image_name + 'blur_75'
            if SAVE:
                save(blurred, bbox_values, s_path, r, c)
            
            # # Mostra il risultato del blurring
            # blurred_bb = draw_bbox(blurred, bbox_values)
            # result = np.zeros((r * 2, c * 2, ch), dtype = np.uint8)
            # result[:r, :c, :] = img
            # result[:r, c:, :] = img_bb
            # result[r:, :c, :] = blurred
            # result[r:, c:, :] = blurred_bb
            # show_img("Risultato blurring", result)
        

        if (SAVE):
            s_path = destination_folder + image_name
            save(img, bbox_values, s_path, r, c)



if __name__ == "__main__":
    main()
