"""
Created on Thu Mar 10 17:06:27 2022

@author: Stefano Talamona
"""

from shutil import copy
import os


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name)
    return names_list
            


def main():
    cwd = os.getcwd() # Current Working Directory
    cwd = cwd[:-5]
    txt_path = cwd + '/note_varie/immagini_da_eliminare.txt'
    image_names_e = load_filenames(txt_path)
    txt_path = cwd + '/note_varie/immagini_da_modificare.txt'
    image_names_m = load_filenames(txt_path)
    image_names = image_names_e + image_names_m
    
    start = 1 # Numero immagine da cui partire
    end = 665 # Numero immagine a cui fermarsi
    
    for n in range(start, end + 1):
        # Carica l'immagine originale
        image_name = 'img-' + str(n)
        if image_name not in image_names:
            # Save jpg image
            old_path = cwd + '/data/annotated-images/' + image_name + '.jpg'
            new_path = cwd + '/data/PROVA/' + image_name + '.jpg'
            copy(old_path, new_path) # "copy" imported from shutil package
            # Save xml file
            old_path = cwd + '/data/annotated-images/' + image_name + '.xml'
            new_path = cwd + '/data/PROVA/' + image_name + '.xml'
            copy(old_path, new_path)
        else:
            print(f" Image {image_name} \"removed\"")
 
    print(f"\n Total images removed: {len(image_names)}")


if __name__ == "__main__":
    main()
    


