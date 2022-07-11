import glob, os


def load_filenames(file):
    names_list = []
    with open(file, 'r') as f:
        for line in f:
            image_name = line.rstrip()
            names_list.append(image_name.lstrip('data/obj/'))
    return names_list


# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

current_dir = 'data/obj'

# Create and/or truncate train.txt and test.txt
file_train = open('data/train.txt', 'w')
    
# Load filenames of images to skip (the ones in 'test.txt')
imgs_to_skip = load_filenames('data/test.txt')

# Populate train.txt
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if (title + ext) not in imgs_to_skip:
        file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        
file_train.close()