import os
import numpy as np
from PIL import Image

def preprocess_binaryclass(image_dir, target_size=(32,32)):
    """
    Preprocess a batch of images from a given directory.

    Parameters:
    - image_dir: str, the directory where the images are located.
    - target_size: tuple, the target size for resizing the images (default is (32, 32)).

    Returns:
    - image_batch: numpy array, preprocessed images ready for model prediction.
    - filenames: list, names of the processed image files.
    """
    
    # traverse the directory 
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg','.jpeg','.png'))]
    
    print(f"Populated {len(image_paths)} images")
    
    # Load and preprocess the images
    preprocessed_images = []
    filenames = []
    for path in image_paths:
        image = Image.open(path)
        image = image.resize(target_size)        #Resizing 
        image_array = np.array(image) / 255.0    #Convert to array + Normalization
        preprocessed_images.append(image_array)
        filenames.append(os.path.basename(path))

    image_batch = np.array(preprocessed_images)  #Reshaping image array to shape (number_of_images, 32, 32, 3)

    return image_batch, filenames
    

def preprocess_multiclass(image_dir, target_size=(32,32)):
    """
    Preprocess a batch of images from a given directory, containing subfolders named after each class.

    Parameters:
    - image_dir: str, the directory where the images are located.
    - target_size: tuple, the target size for resizing the images (default is (32, 32)).

    Returns:
    - image_batch: numpy array, preprocessed images ready for model prediction.
    - filenames: list, names of the processed image files.
    """
    image_paths = []

    for sub_dir in os.listdir(image_dir):
        sub_dir_path = os.path.join(image_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for filename in os.listdir(sub_dir_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(sub_dir_path, filename)
                    image_paths.append(image_path)
                    
    print(f"Populated {len(image_paths)} images")
    
    # Load and preprocess the images
    preprocessed_images = []
    filenames = []
    for path in image_paths:
        image = Image.open(path)
        image = image.resize(target_size)        #Resizing 
        image_array = np.array(image) / 255.0    #Convert to array + Normalization
        preprocessed_images.append(image_array)
        filenames.append(os.path.basename(path))

    image_batch = np.array(preprocessed_images)  #Reshaping image array to shape (number_of_images, 32, 32, 3)

    return image_batch, filenames