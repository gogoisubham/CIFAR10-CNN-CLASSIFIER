import os 
import numpy as np

def auto_label_images(parent_dir):
    '''
    Automatically labels images based on their directory structure
    
    Parameters
    - parent_dir: str, path to the parent directory containing subfolders named after classes
    
    Returns:
    - true_labels: numpy array, the labels corresponding to the images.
    - filenames: list, the filenames of the images.
    '''
    class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    class_labels = {class_name:index for index, class_name in enumerate(class_names)}
    
    filenames = []
    true_labels = []
    
    # Traverse through each subfolder and label images
    for class_name in class_names:
        sub_dir = os.path.join(parent_dir, class_name)
        if os.path.exists(sub_dir):
            for filename in os.listdir(sub_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(sub_dir, filename)
                    true_labels.append(class_labels[class_name])
                    filenames.append(os.path.basename(file_path))
                    
    return np.array(true_labels), filenames