# CIFAR-10 CNN Image Classifier

This project implements Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. It demonstrates both binary and multiclass classification using TensorFlow and Keras. The project is organized with separate subfolders for binary and multiclass models and includes Python scripts for image preprocessing and labeling.

## Project Structure
```plaintext
CIFAR10-CNN-Classifier/
├── README.md                       # Project documentation
├── image_labelling.py              # Script for auto-labeling images based on directory structure
├── image_preprocessing.py          # Script for preprocessing images for model prediction
├── MobileNetV2/                    # Folder containing notebooks and models for multiclass classification
│   ├── multiclassModel.ipynb       # Jupyter notebook for training the multiclass model
│   ├── multiclassModelPredict.ipynb # Jupyter notebook for predicting with the multiclass model
│   └── mobileNetV2.h5              # Trained MobileNetV2 model for multiclass classification
└── Sequential/                     # Folder containing notebooks and models for binary classification
    ├── binaryModel.ipynb            # Jupyter notebook for training the binary model
    ├── binaryModelPredict.ipynb     # Jupyter notebook for predicting with the binary model
    └── binary-model.h5              # Trained Sequential model for binary classification
```

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)

### Installing

Clone the repository:

    ```bash
    git clone https://github.com/yourusername/CIFAR10-CNN-Classifier.git
    cd CIFAR10-CNN-Classifier
    ```

### Running the Code

#### Binary Classification

1. **Training the Binary Model**:
   - Open `binaryModel.ipynb` in the `Sequential/` folder and run all cells to train, evaluate and save the binary classification model.

2. **Predicting with the Binary Model**:
   - Open `binaryModelPredict.ipynb` in the `Sequential/` folder and run the appropriate cells to predict on your downloaded test images using the trained binary model.

#### Multiclass Classification

1. **Training the Multiclass Model**:
   - Open `multiclassModel.ipynb` in the `MobileNetV2/` folder and run all cells to train, evaluate and save the multiclass classification model.

2. **Predicting with the Multiclass Model**:
   - Open `multiclassModelPredict.ipynb` in the `MobileNetV2/` folder and run the appropriate cells to predict on your downloaded test images using the trained multiclass model.

### Project Description

#### Binary Classification

- **Dataset**: CIFAR-10
- **Classes**: Airplane vs. Not Airplane
- **Model**: Convolutional Neural Network (CNN) built using the Sequential API
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam

#### Multiclass Classification

- **Dataset**: CIFAR-10
- **Classes**: 10 classes (Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **Model**: Convolutional Neural Network (CNN) built using the Functional API with MobileNetV2 as the base model
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam

### Image Preprocessing

Use the `image_preprocessing.py` script to preprocess images for prediction. The script provides functions to resize and normalize images before feeding them to the loaded model.

### Auto-Labeling

Use the `image_labelling.py` script to automatically label images based on their directory structure. The script traverses subfolders named after each class and assigns labels accordingly.

### Visualizing Data

The project includes code to visualize the CIFAR-10 images along with their class labels. This helps in understanding the dataset better.

```python
import matplotlib.pyplot as plt

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(Y_train[i])])
plt.show()
```

### Author

- **Subham Gogoi** - [GitHub](https://github.com/gogoisubham)

For any questions or suggestions, please reach out to subhgogoi@gmail.com.