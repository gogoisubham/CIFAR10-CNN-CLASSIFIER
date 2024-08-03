# CIFAR-10 CNN Classifier

This project implements Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. It demonstrates both binary classification and multiclass classification using TensorFlow and Keras. The project is designed to help you understand how to build, train, evaluate, and save a CNN model.

## Project Structure
CIFAR10-CNN-Classifier/
├── README.md
├── data/
│   └── download_data.py  # If you want to include data download scripts
├── notebooks/
│   └── cnn_classification.ipynb  # Jupyter notebook with all code
├── src/
│   ├── model_sequential.py  # Code for the Sequential model
│   ├── model_functional.py  # Code for the Functional model
│   └── train.py  # Training scripts
├── models/
│   └── cifar10_model.h5  # Saved model
└── results/
└── training_history.png  # Plots of training history

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

### Installing

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/CIFAR10-CNN-Classifier.git
    cd CIFAR10-CNN-Classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

#### Training the Model

You can run the Jupyter notebook or the Python scripts to train the model.

1. **Jupyter Notebook**:
    Open `cnn_classification.ipynb` in JupyterLab or Jupyter Notebook and run the cells sequentially to train and evaluate the model.

2. **Python Scripts**:
    Run the following command to train the model using the provided script:
    ```bash
    python src/train.py
    ```

#### Evaluating the Model

The evaluation is done as part of the training process. The model's accuracy and loss on the validation set will be printed and plotted.

### Predicting on New Data

To use the trained model to predict on new, unseen data, you can use the following code snippet:

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('models/cifar10_model.h5')

# Assuming new_data is your new data array
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes)

Project Description

Binary Classification

• Dataset: CIFAR-10
• Classes: Airplane vs. Not Airplane
• Model: Convolutional Neural Network (CNN) built using the Sequential API
• Loss Function: Binary Cross-Entropy
• Optimizer: Adam

Multiclass Classification

• Dataset: CIFAR-10
• Classes: 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
• Model: Convolutional Neural Network (CNN) built using the Functional API
• Loss Function: Categorical Cross-Entropy
• Optimizer: Adam

Visualizing Data

The project includes code to visualize the CIFAR-10 images along with their class labels. This helps in understanding the dataset better.

import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(Y_train[i])])
plt.show()

Authors

• Subham Gogoi - https://github.com/gogoisubham