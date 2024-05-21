# Pneumonia Detection using Convolutional Neural Network (CNN)

This project demonstrates how to build a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The model is trained using Keras and TensorFlow, leveraging data augmentation techniques to improve generalization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- TensorFlow and Keras
- pandas, numpy, Pillow, sklearn

## Project Structure
The project directory should have the following structure:

```bash
project-root/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── pneumonia_detection.ipynb
└── README.md
```

## Dataset
The dataset consists of chest X-ray images categorized into two classes:

- NORMAL: Healthy individuals
- PNEUMONIA: Individuals with pneumonia
- 
Ensure the dataset is organized into the **train, test, and val directories** with appropriate subdirectories for each class.

## Data Preprocessing

The preprocessing steps include:

1. Loading Images: Images are loaded from the specified directories.
2. Resizing with Padding: Images are resized to 256x256 pixels while maintaining the aspect ratio, and padding is applied to fill the empty space.
3. Normalization: Pixel values are normalized to the range [0, 1].

## Model Architecture

The CNN model consists of the following layers:

- 3 Convolutional layers with 32, 64, and 128 filters respectively, each followed by a MaxPooling layer.
- A Flatten layer to convert the 2D feature maps into 1D feature vectors.
- A Dense layer with 256 units and ReLU activation, followed by a Dropout layer for regularization.
- An output Dense layer with a single neuron and sigmoid activation for binary classification.
  
Data augmentation is applied to the training data to improve model generalization.

## Training
The model is compiled with the Adam optimizer and binary cross-entropy loss. Early stopping is used to prevent overfitting. The model is trained for up to 20 epochs with a batch size of 32.

## Evaluation
The trained model is evaluated on the test set, and the accuracy is reported.

## Usage

To run the project:

1. Ensure the dataset is organized correctly.
2. Run the Jupyter notebook pneumonia_detection.ipynb.
F3. ollow the steps in the notebook to preprocess the data, build, train, and evaluate the model.

## Acknowledgements
This project is based on publicly available chest X-ray datasets for pneumonia detection. Special thanks to the contributors of the dataset and the developers of the libraries used in this project.
