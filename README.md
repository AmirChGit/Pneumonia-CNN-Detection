This project aims to streamline the process of building and training an image classification machine learning model for pneumonia detection using Deep Learning technics, TensorFlow and Keras libraries. This kernel provides a comprehensive guide for implementing image classification. Its concise format serves as a time-saving resource for practitioners in the field of medical image analysis.


Kaggle : https://www.kaggle.com/code/amirchachoui/pneumonia-convolutional-neural-network-detection
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pneumonia Detection using Convolutional Neural Network
![image](https://github.com/user-attachments/assets/27e32a18-f6ff-448f-b6d7-d16a646247ab)

## Objective
This project aims to streamline the process of building and training an image classification machine learning model for pneumonia detection using Deep Learning techniques, TensorFlow, and Keras libraries. This guide provides a comprehensive implementation of image classification in a concise format, serving as a resource for practitioners in the field of medical image analysis.

## Table of Contents
1. [Introduction](#introduction)
   - 1.1 [Pneumonia](#pneumonia)
   - 1.2 [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
2. [Libraries](#libraries)
   - 2.1 [Import Libraries](#import-libraries)
3. [Data Loading and Exploration](#data-loading-and-exploration)
   - 3.1 [Loading the Dataset](#loading-the-dataset)
   - 3.2 [Processing Train Dataset](#processing-train-dataset)
   - 3.3 [Processing Valid Dataset](#processing-valid-dataset)
   - 3.4 [Processing Test Dataset](#processing-test-dataset)
   - 3.5 [Splitting the Dataset](#splitting-the-dataset)
   - 3.6 [Image Data Generator](#image-data-generator)
   - 3.7 [Data Sample Visualization](#data-sample-visualization)
   - 3.8 [Exploring the Class Distribution](#exploring-the-class-distribution)
4. [Model Creation and Training](#model-creation-and-training)
   - 4.1 [Model Structure](#model-structure)
   - 4.2 [Model Training](#model-training)
   - 4.3 [Results Interpretation](#results-interpretation)
5. [Conclusion](#conclusion)
6. [References](#references)

## 1. Introduction

### 1.1 Pneumonia
Pneumonia is an inflammatory lung condition affecting the alveoli, causing symptoms like cough, chest pain, fever, and breathing difficulties. It is primarily caused by viral or bacterial infections, with risk factors including respiratory conditions, diabetes, smoking, and weakened immune systems. Diagnosis involves physical examination, chest X-rays, blood tests, and sputum cultures.

### 1.2 Convolutional Neural Networks (CNNs)
CNNs are specialized neural networks designed for processing 2D matrix-like data, such as images. They are commonly employed for tasks like image detection and classification.

## 2. Libraries

### 2.1 Import Libraries
Essential libraries for data manipulation, visualization, machine learning, and deep learning:
- **os**: For file directory operations.
- **PIL**: Image processing.
- **itertools**: Efficient looping.
- **cv2**: Computer vision tasks.
- **numpy**: Numerical operations and array manipulation.
- **pandas**: Data analysis and manipulation.
- **matplotlib.pyplot**: Data visualization.
- **seaborn**: Statistical data visualization.
- **warnings**: Manage warning messages.
- **tensorflow**: Deep learning framework.
- **keras**: High-level neural network API integrated into TensorFlow.
- **scikit-learn**: Model evaluation and data preprocessing tools.

## 3. Data Loading and Exploration

### 3.1 Loading the Dataset
The dataset containing chest X-ray images is loaded from directories for training, validation, and testing:
- **train_data**: Training the model.
- **valid_data**: Validating model performance during training.
- **test_data**: Evaluating the final model.

### 3.2 Processing Train Dataset
- **ImageDataGenerator**: Rescales pixel values to (1/255) for faster convergence.
- **File Paths and Labels**: Associates each image with a class label.
- **DataFrame Creation**: Organizes data into a DataFrame for training.

### 3.3 Processing Valid Dataset
- **ImageDataGenerator**: Rescales pixel values.
- **File Paths and Labels**: Retrieves file paths and labels.
- **DataFrame Creation**: Organizes data into a validation DataFrame.

### 3.4 Processing Test Dataset
- **ImageDataGenerator**: Rescales pixel values uniformly.
- **File Paths and Labels**: Associates images with labels.
- **DataFrame Creation**: Simplifies data management for evaluation.

### 3.5 Splitting the Dataset
Data is split into:
- **Training Data**: For parameter training.
- **Validation Data**: For performance evaluation.
- **Test Data**: For final assessment.

The `train_test_split()` function is used to ensure randomness and reproducibility.

### 3.6 Image Data Generator
Generates image batches for training, validation, and testing:
- **Batch Size**: Number of samples per gradient update.
- **Image Size**: Target size for input images.
- **train_gen**: Generates augmented training data.
- **valid_gen**: Generates validation data.
- **test_gen**: Generates test data without shuffling.

### 3.7 Data Sample Visualization
Visualizes a batch of images:
- **gen_dict**: Class indices from `train_gen`.
- **classes**: Class names.
- **Plotting**: Displays images in a 4x4 grid.

### 3.8 Exploring the Class Distribution
Explores the distribution of classes in the training data.

## 4. Model Creation and Training

### 4.1 Model Structure
A CNN model for image classification:
- **Input Shape**: (224, 224, 3)
- **Convolutional Layers**: Feature extraction with filters (64, 128, 256, 512).
- **MaxPooling Layers**: Reduces spatial dimensions.
- **Flatten Layer**: Prepares data for dense layers.
- **Dense Layers**: Two fully connected layers.
- **Output Layer**: Binary classification (2 neurons).
- **Total Parameters**: 21,154,050 (all trainable).

### 4.2 Model Training
Training setup:
- **Epochs**: 13 iterations.
- **Verbose**: Detailed training progress.
- **Validation Data**: Evaluated after each epoch.

#### Results:
- **Train Loss**: 0.0246, **Train Accuracy**: 99.1%
- **Validation Loss**: 0.0989, **Validation Accuracy**: 96.9%
- **Test Loss**: 0.0741, **Test Accuracy**: 97.7%

### 4.3 Results Interpretation
Performance metrics:
- **Precision**: 96% for NORMAL, 98% for PNEUMONIA.
- **Recall**: 96% for NORMAL, 98% for PNEUMONIA.
- **F1-Score**: 95% for NORMAL, 98% for PNEUMONIA.
- **Overall Accuracy**: 97%.

## 5. Conclusion
The CNN model demonstrates robust performance in pneumonia detection, with high accuracy, precision, and recall. It minimizes false positives and false negatives, showing potential for clinical application. Further refinement and validation are needed for real-world deployment.

## 6. References
- **Data Source**: Mendeley Dataset
- **License**: CC BY 4.0
- **Citation**: Cell
- **Fastai MOOC and Fastai Library**
