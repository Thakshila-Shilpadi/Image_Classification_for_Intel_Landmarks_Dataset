# 🌍 Image Classification for Intel Landmarks Dataset 🏛️

This project demonstrates the use of deep learning techniques to classify images from the Intel Landmarks Dataset. The dataset contains various images of landmarks such as buildings, forests, streets, and more. Using a Convolutional Neural Network (CNN), we aim to train the model to recognize and categorize these images into their respective classes.

## 📌 Project Overview

The main goal of this project is to build an image classification model that can accurately predict the class of an image. With a CNN, we leverage powerful techniques to train the model on labeled images, providing it with the ability to classify unseen data.

## 🔑 Key Features:
Convolutional Neural Network (CNN): 🧠 Used to build the image classification model.
Multi-class Classification: 🎨 The model is trained to classify images into multiple categories.
Intel Landmarks Dataset: 🏙️ A rich collection of images, including buildings, streets, and forests, used to train and test the model.

## 🛠️ Tech Stack

Python 🐍
Keras (for building and training the CNN)
TensorFlow (backend for Keras)
Matplotlib 📊 (for visualizations)
NumPy, Pandas 📅 (for data manipulation)
Scikit-Learn ⚙️ (for model evaluation and splitting datasets)

## 📚 Steps Involved:

Data Preprocessing: 🧹 Loading the dataset, resizing images, and normalizing the pixel values.
Model Architecture: 🏗️ Building the CNN with layers like Conv2D, MaxPooling2D, and Dense layers.
Model Compilation and Training: ⚡ Compiling the model with categorical_crossentropy loss function, Adam optimizer, and accuracy as the evaluation metric.
Model Evaluation and Prediction: 🔍 Testing the trained model on unseen images and evaluating performance.

## 🚀 How to Use:

1. Clone the repository:
   git clone https://github.com/thakshilashilpadi/Intel-Image-Classification.git
2. Install the required libraries:
   pip install -r requirements.txt
3. Run the notebook: Open the Intel_Image_Classification.ipynb notebook and follow the steps to train the model and generate predictions.

## 📊 Results:

The model achieves high accuracy in classifying various landmarks, demonstrating the effectiveness of CNNs in image classification tasks. You can visualize the training and validation accuracy, as well as the loss during each epoch.

## 🔮 Future Work:

Experiment with more complex architectures like ResNet or VGG. 🏗️
Implement data augmentation techniques to improve model robustness. 🔄
Test the model on additional datasets like satellite or drone images. 🚁
