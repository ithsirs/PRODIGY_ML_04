# Task -04

Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems.

## Hand Gesture Recognition with CNN

This repository contains a Jupyter Notebook for building a **Hand Gesture Recognition** model using Convolutional Neural Networks (CNNs). The project utilizes the [Hand Gesture Recognition dataset](https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition) from Kaggle to classify static hand gestures captured via a Leap Motion sensor.

## 📁 Project Structure

- `hand_gesture_final.ipynb`: Main notebook that includes data loading, preprocessing, model creation, training, evaluation, and visualization.
- (Optional) `real_time_gesture_recognition.py`: Script for real-time gesture prediction using a webcam (to be included in the repo if available).

## 📌 Key Features

- Uses `image_dataset_from_directory` for efficient dataset loading.
- CNN architecture built using TensorFlow/Keras.
- Includes preprocessing steps like resizing, normalization, and label encoding.
- Performance metrics such as accuracy and loss visualized across epochs.
- Model evaluation on test data for gesture classification.

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn (for evaluation metrics)

## 📊 Dataset

**Hand Gesture Recognition Dataset**
The hand gesture recognition dataset was created by subtracting the background from the hand images using OpenCV.

The dataset contains of 10 classes: [call_me, rock_on, fingers_crossed, okay, paper, peace, rock, scissor, thumbs, up]

Each class consists of around 500 images.


[Kaggle Dataset Link]([https://www.kaggle.com/datasets/gti-upm/leapgestrecog](https://www.kaggle.com/datasets/roobansappani/hand-gesture-recognition))

## 🧱 Model Description

The Convolutional Neural Network (CNN) model used in this project follows a straightforward architecture designed for image classification tasks:

- **Input Layer**: Accepts 64x64 resized grayscale images.
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation and increasing filter sizes to capture spatial hierarchies in gesture patterns.
- **MaxPooling Layers**: Reduce dimensionality and extract dominant features.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers with dropout for regularization.
- **Output Layer**: A dense layer with softmax activation to classify gestures into 10 categories.

The model is trained using the Adam optimizer and categorical crossentropy loss function, suitable for multi-class classification.

## 🚀 How to Run

1. Clone this repository.
2. Download the dataset from Kaggle and extract it into a directory (e.g., `data/`).
3. Open `hand_gesture_final.ipynb` in Jupyter or Google Colab.
4. Run all cells to train the model and visualize results.

