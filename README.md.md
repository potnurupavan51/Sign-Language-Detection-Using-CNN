# **Sign Language Detection Using CNN**

## **Overview**

This project implements a Convolutional Neural Network (CNN) to recognize and classify sign language gestures from images. It provides a way to translate sign language into text, which can help improve communication between deaf and hearing individuals.

## **Features**

* **Image Classification:** Uses a CNN model to classify sign language gestures from images.  
* **TensorFlow/Keras:** Built using TensorFlow and Keras, popular open-source libraries for deep learning.  
* **Image Preprocessing:** Includes image loading and preprocessing steps to ensure compatibility with the CNN model.  
* **Prediction:** Provides functionality to load a trained model and predict the sign represented in a given image.

## **Technical Details**

* **Model Architecture:**  
  * Convolutional Neural Network (CNN)  
  * The model architecture consists of convolutional layers (conv2d), and likely includes other layers like pooling, and dense layers for classification. The first convolutional layer in the provided code is named "conv2d\_5". More details would be needed from the training script to fully document the architecture.  
* **Dataset:**  
  * The code loads an image from a specified path ("/content/A.png"). To provide a complete description, the training dataset should be described here. If it is a standard dataset, provide a link. If it's custom, describe how it was collected/created.  
* **Preprocessing:**  
  * Image resizing to 48x48 pixels.  
  * Pixel normalization to the range \[0, 1\].  
* Libraries:  
  \* TensorFlow  
  \* Keras  
  \* NumPy  
  \* PIL (implicitly used by Keras image utils)

## **How to Use**

1. **Prerequisites:**  
   * Install Python (e.g., Python 3.9 or later).  
   * Install the required libraries:  
     pip install tensorflow keras numpy Pillow

2. **Download the code:**  
   * Download the code and model file.  
3. **Prepare your image:**  
   * Save the image you want to predict. The provided code uses "/content/A.png" as an example.  
4. **Run the prediction script:**  
   * Open a Python environment or terminal.  
   * Run the Python script, ensuring that the paths to the model and image are correct:  
     import tensorflow as tf  
     from tensorflow.keras.models import load\_model  
     from tensorflow.keras.preprocessing import image  
     import numpy as np

     \# Load your trained model  
     model \= load\_model('/content/drive/MyDrive/signlanguage\_model2.h5')

     \# Path to the image you want to predict  
     img\_path \= '/content/A.png'

     \# Load and preprocess the image  
     img \= image.load\_img(img\_path, target\_size=(48, 48))  
     img\_array \= image.img\_to\_array(img)  
     img\_array \= np.expand\_dims(img\_array, axis=0)  
     img\_array /= 255.0

     \# Make the prediction  
     predictions \= model.predict(img\_array)

     \# Get the predicted class index  
     predicted\_class\_index \= np.argmax(predictions, axis=1)

     \# If you have class labels, you can map the index to the class name  
     class\_labels \= \['A', 'M', 'N', 'S', 'T', 'blank'\]  
     predicted\_class\_label \= class\_labels\[predicted\_class\_index\[0\]\]

     print(f"Predicted class index: {predicted\_class\_index}")  
     print(f"Predicted class label: {predicted\_class\_label}")  
     print(f"Probability scores: {predictions}")

     Replace the model path ('/content/drive/MyDrive/signlanguage\_model2.h5') and image path ('/content/A.png') with the actual paths to your files.  
5. **View the output:**  
   * The script will print the predicted sign language gesture label, its index, and the probability scores.

## **Model File**

* The code assumes the model is saved at '/content/drive/MyDrive/signlanguage\_model2.h5'. You will need to provide your own trained model file.

## **Image File**

* The code expects the input image to be at '/content/A.png'. You can change this to the path of your image. The image should be a sign language gesture.

## **Class Labels**

* The code defines the class labels as \['A', 'M', 'N', 'S', 'T', 'blank'\]. Ensure these match the classes your model was trained to recognize, and that they are in the correct order.

## **Further Improvements**

* To make this README more complete, add the following:  
  * **Training Details**: Information about how the model was trained, including the dataset, training parameters, and any data augmentation techniques used.  
  * **Model Architecture Details**: A more detailed description of the CNN architecture.  
  * **Performance Metrics**: Report the model's performance on a test set (e.g., accuracy, precision, recall).