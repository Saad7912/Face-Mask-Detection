# Face-Mask-Detection
Face Mask Detection
This project implements a face mask detection algorithm using a convolutional neural network (CNN). The model is trained to classify images of people wearing masks and not wearing masks.

# Key Libraries and Techniques
Libraries:
TensorFlow
Keras
NumPy
Matplotlib
OpenCV
PIL (Pillow)
scikit-learn
Kaggle API
# Techniques:
Convolutional Neural Network (CNN)
Data Augmentation
Early Stopping
Image Preprocessing
# Model Architecture
Convolutional Layers: Extract features from images
MaxPooling Layers: Reduce spatial dimensions
Flatten Layer: Convert 2D feature maps to 1D
Dense Layers: Classify images
Dropout Layers: Prevent overfitting
Output Layer: Softmax activation for classification
# Training and Evaluation
Data Augmentation: Improve model generalization
Early Stopping: Prevent overfitting by monitoring validation loss
Performance Metrics: Accuracy on test data
# Prediction
The trained model can be used to predict whether a person in an image is wearing a mask or not.
For detailed implementation, refer to the project code.
