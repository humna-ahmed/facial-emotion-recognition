# Facial Emotion Recognition System

A Convolutional Neural Network (CNN) based system that recognizes human emotions from facial expressions using the FER-2013 dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)

## Features

- **Emotion Classification**: Identifies 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Real-time Detection**: Webcam-based live emotion detection
- **High Accuracy**: CNN model trained on FER-2013 dataset
- **Data Augmentation**: Enhanced training with image transformations

## Dataset

This project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) which contains:
- 28,709 training images
- 7,178 test images
- 48x48 pixel grayscale images
- 7 emotion categories

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition

2. Install dependencies:

bash
pip install -r requirements.txt

3. Download the FER-2013 dataset from Kaggle and place it in the data/ folder with the structure:

text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/

##Usage

###Training the Model
bash
python src/emotion_recognition.py

###Real-time Emotion Detection
bash
python src/webcam_detection.py

###Using Pre-trained Model
Download the pre-trained model and place it in the models/ folder, then run the webcam detection.


##Model Architecture

###The CNN architecture includes:

3 Convolutional blocks with Batch Normalization

MaxPooling layers for dimensionality reduction

Dropout layers for regularization

Fully connected layers for classification


###Results

The model achieves approximately 60-70% accuracy on the test set, which is competitive for the FER-2013 dataset.

###Project Structure
text
facial-emotion-recognition/
├── models/                 # Saved models
├── data/                  # Dataset
├── src/                   # Source code
│   ├── emotion_recognition.py  # Training script
│   └── webcam_detection.py     # Real-time detection
├── requirements.txt       # Dependencies
└── README.md             # Project documentation

###Contributing

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request


###License
This project is licensed under the MIT License - see the LICENSE file for details.


###Acknowledgments
FER-2013 dataset providers

TensorFlow and Keras teams

OpenCV community



## 4. Organize Your Code

### src/emotion_recognition.py
Use your existing code but make sure it's the corrected version without errors.

### src/webcam_detection.py
```python
import tensorflow as tf
import cv2
import numpy as np
import os

class EmotionPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.img_size = 48
    
    def predict_emotion(self, image):
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype('float32') / 255.0
        image = image.reshape(1, self.img_size, self.img_size, 1)
        
        # Make prediction
        prediction = self.model.predict(image, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        confidence = prediction[emotion_idx]
        
        return self.emotion_labels[emotion_idx], confidence

def detect_emotion_from_webcam():
    """
    Real-time emotion detection using webcam
    """
    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load trained model
    model_path = os.path.join('models', 'fer_emotion_model.h5')
    try:
        predictor = EmotionPredictor(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure the model file exists in the models/ folder")
        return
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit the webcam feed.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence = predictor.predict_emotion(face_roi)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion_from_webcam()