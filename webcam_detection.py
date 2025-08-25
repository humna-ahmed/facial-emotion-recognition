import tensorflow as tf
import cv2
import numpy as np

# Simple class just for prediction
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
    try:
        predictor = EmotionPredictor("fer_emotion_model.h5")
        print("Model loaded successfully!")
    except:
        print("Model not found! Please make sure 'fer_emotion_model.h5' is in the same directory.")
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