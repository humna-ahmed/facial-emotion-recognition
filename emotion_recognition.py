import tensorflow as tf
# Use:
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
utils = tf.keras.utils
callbacks = tf.keras.callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

class FacialEmotionRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.img_size = 48
        
    def load_image_data(self, data_dir):
        """
        Load images from directory structure
        Expected structure: data_dir/emotion_label/*.jpg
        """
        print(f"Loading images from {data_dir}...")
        
        X = []
        y = []
        
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(data_dir, emotion)
            
            if not os.path.exists(emotion_dir):
                print(f"Warning: Directory {emotion_dir} does not exist. Skipping.")
                continue
                
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                
                try:
                    # Read and preprocess image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img.astype('float32') / 255.0
                    
                    X.append(img)
                    y.append(emotion_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for CNN (add channel dimension)
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        print(f"Loaded {X.shape[0]} samples from {data_dir}")
        print(f"Emotion distribution:")
        for i, emotion in enumerate(self.emotion_labels):
            count = np.sum(y == i)
            print(f"  {emotion}: {count} samples")
        
        return X, y
    
    def prepare_data(self, X_train, y_train, X_test, y_test, val_size=0.2):
        """
        Prepare data for training, including validation split
        """
        # Convert labels to categorical
        y_train_categorical = utils.to_categorical(y_train, num_classes=len(self.emotion_labels))
        y_test_categorical = utils.to_categorical(y_test, num_classes=len(self.emotion_labels))
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train_categorical, test_size=val_size, random_state=42, stratify=y_train
        )
        
        print(f"Data split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test_categorical
    
    def create_cnn_model(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Create a simpler CNN architecture for emotion recognition
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def setup_data_augmentation(self):
        """
        Setup data augmentation for training
        """
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        return datagen
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        """
        Train the CNN model
        """
        # Create model
        self.model = self.create_cnn_model()
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
        # Setup callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Setup data augmentation
        datagen = self.setup_data_augmentation()
        datagen.fit(X_train)
        
        print("Starting training...")
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)[1]
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.emotion_labels))
        
        return y_true, y_pred, test_accuracy
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        return cm
    
    def predict_emotion(self, image):
        """
        Predict emotion for a single image
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype('float32') / 255.0
        image = image.reshape(1, self.img_size, self.img_size, 1)
        
        # Make prediction
        prediction = self.model.predict(image)[0]
        emotion_idx = np.argmax(prediction)
        confidence = prediction[emotion_idx]
        
        return self.emotion_labels[emotion_idx], confidence, prediction
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        """
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and main execution
def main():
    """
    Main function to demonstrate the facial emotion recognition system
    """
    # Initialize the recognizer
    fer = FacialEmotionRecognizer()
    
    try:
        # Load training and test data
        print("Loading training data...")
        X_train, y_train = fer.load_image_data("train")
        
        print("Loading test data...")
        X_test, y_test = fer.load_image_data("test")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = fer.prepare_data(
            X_train, y_train, X_test, y_test
        )
        
        # Train model
        history = fer.train_model(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
        
        # Evaluate model
        y_true, y_pred, accuracy = fer.evaluate_model(X_test, y_test)
        
        # Plot results
        fer.plot_training_history()
        fer.plot_confusion_matrix(y_true, y_pred)
        
        # Save model
        fer.save_model("fer_emotion_model.h5")
        
        print(f"\nFinal Test Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Additional utility functions for real-time emotion detection
def detect_emotion_from_webcam():
    """
    Real-time emotion detection using webcam
    """
    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load trained model
    fer = FacialEmotionRecognizer()
    try:
        fer.load_model("fer_emotion_model.h5")
    except:
        print("Model not found! Please train the model first.")
        return
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
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
            emotion, confidence, _ = fer.predict_emotion(face_roi)
            
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
    main()
    
    # Uncomment the following line after training to try real-time detection
    detect_emotion_from_webcam()