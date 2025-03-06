import sys
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model (Updated Path)
MODEL_PATH = "models/emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion labels (same order as dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(image_path):
    """Preprocess input image to match model requirements (48x48 grayscale)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        raise ValueError(f"Error: Cannot read image {image_path}")
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (48,48,1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1,48,48,1)
    return img

def predict_emotion(image_path):
    """Predict the emotion from the input image."""
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_emotion = emotion_labels[np.argmax(predictions)]  # Get the highest probability label
    return predicted_emotion

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    emotion = predict_emotion(image_path)
    print(f"Predicted Emotion: {emotion}")
