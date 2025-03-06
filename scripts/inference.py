import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

emotion_labels = sorted(os.listdir("dataset/train")) 


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=-1)  
    return img


def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion_idx = np.argmax(prediction)
    emotion = emotion_labels[emotion_idx]
    return emotion

if __name__ == "__main__":
    test_image = "test_sample.jpg" 
    if os.path.exists(test_image):
        predicted_emotion = predict_emotion(test_image)
        print(f"Predicted Emotion: {predicted_emotion}")
    else:
        print("❌ Test image not found! Place an image named 'test_sample.jpg' in the same folder.")
