from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
import io

app = FastAPI()

# Charger le modèle
model = tf.keras.models.load_model("models/emotion_model.h5")
emotions = ["Colère", "Dégoût", "Peur", "Joie", "Tristesse", "Surprise", "Neutre"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48)) / 255.0
    img = img.reshape(1, 48, 48, 1)

    prediction = model.predict(img)
    emotion = emotions[np.argmax(prediction)]

    return {"emotion": emotion}
