from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model
model = tf.keras.models.load_model("models/emotion_model.h5")
emotions = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Neutral"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Log the file details
        logger.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

        # Read the file
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        # Check if the image was decoded correctly
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocess the image
        img = cv2.resize(img, (48, 48)) / 255.0
        img = img.reshape(1, 48, 48, 1)

        # Make a prediction
        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]

        return {"emotion": emotion}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))