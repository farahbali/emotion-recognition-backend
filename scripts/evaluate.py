import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
MODEL_PATH = "emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define dataset paths
DATASET_DIR = "dataset"
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image properties
IMG_SIZE = 48
BATCH_SIZE = 32

# Define emotion categories based on dataset structure
emotion_labels = sorted(os.listdir(TEST_DIR))

# Data Generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Evaluate Model
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("✅ Model Evaluation Results:")
print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
