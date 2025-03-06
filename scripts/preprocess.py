import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# Define dataset paths
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image properties
IMG_SIZE = 48  # Target image size (48x48)
BATCH_SIZE = 32

# Define emotion categories based on dataset structure
emotion_labels = sorted(os.listdir(TRAIN_DIR))  # ["angry", "disgust", "fear", ...]

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 48x48
    img = cv2.equalizeHist(img)  # Apply histogram equalization
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (48,48,1)
    return img

# Create ImageDataGenerator for augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Rotate images randomly
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom in and out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode="nearest"  # Fill missing pixels
)

test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for test data

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Calculate class weights to handle imbalanced data
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_generator.classes),
    train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Save processed dataset (if needed)
def save_preprocessed_data():
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Load train images
    for emotion in emotion_labels:
        folder_path = os.path.join(TRAIN_DIR, emotion)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_and_preprocess_image(img_path)
            X_train.append(img)
            y_train.append(emotion_labels.index(emotion))

    # Load test images
    for emotion in emotion_labels:
        folder_path = os.path.join(TEST_DIR, emotion)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_and_preprocess_image(img_path)
            X_test.append(img)
            y_test.append(emotion_labels.index(emotion))

    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Save data
    np.savez("processed_dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("✅ Preprocessed dataset saved as 'processed_dataset.npz'.")

if __name__ == "__main__":
    save_preprocessed_data()