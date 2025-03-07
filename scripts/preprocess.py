import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = 48
BATCH_SIZE = 32


emotion_labels = sorted(os.listdir(TRAIN_DIR)) 


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

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

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

def save_preprocessed_data():
    X_train, y_train = [], []
    X_test, y_test = [], []

    
    for emotion in emotion_labels:
        folder_path = os.path.join(TRAIN_DIR, emotion)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img) 
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            X_train.append(img)
            y_train.append(emotion_labels.index(emotion))

  
    for emotion in emotion_labels:
        folder_path = os.path.join(TEST_DIR, emotion)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.equalizeHist(img)  
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            X_test.append(img)
            y_test.append(emotion_labels.index(emotion))

   
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    np.savez("processed_dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("✅ Preprocessed dataset saved as 'processed_dataset.npz'.")

if __name__ == "__main__":
    save_preprocessed_data()