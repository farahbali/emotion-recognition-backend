import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

# Define dataset paths
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image properties
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs for better convergence

# Define emotion categories based on dataset structure
emotion_labels = sorted(os.listdir(TRAIN_DIR))  # ["angry", "disgust", "fear", ...]

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Increased rotation range
    width_shift_range=0.2,  # Increased shift range
    height_shift_range=0.2,
    shear_range=0.2,  # Increased shear range
    zoom_range=0.2,  # Increased zoom range
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training & test data
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

# Calculate class weights for imbalanced datasets
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Model Architecture
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(512, (3,3), activation="relu"),
    BatchNormalization(),
    GlobalAveragePooling2D(),  # Reduces overfitting compared to Flatten

    Dense(256, activation="relu"),
    Dropout(0.6),  # Increased dropout rate
    Dense(len(emotion_labels), activation="softmax")  # Output layer
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("emotion_model.h5", save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7)

# Train the Model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    class_weight=class_weights_dict  # Use class weights for imbalanced data
)

# Save model
model.save("emotion_model.h5")
print("✅ Model training complete. Saved as 'emotion_model.h5'")