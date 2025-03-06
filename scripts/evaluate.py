import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess_data

# Charger le modèle
model = tf.keras.models.load_model("models/emotion_model.h5")

# Charger les données de test
X_test, y_test = load_and_preprocess_data("dataset/fer2013.csv")

# Prédictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Rapport de classification
print(classification_report(y_test, y_pred))
