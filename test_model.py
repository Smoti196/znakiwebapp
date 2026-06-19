import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "model_znaki.keras"
CLASS_PATH = "class_names.txt"
IMG_SIZE = (224, 224)

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# load classes
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

def preprocess(path):
    img = cv2.imread(path)

    # IMPORTANT FIX (OpenCV BGR -> RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

folder = "test_images"

for file in os.listdir(folder):
    path = os.path.join(folder, file)

    img = preprocess(path)
    pred = model.predict(img, verbose=0)

    class_id = np.argmax(pred)
    confidence = np.max(pred)

    print("\nFILE:", file)
    print("CLASS:", class_names[class_id])
    print("CONFIDENCE:", confidence)