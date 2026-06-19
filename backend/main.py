import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

# -------------------------
# APP
# -------------------------
app = FastAPI()

# -------------------------
# ŚCIEŻKI (KLUCZOWE FIXY)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_znaki.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")

# -------------------------
# MODEL
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# CLASS NAMES
# -------------------------
with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

# -------------------------
# PREPROCESS IMAGE
# -------------------------
def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))  # zmień jeśli Twój model ma inne wejście
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# -------------------------
# ENDPOINT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = preprocess_image(image_bytes)
    prediction = model.predict(img)

    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "class": class_names[class_id],
        "confidence": confidence
    }

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------
# RUN (Render friendly)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)