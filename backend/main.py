import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_znaki.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.txt")

print("MODEL:", MODEL_PATH)
print("CLASS FILE:", CLASS_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = f.read().splitlines()

print("LICZBA KLAS:", len(class_names))
print("KLASY:", class_names)
print("OUTPUT SHAPE:", model.output_shape)


def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Nie udało się odczytać obrazu")

    print("ORYGINALNY KSZTALT:", img.shape)

    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    print("INPUT SHAPE:", img.shape)
    print("MIN:", img.min())
    print("MAX:", img.max())

    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = preprocess_image(image_bytes)

    prediction = model.predict(img)

    print("RAW PREDICTION:")
    print(prediction)

    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    print("CLASS ID:", class_id)
    print("CLASS NAME:", class_names[class_id])
    print("CONFIDENCE:", confidence)

    return {
        "class": class_names[class_id],
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    return {
        "num_classes": len(class_names),
        "classes": class_names,
        "output_shape": str(model.output_shape)
    }

@app.get("/test-random")
def test():
    x = np.random.rand(1, 224, 224, 3).astype(np.float32)
    pred = model.predict(x)

    return {
        "raw": pred.tolist(),
        "class": class_names[int(np.argmax(pred))]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port
    )
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)