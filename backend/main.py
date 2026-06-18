from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()

# pozwala telefonowi łączyć się z API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 wczytanie modelu (NA RAZIE MOŻE NIE ISTNIEĆ)
# model = tf.keras.models.load_model("model_znaki.keras")

# klasy znaków (muszą pasować do treningu)
class_names = [
    "droga-dla-pieszych",
    "droga-dla-pieszych-i-rowerow",
    "droga-dla-rowerow",
    "koniec-drogi-dla-pieszych",
    "koniec-drogi-dla-rowerow",
    "koniec-minimalnej-predkosci",
    "koniec-nakazu-uzywania-lancuchow-przeciwposlizgowych",
    "nakazany-kierunek-jazdy-dla-pojazdow-z-materialami-niebezpiecznymi",
    "nakaz-jazdy-prosto",
    "nakaz-jazdy-prosto-lub-w-lewo",
    "nakaz-jazdy-prosto-lub-w-prawo",
    "nakaz-jazdy-w-lewo-lub-w-prawo",
    "nakaz-jazdy-w-lewo-przed-znakiem",
    "nakaz-jazdy-w-lewo-za-znakiem",
    "nakaz-jazdy-w-prawo-przed-znakiem",
    "nakaz-jazdy-w-prawo-za-znakiem",
    "nakaz-jazdy-z-lewej-strony-znaku",
    "nakaz-jazdy-z-prawej-lub-lewej-strony-znaku",
    "nakaz-jazdy-z-prawej-strony-znaku",
    "nakaz-uzywania-lancuchow-przeciwposlizgowych",
    "wskazanie-strony-drogi-dla-pieszych-i-rowerow",
    "predkosc-minimalna",
    "ruch-okrezny"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    return {
        "label": "TEST_ZNAK (backend działa)",
        "confidence": 0.99
    }
    contents = await file.read()

    # zamiana pliku na obraz
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # przygotowanie obrazu dla AI
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # predykcja
    prediction = model.predict(img)
    index = np.argmax(prediction)

    return {
        "label": class_names[index],
        "confidence": float(np.max(prediction))
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)