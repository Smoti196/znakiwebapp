import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

DATASET_DIR = pathlib.Path("generated_dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# 📊 wczytanie danych
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("KLASY:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# 🧠 model bazowy (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# 🏗️ budowa modelu
model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

# ⚙️ kompilacja
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🚀 trening
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 💾 zapis modelu
model.save("model_znaki.keras")

# 🏷️ zapis klas
with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("MODEL ZAPISANY ✔")