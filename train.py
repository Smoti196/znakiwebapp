import tensorflow as tf
from tensorflow.keras import layers
import pathlib

DATASET_DIR = pathlib.Path("generated_dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# 📊 wczytanie danych
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
print("KLASY:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

# 🔥 NORMALIZACJA (ważne!)
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize).prefetch(AUTOTUNE)
val_ds = val_ds.map(normalize).prefetch(AUTOTUNE)

# 🧠 model bazowy
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# 🏗️ MODEL (POPRAWNA STRUKTURA)
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# ⚙️ kompilacja
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
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