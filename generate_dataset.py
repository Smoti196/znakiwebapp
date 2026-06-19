import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

SOURCE_DIR = "dataset"
OUTPUT_DIR = "generated_dataset"

IMG_SIZE = (224, 224)
AUG_PER_CLASS = 500

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    if len(images) == 0:
        print(f"❌ PUSTA KLASA: {class_name}")
        continue

    # bierzemy pierwsze zdjęcie wzorcowe
    img_path = os.path.join(class_path, images[0])

    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    save_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"➡ Generuję: {class_name}")

    i = 0

    for batch in datagen.flow(
        x,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix="aug",
        save_format="png"
    ):
        i += 1
        if i >= AUG_PER_CLASS:
            break

    print(f"✔ Zrobiono: {class_name} ({i} obrazów)")

print("🎉 GOTOWE - dataset wygenerowany")