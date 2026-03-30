import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# SETTINGS
# ==========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
DATASET_PATH = "final_dataset"

# ==========================
# LOAD DATASET
# ==========================
train_dataset = image_dataset_from_directory(
    DATASET_PATH + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    DATASET_PATH + "/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print("Classes:", class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================
# DATA AUGMENTATION
# ==========================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ==========================
# LOAD PRETRAINED MODEL
# ==========================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # 🔥 IMPORTANT (this gave 94%)

# ==========================
# BUILD MODEL
# ==========================
model = models.Sequential([
    data_augmentation,
    layers.Lambda(preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# TRAIN MODEL
# ==========================
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# ==========================
# SAVE MODEL
# ==========================
model.save("waste_classifier_model.keras")

print("✅ Model training complete and saved!")