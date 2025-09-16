import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# Settings
img_size = (150, 150)
batch_size = 32

train_dir = "./dataset/train"
test_dir = "./dataset/test"

# Data generators - CHANGED TO CATEGORICAL
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"  # CHANGED FROM "binary"
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"  # CHANGED FROM "binary"
)

print("Found classes:", train_gen.class_indices)

# Pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150,150,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze layers

# CHANGED: 2 output neurons with softmax
num_classes = train_gen.num_classes  # auto-detect classes

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")  # CHANGED FROM Dense(1, sigmoid)
])

# CHANGED: categorical_crossentropy
model.compile(
    optimizer="adam", 
    loss="categorical_crossentropy",  # CHANGED FROM "binary_crossentropy"
    metrics=["accuracy"]
)

# Train quickly (5–10 epochs usually enough)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
]

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    callbacks=callbacks
)

# Save
import os
os.makedirs("models", exist_ok=True)
model.save("models/catsdogs_model_fixed.h5")
print("✅ Fixed model saved as models/catsdogs_model_fixed.h5")
print("✅ This model will work correctly with the evaluation code")