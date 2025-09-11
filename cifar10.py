# prepare_cifar10.py
import os
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.models import load_model

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Only keep cats (class=3) and dogs (class=5)
train_mask = np.isin(y_train, [3, 5]).flatten()
test_mask = np.isin(y_test, [3, 5]).flatten()

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Relabel to 0=cat, 1=dog
y_train = (y_train == 5).astype(int).flatten()
y_test = (y_test == 5).astype(int).flatten()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build tiny CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Save model
model.save("model.h5")

# Export test images in dataset/ directory
from PIL import Image
import os

class_names = ["cats", "dogs"]
for cls in class_names:
    os.makedirs(os.path.join("dataset", cls), exist_ok=True)

for i in range(50):  # export 50 test images
    arr = (x_test[i] * 255).astype(np.uint8)  # scale back to 0–255 and cast
    img = Image.fromarray(arr)
    label = "cats" if y_test[i] == 0 else "dogs"
    img.save(os.path.join("dataset", label, f"img_{i}.png"))

