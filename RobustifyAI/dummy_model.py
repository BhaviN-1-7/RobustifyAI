# make_dummy_model.py
import tensorflow as tf
from keras import layers, models

# Simple CNN
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(2, activation="softmax")  # 2 classes: cats, dogs
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fake training data (random noise)
import numpy as np
X = np.random.rand(10, 64, 64, 3)
y = np.random.randint(0, 2, 10)

model.fit(X, y, epochs=1, verbose=0)

# Save model
model.save("model.h5")
print("Dummy model saved as model.h5")
