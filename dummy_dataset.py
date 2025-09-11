# make_dummy_dataset.py
import os
from PIL import Image
import numpy as np

os.makedirs("dataset/cats", exist_ok=True)
os.makedirs("dataset/dogs", exist_ok=True)

for i in range(2):
    img = Image.fromarray(np.random.randint(0, 255, (64,64,3), dtype=np.uint8))
    img.save(f"dataset/cats/cat{i}.jpg")

    img = Image.fromarray(np.random.randint(0, 255, (64,64,3), dtype=np.uint8))
    img.save(f"dataset/dogs/dog{i}.jpg")

print("Dummy dataset created at ./dataset")
