import os
import tensorflow as tf
import shutil

# Download dataset (200MB, filtered cats vs dogs from TensorFlow)
url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url, extract=True)

# Unzipped folder path
base_dir = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

# Target structure
target_dir = "./dataset"
train_dir = os.path.join(target_dir, "train")
test_dir = os.path.join(target_dir, "test")

# Clean up any previous run
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(train_dir)
os.makedirs(test_dir)

# Move files into train/ and test/
for split, new_split in [("train", "train"), ("validation", "test")]:
    for cls in ["cats", "dogs"]:
        src = os.path.join(base_dir, split, cls)
        dst = os.path.join(target_dir, new_split, cls)
        os.makedirs(dst, exist_ok=True)
        for fname in os.listdir(src):
            fsrc = os.path.join(src, fname)
            fdst = os.path.join(dst, fname)
            shutil.copy(fsrc, fdst)

print("✅ Cats vs Dogs dataset prepared under ./dataset/train and ./dataset/test")
