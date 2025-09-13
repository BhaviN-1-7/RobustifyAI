from common_preprocess import preprocess_fp
from keras.models import load_model
import numpy as np

model = load_model("models/catsdogs_model.h5")

cat_arr = preprocess_fp("./dataset/test/cats/cat.2000.jpg", debug=True)
dog_arr = preprocess_fp("./dataset/test/dogs/dog.2000.jpg", debug=True)

print("Cat prob:", model.predict(cat_arr[np.newaxis]))
print("Dog prob:", model.predict(dog_arr[np.newaxis]))
