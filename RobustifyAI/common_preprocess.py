# common_preprocess.py
from keras.utils import load_img, img_to_array
import numpy as np

TARGET_SIZE = (150, 150)   # must match training

def preprocess_fp(fp, target_size=TARGET_SIZE, debug=False):
    """
    Load + preprocess one image.
    Matches training pipeline:
    - resize to 150x150
    - convert to float32
    - rescale to [0,1]
    """
    img = load_img(fp, target_size=target_size)     # resize
    arr = img_to_array(img).astype("float32") / 255.0  # normalize like training
    
    if debug:
        print(f"[DEBUG] {fp} shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}")
    
    return arr
