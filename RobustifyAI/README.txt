# RobustifyAI - Demo Dataset Information

## Model Training & Storage
- **Training Scripts**: Use the provided training scripts to create your models:
  - `train_catsdogs.py` - Train cats vs dogs classifier
  - `dummy_model.py` - Create simple test models
  - `cifar10.py` - Train CIFAR-10 classifier
- **Model Storage**: Save trained models as `.h5` files in the `./models/` folder
- **Demo Model**: Pre-trained cats vs dogs classifier (`models/catsdogs_model.h5`)

## Dataset Structure
- **Storage Location**: `./dataset/` folder with the following structure:
  ```
  dataset/
  ├── train/
  │   ├── cats/     # Training images of cats
  │   └── dogs/     # Training images of dogs
  └── test/
      ├── cats/     # Test images of cats  
      └── dogs/     # Test images of dogs
  ```

## Dataset Requirements
- **Format**: JPEG/PNG images
- **Classes**: Binary classification (cats vs dogs)
- **Size**: Minimum 100 images per class recommended
- **Split**: Pre-split into train/test folders, or use `--test_ratio` for auto-split

## How to Run

1. Create environment (Python 3.8 recommended)

# Using venv
python3.8 -m venv venv
source venv/Scripts/activate       #Linux/Mac
venv\Scripts\activate          # Windows

# Or using conda
conda create -n evalenv python=3.8 -y
conda activate evalenv

2. Install requirements
pip install -r requirements.txt

3. Prepare your dataset
- Place images in the dataset folder structure shown above
- Or use your own dataset with similar structure

4. Run evaluation
python run_evaluation.py --dataset ./dataset --model ./models/catsdogs_model.h5 --out ./reports --seed 1234

5. Open report
Open reports/report.html in your browser
