# How to run

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

3. Run evaluation
 python run_evaluation.py --dataset ./dataset --model ./models/catsdogs_model.h5 --out ./reports --seed 1234

4. Open report
Open reports/report.html in your browser
