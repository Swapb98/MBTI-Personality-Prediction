MBTI Personality Type Prediction Using Deep Learning

This project predicts Myers-Briggs Type Indicator (MBTI) personality types based on user-generated text. It uses TF-IDF vectorization and LSTM-based deep learning models to classify users into one of the 16 MBTI types.

Project Contents:
- MBTI_Text_Classifier.ipynb : Jupyter notebook for training and analysis
- predict.py : Script to run real-time MBTI predictions
- requirements.txt : Python dependencies
- README.md : Project documentation

Getting Started:

1. Clone the Repository:
git clone https://github.com/yourusername/mbti-prediction.git
cd mbti-prediction

2. Install Dependencies:
Make sure you have Python 3.x installed, then run:
pip install -r requirements.txt

How It Works:

The MBTI personality is broken into four dimensions:
- E/I – Extroversion vs. Introversion
- N/S – Intuition vs. Sensing
- F/T – Feeling vs. Thinking
- J/P – Judging vs. Perceiving

The project builds four binary classifiers (one for each dimension) using TF-IDF features and LSTM networks trained on text data. Each classifier predicts one letter of the MBTI type, which are then combined.

Running the Code:

Option 1: Use the Notebook
Open the notebook and run all cells:
jupyter notebook MBTI_Text_Classifier.ipynb

Option 2: Run the Real-Time Predictor
If you've already trained the models and saved them under a "model/" folder, run:
python predict.py

You'll be prompted to input text. The script will return the predicted MBTI type.

Dependencies (from requirements.txt):

keras==2.12.0
scipy==1.10.1
scikit-learn==1.2.2
pandas==1.5.3
nltk==3.8.1
matplotlib==3.7.1
seaborn==0.12.2
wordcloud==1.8.2.2
session_info==1.0.0

Install them using:
pip install -r requirements.txt

Results:
- Models trained separately for each MBTI trait
- Performance evaluated using F1-score and accuracy
- Best results achieved using LSTM with dropout and early stopping

Notes:
- Ensure the "model/" directory contains the saved .h5 model files and tfidf.pkl vectorizer before running predict.py
- If not available, retrain the models using the notebook

Authors:
- Swapnil Bhatnagar
- Mithil Mangukia
- Salman Ahmad

Developed for the SEP740 Deep Learning course at McMaster University

License:
This project is for academic use only.
