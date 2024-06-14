# Voice-to-Gender
This project focuses on developing an advanced voice recognition system that combines multiple machine learning models to achieve high accuracy and retraining on failed predictions. The voice input will be sourced from Mike and can also be uploaded for processing.

# Model
I used four different models: Support Vector Classifier (SVC), Random Forest, XGBoost, and a Stacking Model in which i used Logistic Regression that combines these classifiers for improved accuracy.

# Required libraries
streamlit

numpy

pandas

sounddevice

librosa

scikit-learn

xgboost

seaborn

matplotlib

scipy

# How to run this project
open app.py

open Terminal (note! terminal directory shoud be of app.py)

Type => streamlit run app.py
