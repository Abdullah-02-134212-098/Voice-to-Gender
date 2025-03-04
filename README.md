# 🎤 Voice-To-Gender Classification System  
An advanced voice recognition system that predicts gender from voice input using multiple machine learning models. It supports both real-time microphone input and uploaded audio files, with an integrated feedback loop for retraining on incorrect predictions.  

## 🚀 Features  
✅ Predicts gender from voice recordings using ML models  
✅ Supports real-time audio input via microphone  
✅ Allows audio file uploads for processing  
✅ Uses a Stacking Model for improved accuracy  
✅ Enables retraining on incorrect predictions  
✅ Interactive and user-friendly Streamlit interface  

## 🧠 Machine Learning Models  
This project leverages four powerful classifiers to achieve high accuracy:  
🔹 **Support Vector Classifier (SVC)**  
🔹 **Random Forest**  
🔹 **XGBoost**  
🔹 **Stacking Model** (with Logistic Regression as a meta-classifier)  

## 📦 Required Libraries  
Ensure you have the following dependencies installed before running the project:  
```bash
pip install streamlit numpy pandas sounddevice librosa scikit-learn xgboost seaborn matplotlib scipy
```

## 🚀 How to Run  
1️⃣ **Clone the repository**  
```bash
https://github.com/Abdullah-02-134212-098/Voice-To-Gender.git
```

2️⃣ **Run the application**
```bash
streamlit run app.py
```

3️⃣ **Interact with the interface and start classifying voice recordings! 🎙️**



