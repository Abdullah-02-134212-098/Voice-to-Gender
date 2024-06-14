import streamlit as st
import numpy as np
import pandas as pd
import sounddevice as sd
import librosa
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Voice Recognition System", page_icon=":microphone:", layout="wide")

# Initialize session state variables
if 'correct_label' not in st.session_state:
    st.session_state.correct_label = "Choose Gender"
if 'features' not in st.session_state:
    st.session_state.features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'last_app_mode' not in st.session_state:
    st.session_state.last_app_mode = None

# Function to reset session state variables
def reset_session_state():
    st.session_state.correct_label = "Choose Gender"
    st.session_state.features = None
    st.session_state.prediction = None

# Sidebar for navigation
st.sidebar.title("Navigation")
current_app_mode = st.sidebar.selectbox("Choose the app mode", ["Train Classifiers", "Record & Analyze Voice", "Load Audio File", "Visualize Data"], key="app_mode_selection")

# Check if the app mode has changed and reset session state if necessary
if st.session_state.last_app_mode != current_app_mode:
    reset_session_state()
    st.session_state.last_app_mode = current_app_mode

# Global variables
classifiers = {}
le = None

def extract_features(audio, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).flatten()
    meanfreq = np.mean(spectral_centroid)
    sd = np.std(spectral_centroid)
    median = np.median(spectral_centroid)
    mode = float(stats.mode(spectral_centroid)[0])
    Q25 = np.percentile(spectral_centroid, 25)
    Q75 = np.percentile(spectral_centroid, 75)
    IQR = Q75 - Q25
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    return {
        "meanfreq": round(meanfreq / 1000, 2),
        "sd": round(sd / 1000, 2),
        "median": round(median / 1000, 2),
        "mode": round(mode / 1000, 2),
        "Q25": round(Q25 / 1000, 2),
        "Q75": round(Q75 / 1000, 2),
        "IQR": round(IQR / 1000, 2),
        "skew": round(spectral_contrast / 1000, 2),
        "kurt": round(spectral_rolloff / 1000, 2),
    }

def train_classifiers(X, y):
    classifiers = {
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier()
    }

    for name, clf in classifiers.items():
        clf.fit(X, y)
        with open(f'{name}_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        st.success(f'{name} trained successfully')

    return classifiers

def train_stacking_model(X, y):
    base_models = []
    for model_name in ['SVM', 'Random Forest', 'XGBoost']:
        with open(f'{model_name}_model.pkl', 'rb') as f:
            base_models.append(pickle.load(f))

    base_predictions = np.zeros((X.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        base_predictions[:, i] = model.predict_proba(X)[:, 1]

    stacking_model = LogisticRegression(max_iter=1000)
    stacking_model.fit(base_predictions, y)

    with open('stacking_model.pkl', 'wb') as f:
        pickle.dump(stacking_model, f)
    st.success('Stacking model trained successfully')

def load_classifiers():
    global classifiers, le
    classifiers = {}
    for model_name in ['SVM', 'Random Forest', 'XGBoost']:
        if os.path.exists(f'{model_name}_model.pkl'):
            with open(f'{model_name}_model.pkl', 'rb') as f:
                classifiers[model_name] = pickle.load(f)
    if os.path.exists('stacking_model.pkl'):
        with open('stacking_model.pkl', 'rb') as f:
            classifiers['Stacking'] = pickle.load(f)
    if os.path.exists('label_encoder.pkl'):
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

def record_audio(duration=5, sr=22050):
    st.info("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    st.info("Recording stopped.")
    return audio.flatten(), sr

def load_and_scale_audio(file):
    try:
        audio, sr = librosa.load(file, sr=22050)
        scaler = StandardScaler()
        scaled_audio = scaler.fit_transform(audio.reshape(-1, 1)).flatten()
        return scaled_audio, sr
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None

def compare_classifiers(classifiers, X_test, y_test):
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        st.write(f'\n{name} Classifier:')
        st.write(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        st.write(classification_report(y_test, y_pred))

def handle_incorrect_prediction(features, correct_label):
    if correct_label == "Choose Gender":
        st.error("Please select a valid gender before submitting.")
        return

    incorrect_data = {**features, "Label": correct_label}
    df = pd.DataFrame([incorrect_data])

    # Debug statements
    st.write("Debug: Data to be added")
    st.write(df)

    # Define the file paths
    incorrect_predictions_file = os.path.abspath('incorrect_predictions.csv')
    gendervoice_file = os.path.abspath('gendervoice.csv')

    try:
        # Ensure the CSV file exists and has the correct columns
        if os.path.exists(incorrect_predictions_file):
            incorrect_df = pd.read_csv(incorrect_predictions_file)
            #drop duplicates
            incorrect_df = incorrect_df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label'])
            if 'id' not in incorrect_df.columns:
                incorrect_df.insert(0, 'id', range(len(incorrect_df)))
            max_id = incorrect_df['id'].max() + 1
            df.insert(0, 'id', max_id)
            incorrect_df = pd.concat([incorrect_df, df], ignore_index=True)
        else:
            df.insert(0, 'id', 0)
            incorrect_df = df

        incorrect_df.to_csv(incorrect_predictions_file, index=False)

        # Ensure 'gendervoice.csv' is also updated
        if os.path.exists(gendervoice_file):
            gendervoice_df = pd.read_csv(gendervoice_file)
            #drop duplicates records
            gendervoice_df = gendervoice_df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label'])
            if 'id' not in gendervoice_df.columns:
                gendervoice_df.insert(0, 'id', range(len(gendervoice_df)))
            max_id = gendervoice_df['id'].max() + 1
            df['id'] = range(max_id, max_id + len(df))
            updated_df = pd.concat([gendervoice_df, df], ignore_index=True)
        else:
            df.insert(0, 'id', 0)
            updated_df = df

        updated_df.to_csv(gendervoice_file, index=False)

        st.success("Incorrect prediction saved and concatenated to incorrect_predictions.csv for future training. To get the best results, restart the program.")
        st.write("Debug: Updated DataFrame")
        #gendervoice_file = gendervoice_file.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label'])
        st.write(pd.read_csv(gendervoice_file).tail())
    except PermissionError as e:
        st.error(f"Permission error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def visualize_data(df, classifiers, le):
    st.sidebar.header("Visualization Options")
    choice = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Select...", "Bar Plot of Feature Means", "Histograms of All Features", "Confusion Matrix of All Models", 
         "Heatmap of Feature Correlations", "Scatter Plot", "Displot", "Pairplot", "Pie Chart of Label Distribution", 
         "Box Plot of Features", "Bar Chart of Label Distribution", "Contour Plot"], key="visualization_type"
    )

    if choice == "Bar Plot of Feature Means":
        plt.figure(figsize=(12, 6))
        mean_values = df[['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']].mean()
        mean_values.plot(kind='bar', color=sns.color_palette("husl", len(mean_values)))
        plt.title('Mean Values of Features')
        plt.xlabel('Features')
        plt.ylabel('Mean Values')
        st.pyplot(plt)

    elif choice == "Histograms of All Features":
        features = ['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']
        df[features].hist(bins=20, figsize=(20, 15), layout=(3, 3), color='skyblue', edgecolor='black')
        plt.suptitle('Histograms of All Features')
        plt.tight_layout()
        fig = plt.gcf()  # Get the current figure
        fig.subplots_adjust(top=0.95)  # Adjust the top margin
        st.pyplot(fig)
         
    elif choice == "Confusion Matrix of All Models":
        X = df[['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']]
        y = df['Label']
        
        # Fit the label encoder on the entire dataset
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        for model_name, clf in classifiers.items():
            if model_name == 'Stacking':
                # Generate base model predictions
                base_predictions = np.zeros((X_test.shape[0], len(classifiers) - 1))
                for i, base_model_name in enumerate(['SVM', 'Random Forest', 'XGBoost']):
                    base_predictions[:, i] = classifiers[base_model_name].predict_proba(X_test)[:, 1]
                y_pred = clf.predict(base_predictions)
            else:
                y_pred = clf.predict(X_test)
                
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {model_name}')
            st.pyplot(plt)

    elif choice == "Heatmap of Feature Correlations":
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='viridis')
        plt.title('Heatmap of Feature Correlations')
        st.pyplot(plt)

    elif choice == "Scatter Plot":
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("coolwarm", 2)
        scatter_plot = sns.scatterplot(x='meanfreq', y='sd', hue='Label', data=df, palette=colors)
        plt.title('Scatter Plot of Mean Frequency vs Standard Deviation')
        plt.xlabel('Mean Frequency')
        plt.ylabel('Standard Deviation')

        # Get the legend handles and labels
        handles, _ = scatter_plot.get_legend_handles_labels()
        labels = list(le.inverse_transform([0, 1]))  # Convert to list to avoid ValueError

        # Update legend with correct labels
        scatter_plot.legend(handles=handles, labels=labels, title="Gender", loc="upper right")

        st.pyplot(plt)

    elif choice == "Displot":
        plt.figure(figsize=(10, 6))
        sns.displot(df, x='meanfreq', hue='Label', kind='kde', fill=True)
        plt.title('Displot of Mean Frequency')
        st.pyplot(plt)

    elif choice == "Pairplot":
        plt.figure(figsize=(10, 6))
        sns.pairplot(df, hue='Label')
        plt.title('Pairplot of Features')
        st.pyplot(plt)

    elif choice == "Pie Chart of Label Distribution":
        label_counts = df['Label'].value_counts()
        labels = le.inverse_transform(label_counts.index)  # Get the class names
        colors = sns.color_palette("coolwarm", len(label_counts))
        plt.figure(figsize=(8, 8))
        plt.pie(label_counts, labels=labels, autopct='%1.1f%%', colors=colors, shadow=True, startangle=140)
        plt.title('Pie Chart of Label Distribution')
        plt.legend(labels=labels, loc="best", title="Gender")
        st.pyplot(plt)

    elif choice == "Box Plot of Features":
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']])
        plt.title('Box Plot of Features')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif choice == "Bar Chart of Label Distribution":
        label_counts = df['Label'].value_counts()
        labels = le.inverse_transform(label_counts.index)  # Get the class names
        colors = sns.color_palette("coolwarm", len(label_counts))
        plt.figure(figsize=(10, 6))
        colors.reverse()
        ax = sns.barplot(x=label_counts.index, y=label_counts.values, palette=colors)
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        plt.title('Bar Chart of Label Distribution', fontsize=16)
        plt.xlabel('Label', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(ticks=label_counts.index, labels=labels, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Create custom legend
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        plt.legend(handles=legend_patches, title="Gender", loc='best', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=14)
        st.pyplot(plt)

    elif choice == "Contour Plot":
        plt.figure(figsize=(10, 6))
        sns.kdeplot(x='meanfreq', y='sd', data=df, fill=True, cmap='viridis')
        plt.title('Contour Plot of Mean Frequency and Standard Deviation')
        plt.xlabel('Mean Frequency')
        plt.ylabel('Standard Deviation')
        st.pyplot(plt)

# Main panel based on sidebar selection
if current_app_mode == "Train Classifiers":
    st.title("Train Classifiers")
    df = pd.read_csv('gendervoice.csv')
    df = df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label']) 
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    
    X = df[['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt']]
    y = df['Label']
    classifiers = train_classifiers(X, y)
    train_stacking_model(X, y)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

elif current_app_mode == "Record & Analyze Voice":
    st.title("Record & Analyze Voice")
    duration = st.number_input("Recording Duration (seconds)", min_value=1, max_value=10, value=5, key="record_duration")
    load_classifiers()
    if st.button("Record"):
        if classifiers:  # Check if classifiers are loaded
            audio, sr = record_audio(duration)
            features = extract_features(audio, sr)
            st.info("Extracted Features:")
            st.write(features)
            stacking_model = classifiers.get('Stacking')
            base_models = [classifiers.get(name) for name in ['SVM', 'Random Forest', 'XGBoost']]
            base_predictions = np.zeros((1, len(base_models)))
            for i, model in enumerate(base_models):
                base_predictions[:, i] = model.predict_proba([list(features.values())])[:, 1]
            prediction = stacking_model.predict(base_predictions)
            predicted_label = le.inverse_transform(prediction)[0]
            st.success(f'Predicted Label: {predicted_label}')
            st.session_state.features = features
            st.session_state.prediction = predicted_label
        else:
            st.error("Classifiers are not trained. Please train classifiers first.")
            
    if st.session_state.features is not None and st.session_state.prediction is not None:
        #for checking if session is mantained
        #st.info(f"Predicted Label: {st.session_state.prediction}")
        correct_label = st.selectbox("Enter the correct label if the prediction is incorrect:", ["Choose Gender", "Male", "Female"], index=0, key="correct_label_record")
        st.session_state.correct_label = correct_label
        if st.button("Submit Correct Label"):
            handle_incorrect_prediction(st.session_state.features, st.session_state.correct_label)

elif current_app_mode == "Load Audio File":
    st.title("Load Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"], key="upload_audio")
    load_classifiers()
    if uploaded_file is not None:
        audio, sr = load_and_scale_audio(uploaded_file)
        if audio is not None:
            features = extract_features(audio, sr)
            st.info("Extracted Features:")
            st.write(features)
            stacking_model = classifiers.get('Stacking')
            base_models = [classifiers.get(name) for name in ['SVM', 'Random Forest', 'XGBoost']]
            base_predictions = np.zeros((1, len(base_models)))
            for i, model in enumerate(base_models):
                base_predictions[:, i] = model.predict_proba([list(features.values())])[:, 1]
            prediction = stacking_model.predict(base_predictions)
            predicted_label = le.inverse_transform(prediction)[0]
            st.success(f'Predicted Label: {predicted_label}')
            st.session_state.features = features
            st.session_state.prediction = predicted_label
            
            correct_label = st.selectbox("Enter the correct label if the prediction is incorrect:", ["Choose Gender", "Male", "Female"], index=0, key="correct_label_load")
            st.session_state.correct_label = correct_label
            if st.button("Submit Correct Label"):
                handle_incorrect_prediction(st.session_state.features, st.session_state.correct_label)

elif current_app_mode == "Visualize Data":
    st.title("Visualize Data")
    df = pd.read_csv('gendervoice.csv')
    #drop duplicates
    df = df.drop_duplicates(subset=['meanfreq', 'sd', 'median', 'mode', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'Label'])
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    load_classifiers()
    visualize_data(df, classifiers, le)