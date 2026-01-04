
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_save_model():
    # 1. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    data_path = os.path.join("Data", "arrhythmia.csv")
    if not os.path.exists(data_path):
        # Fallback to searching in root or other common locations if moved
        possible_paths = ["arrhythmia1.csv", "arrhythmia.csv"]
        for p in possible_paths:
            if os.path.exists(p):
                data_path = p
                break
    
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, header=None)
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure 'Data/arrhythmia.csv' exists.")
        return

    # 2. Preprocessing Steps (Pre-Pipeline)
    print("Preprocessing data...")
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop column 13 (contains many missing values)
    # Column 13 is the 14th column (index 13)
    df.drop(columns=[13], inplace=True)
    
    # Split Features and Target
    # Original dataset has 280 columns. After dropping 1, we have 279.
    # The target was the last column (index 279 in original). 
    # Since we dropped one column from the middle (13), the indices > 13 shift left by 1.
    # So the target is still the last column.
    
    X = df.iloc[:, :-1] # All columns except the last
    y = df.iloc[:, -1]  # The last column
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 3. Create Pipeline
    # The notebook uses:
    # - SimpleImputer (strategy='mean')
    # - StandardScaler (implied by 'standardized data' in README and typical PCA usage)
    # - PCA (n_components=0.98)
    # - Kernelized SVM (kernel='sigmoid', C=10, gamma=0.001)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.98)),
        ('svc', SVC(kernel='sigmoid', C=10, gamma=0.001))
    ])

    # 4. Train
    print("Training model...")
    # To strictly verify performance, we can split. For the final app, training on all data is usually preferred,
    # but here let's stick to a reliable evaluation split first to confirm we match the notebook's ~80% accuracy.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 6. Save Model
    # We train on the full dataset for the production app model
    print("Retraining on full dataset for production...")
    pipeline.fit(X, y)
    
    model_dir = "Model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "arrhythmia_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    # 7. Generate Sample Data for App Testing
    print("Generating sample_data.csv...")
    # Get a random row (e.g., first row)
    sample_row = df.iloc[0, :-1] # Features only, keep col 13 for now as app expects it and drops it
    # Wait, df in memory already has col 13 dropped.
    # To properly simulate a user upload, we should grab a row from the ORIGINAL raw file 
    # OR reconstruct a row that matches what the app expects BEFORE internal preprocessing.
    # The app expects raw data (279 or 280 cols).
    # So let's re-read raw just for this sample.
    
    try:
        raw_df = pd.read_csv(data_path, header=None)
        # Take first row, all features (first 279 cols)
        sample = raw_df.iloc[0:1, :-1]
        sample.to_csv("sample_data.csv", header=False, index=False)
        print("sample_data.csv created successfully.")
    except Exception as e:
        print(f"Failed to create sample data: {e}")

if __name__ == "__main__":
    train_and_save_model()
