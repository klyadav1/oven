import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Configuration
CSV_PATH = "D:/Kanhaiya/Research Data CED OVEN/merged_oven_data.csv"
MODEL_PATH = "D:/Kanhaiya/oven_time_predictor.pkl"
REQUIREMENTS_PATH = "D:/Kanhaiya/requirements.txt"
README_PATH = "D:/Kanhaiya/README.md"
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

# Step 1: Read and prepare data
def prepare_training_data(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()  # Clean column names
        print("CSV columns:", df.columns.tolist())  # Debug print
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%b-%y %I:%M:%S %p', errors='coerce')
        df.dropna(subset=['DateTime'], inplace=True)
    except Exception as e:
        print("CSV read error:", e)
        return pd.DataFrame()

    dfs = []
    for sensor in SENSOR_TARGETS:
        try:
            reach_time = df[df[sensor] >= SENSOR_TARGETS[sensor]].iloc[0]
            dfs.append(pd.DataFrame({
                'sensor': [sensor],
                'start_temp': [df[sensor].iloc[0]],
                'max_temp': [df[sensor].max()],
                'time_to_target': [(reach_time['DateTime'] - df['DateTime'].iloc[0]).total_seconds() / 60],
                'date': [df['DateTime'].iloc[0]]
            }))
        except:
            continue

    return pd.concat(dfs) if dfs else pd.DataFrame()

# Step 2: Feature engineering
def create_features(df):
    features = []
    for _, row in df.iterrows():
        features.append({
            'sensor': row['sensor'],
            'start_temp': row['start_temp'],
            'ambient_temp': 25.0,
            'humidity': 60,
            'target_temp': SENSOR_TARGETS[row['sensor']],
            'time_to_target': row['time_to_target']
        })
    return pd.DataFrame(features)

# Step 3: Train and save model
def train_model(features):
    features = pd.get_dummies(features, columns=['sensor'])
    feature_names = features.drop('time_to_target', axis=1).columns.tolist()
    X = features.drop('time_to_target', axis=1)
    y = features['time_to_target']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    preds = model.predict(X)
    print("Model MAE:", round(mean_absolute_error(y, preds), 2), "minutes")
    joblib.dump((model, feature_names), MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

# Step 4: Generate requirements.txt and README.md
def generate_files():
    with open(REQUIREMENTS_PATH, "w") as f:
        f.write("pandas\nnumpy\nscikit-learn\njoblib\n")

    with open(README_PATH, "w") as f:
        f.write("""# Oven Heat-Up Time Predictor

This script trains a machine learning model to predict how long an oven sensor will take to reach its target temperature.

## Features

- Reads oven sensor data from CSV
- Parses date-time and sensor readings
- Trains a Random Forest regression model
- Saves model as oven_time_predictor.pkl

## Requirements

See `requirements.txt` for dependencies.

## How to Run

1. Ensure your CSV file is located at:
   D:/Kanhaiya/Research Data CED OVEN/merged_oven_data.csv

2. Run the script:
   python main.py

3. Output:
   - Trained model saved as oven_time_predictor.pkl
   - requirements.txt and README.md auto-generated
""")
    print("requirements.txt and README.md created.")

# Main execution
if __name__ == "__main__":
    df_raw = prepare_training_data(CSV_PATH)
    if df_raw.empty:
        raise ValueError("No valid training data found.")
    df_feat = create_features(df_raw)
    train_model(df_feat)
    generate_files()
