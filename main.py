import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import requests
import joblib
from datetime import datetime

# Configuration
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "oven_time_predictor.pkl"
CSV_PATH = "data/merged_oven_data.csv"

# Required columns
COLUMNS = ['Date', 'Time', 'WU311', 'WU312', 'WU314', 'WU321', 'WU322', 'WU323']
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

# 1. Weather API
def get_weather():
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
    response = requests.get(url)
    data = response.json()
    return {
        "temp": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"],
        "conditions": data["current"]["condition"]["text"]
    }

# 2. Prepare training data
def prepare_training_data(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except Exception as e:
        print("Error reading CSV:", e)
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    missing_cols = [col for col in COLUMNS if col not in df.columns]
    if missing_cols:
        print("Missing columns in", csv_file, ":", missing_cols)
        return pd.DataFrame()

    df['Time'] = df['Time'].astype(str).str.strip()

    # ✅ Specify format to avoid warning
    try:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%b-%y %I:%M:%S %p', errors='coerce')
    except Exception:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    df.dropna(subset=['DateTime'], inplace=True)

    print("Sensor max temperatures:")
    for sensor in SENSOR_TARGETS:
        if sensor in df.columns:
            print(f"{sensor}: {df[sensor].max()} deg C")

    dfs = []
    for sensor in SENSOR_TARGETS.keys():
        col = sensor
        if col in df.columns:
            target_temp = SENSOR_TARGETS[sensor]
            try:
                reach_time = df[df[col] >= target_temp].iloc[0]
                dfs.append(pd.DataFrame({
                    'sensor': [sensor],
                    'start_temp': [df[col].iloc[0]],
                    'max_temp': [df[col].max()],
                    'time_to_target': [(reach_time['DateTime'] - df['DateTime'].iloc[0]).total_seconds() / 60],
                    'date': [df['DateTime'].iloc[0]]
                }))
            except IndexError:
                print(f"{sensor}: target {target_temp} deg C not reached")
                continue
    return pd.concat(dfs) if dfs else pd.DataFrame()

# 3. Feature engineering
def create_features(df):
    if df.empty:
        return pd.DataFrame()
    features = []
    for _, row in df.iterrows():
        weather = {'temp': 25.0, 'humidity': 60}
        features.append({
            'sensor': row['sensor'],
            'start_temp': row['start_temp'],
            'ambient_temp': weather['temp'],
            'humidity': weather['humidity'],
            'target_temp': SENSOR_TARGETS[row['sensor']],
            'time_to_target': row['time_to_target']
        })
    return pd.DataFrame(features)

# 4. Train model
def train_model(features):
    if features.empty:
        raise ValueError("No training data available")
    features = pd.get_dummies(features, columns=['sensor'])
    global feature_names
    feature_names = features.drop('time_to_target', axis=1).columns.tolist()
    X = features.drop('time_to_target', axis=1)
    y = features['time_to_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Model MAE:", round(mean_absolute_error(y_test, preds), 2), "minutes")
    joblib.dump((model, feature_names), MODEL_PATH)
    return model, feature_names

# 5. Predict
def predict_heating_time(sensor, current_temp):
    model, features = joblib.load(MODEL_PATH)
    weather = get_weather()
    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [SENSOR_TARGETS[sensor]],
        'sensor_WU311': [0],
        'sensor_WU312': [0],
        'sensor_WU314': [0],
        'sensor_WU321': [0],
        'sensor_WU322': [0],
        'sensor_WU323': [0]
    })
    input_data[f'sensor_{sensor}'] = 1
    final_input = input_data[features]
    return model.predict(final_input)[0]

# 6. Auto-generate requirements.txt and README.md
def generate_project_files():
    with open("requirements.txt", "w") as f:
        f.write("pandas\nnumpy\nscikit-learn\nrequests\njoblib\n")

    with open("README.md", "w") as f:
        f.write("""# Oven Heat-Up Time Predictor

This project trains a machine learning model to predict how long an industrial oven sensor will take to reach its target temperature, based on starting temperature, ambient weather, and sensor type.

## Features

- Reads merged oven sensor data from CSV
- Parses date-time and sensor readings
- Uses live weather data (temperature, humidity) via WeatherAPI
- Trains a Random Forest regression model
- Predicts time-to-target for selected sensor and current oven temperature

## Requirements

See `requirements.txt` for dependencies.

## How to Run

1. Place your merged CSV file at:
   D:/kanhaiya/Research Data CED OVEN/merged_oven_data.csv

2. Run the script:
   python main.py

3. Enter current oven temperature and sensor type when prompted.

## Output

- Trained model saved as `oven_time_predictor.pkl`
- Console output shows predicted time and current weather

## Author

Kanhaiya — Industrial ML workflow builder and automation expert
""")

# Main
if __name__ == "__main__":
    print("Reading CSV file:", CSV_PATH)
    oven_data = prepare_training_data(CSV_PATH)
    features = create_features(oven_data)
    if not features.empty:
        model, feature_names = train_model(features)
        generate_project_files()
    else:
        raise ValueError("No valid training data could be processed")

    try:
        current_oven_temp = float(input("Enter current oven temperature in Celsius: "))
        sensor_type = input("Enter sensor (WU311/WU312/WU314/WU321/WU322/WU323): ").strip().upper()
        if sensor_type not in SENSOR_TARGETS:
            raise ValueError("Invalid sensor type")
        prediction = predict_heating_time(sensor=sensor_type, current_temp=current_oven_temp)
        weather = get_weather()
        print("Predicted time to target:", round(prediction + 10, 1), "minutes")
        print("Current weather in", LOCATION, ":", weather['temp'], "deg C,", weather['humidity'], "% humidity")
    except Exception as e:
        print("Prediction error:", str(e))
