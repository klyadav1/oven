import pandas as pd
import requests
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

WEATHER_API_KEY = "your_api_key_here"
LOCATION = "Pune"
MODEL_PATH = "model/oven_time_predictor.pkl"
COLUMNS = ['Date', 'Time', 'WU311', 'WU312', 'WU314', 'WU321', 'WU322', 'WU323']
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

def get_weather():
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
        response = requests.get(url)
        data = response.json()
        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "conditions": data["current"]["condition"]["text"]
        }
    except Exception:
        return {"temp": 25.0, "humidity": 60, "conditions": "Unknown"}

def prepare_training_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    df.columns = df.columns.str.strip()
    df['Time'] = df['Time'].astype(str).str.strip()
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df.dropna(subset=['DateTime'], inplace=True)

    dfs = []
    for sensor in SENSOR_TARGETS:
        if sensor in df.columns:
            target_temp = SENSOR_TARGETS[sensor]
            try:
                reach_time = df[df[sensor] >= target_temp].iloc[0]
                dfs.append(pd.DataFrame({
                    'sensor': [sensor],
                    'start_temp': [df[sensor].iloc[0]],
                    'max_temp': [df[sensor].max()],
                    'time_to_target': [(reach_time['DateTime'] - df['DateTime'].iloc[0]).total_seconds() / 60],
                    'date': [df['DateTime'].iloc[0]]
                }))
            except IndexError:
                continue
    return pd.concat(dfs) if dfs else pd.DataFrame()

def create_features(df):
    if df.empty:
        return pd.DataFrame()
    weather = get_weather()
    features = []
    for _, row in df.iterrows():
        features.append({
            'sensor': row['sensor'],
            'start_temp': row['start_temp'],
            'ambient_temp': weather['temp'],
            'humidity': weather['humidity'],
            'target_temp': SENSOR_TARGETS[row['sensor']],
            'time_to_target': row['time_to_target']
        })
    return pd.DataFrame(features)

def train_model(features):
    features = pd.get_dummies(features, columns=['sensor'])
    feature_names = features.drop('time_to_target', axis=1).columns.tolist()
    X = features[feature_names]
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
    return model

def predict_heating_time(sensor, current_temp):
    model, features = joblib.load(MODEL_PATH)
    weather = get_weather()
    target = SENSOR_TARGETS[sensor]
    if current_temp >= target:
        return 0.0
    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [target],
        **{f'sensor_{s}': [1 if s == sensor else 0] for s in SENSOR_TARGETS}
    })
    final_input = input_data[features]
    return model.predict(final_input)[0]

def predict_startup_times(current_temp):
    predictions = []
    for sensor in SENSOR_TARGETS:
        time = predict_heating_time(sensor, current_temp)
        eta = datetime.now() + pd.to_timedelta(time, unit='m')
        predictions.append([sensor, round(time, 2), eta.strftime("%Y-%m-%d %H:%M")])
    return predictions

def generate_project_files():
    with open("requirements.txt", "w") as f:
        f.write("streamlit\nscikit-learn\npandas\nnumpy\nrequests\njoblib\n")

    with open("README.md", "w") as f:
        f.write("""# Oven Heat-Up Time Predictor

Predict how long each oven sensor takes to reach target temperature using ML and weather data.

## Features
- CSV parsing and feature engineering
- Weather API integration
- Random Forest model training
- Streamlit dashboard for predictions

## Setup
1. Place your CSV in `data/merged_oven_data.csv`
2. Run `train.py` to train the model
3. Launch `app.py` to use the dashboard

## Author
Kanhaiya — Industrial ML workflow builder and automation expert
""")
