import streamlit as st
import pandas as pd
import joblib
import requests

# Configuration
MODEL_PATH = "oven_time_predictor.pkl"
CSV_PATH = "merged_oven_data.csv"
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Get weather
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
    except:
        return {"temp": 25.0, "humidity": 60, "conditions": "Unknown"}

# Predict
def predict(sensor, current_temp, model, features):
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
    prediction = model.predict(final_input)[0]
    return prediction, weather

# UI
st.set_page_config(page_title="Oven Heat-Up Predictor", layout="centered")
st.title("🔥 Oven Heat-Up Time Predictor")
st.markdown("Predict how long your oven sensor will take to reach its target temperature.")

sensor = st.selectbox("Select Sensor", list(SENSOR_TARGETS.keys()))
current_temp = st.number_input("Enter Current Oven Temperature (°C)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Predict Time to Target"):
    try:
        model, features = load_model()
        prediction, weather = predict(sensor, current_temp, model, features)
        st.success(f"Estimated time to reach target: **{round(prediction + 10, 1)} minutes**")
        st.info(f"Weather in {LOCATION}: {weather['temp']}°C, {weather['humidity']}% humidity, {weather['conditions']}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
