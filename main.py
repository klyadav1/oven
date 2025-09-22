import streamlit as st
import requests
import pickle
import pandas as pd

# Load trained model
def load_model():
    with open('oven_startup_model.pkl', 'rb') as f:
        return pickle.load(f)

# Get current temperature from OpenWeatherMap
def get_current_temperature(city="Pune"):
    api_key = "949ab7227e4144d0d493edad198016dd"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['main']['temp']
    except:
        return None

# Predict startup time
def predict_startup_time(model, start_temp, heating_rate):
    X = pd.DataFrame([[start_temp, heating_rate]], columns=['start_temp', 'heating_rate'])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit UI
st.title("Industrial Oven Startup Time Predictor 🔥")

city = st.text_input("Enter City for Weather Temperature", value="Pune")
heating_rate = st.number_input("Enter Heating Rate (°C/min)", min_value=0.1, max_value=10.0, value=2.0)

if st.button("Predict Startup Time"):
    temp = get_current_temperature(city)
    if temp is None:
        st.error("Failed to fetch temperature from API.")
    else:
        model = load_model()
        time = predict_startup_time(model, temp, heating_rate)
        st.success(f"Predicted Startup Time: {time:.2f} minutes")
        st.write(f"Current Temperature in {city}: {temp}°C")
