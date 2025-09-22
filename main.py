import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Step 1: Get current temperature from OpenWeatherMap API
def get_current_temperature(city="Pune"):
    api_key = "949ab7227e4144d0d493edad198016dd"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temp = data['main']['temp']
        print(f"Current temperature in {city}: {temp}°C")
        return temp
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Step 2: Train ML model using historical oven data
def train_model():
    data = {
        'start_temp': [33.45, 34.10, 37.48, 34.92, 33.72, 32.92],
        'heating_rate': [4.60, 3.71, 1.86, 0.89, 0.96, 1.82],
        'time_to_target': [27.5, 42.0, 82.0, 174.5, 163.5, 86.5]
    }
    df = pd.DataFrame(data)
    X = df[['start_temp', 'heating_rate']]
    y = df['time_to_target']

    model = LinearRegression()
    model.fit(X, y)

    with open('oven_startup_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as oven_startup_model.pkl")

# Step 3: Predict startup time using live temperature
def predict_startup_time(start_temp, heating_rate):
    try:
        with open('oven_startup_model.pkl', 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict([[start_temp, heating_rate]])
        print(f"Predicted startup time: {prediction[0]:.2f} minutes")
        return prediction[0]
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return None

# Step 4: Create requirements.txt
def create_requirements():
    with open('requirements.txt', 'w') as f:
        f.write("requests\npandas\nnumpy\nscikit-learn\n")
    print("requirements.txt created")

# Main Execution
if __name__ == "__main__":
    train_model()
    weather_temp = get_current_temperature(city="Pune")
    if weather_temp is not None:
        heating_rate = 2.0  # Example value, can be fetched from PLC
        predict_startup_time(start_temp=weather_temp, heating_rate=heating_rate)
    create_requirements()
