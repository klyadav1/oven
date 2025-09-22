import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import requests
import joblib
import glob
import os
from datetime import datetime
import streamlit as st

# Configuration
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "oven_time_predictor.pkl"

# Column names from your CSVs
COLUMNS = [
    'Date', 'Time',
    'PT_ECO.TR01_WU311_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU312_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU314_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU321_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR02_WU322_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR02_WU323_B15.AA.R2251_ActValue[°C]'
]

# Sensor to target temperature mapping
SENSOR_TARGETS = {
    'WU311': 160,
    'WU312': 190,
    'WU314': 190,
    'WU321': 190,
    'WU322': 190,
    'WU323': 190
}

# 1. Weather API Integration
def get_weather():
    """Fetch current weather data from WeatherAPI.com"""
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
        response = requests.get(url)
        data = response.json()
        
        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "conditions": data["current"]["condition"]["text"]
        }
    except Exception as e:
        st.error(f"Weather API error: {str(e)}")
        return {
            "temp": 25.0,
            "humidity": 60,
            "conditions": "clear"
        }

# 2. Data Preparation
def prepare_training_data(uploaded_files):
    """Process multiple CSV files into training data"""
    dfs = []
    
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, delimiter='\t', encoding='utf-16')
            
            # Validate columns
            missing_cols = [col for col in COLUMNS if col not in df.columns]
            if missing_cols:
                st.warning(f"Warning: Missing columns in {file.name}: {missing_cols}")
                continue
                
            # Create datetime column with explicit format
            df['DateTime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d-%b-%y %I:%M:%S %p',
                errors='coerce'
            )
            
            # Drop rows with invalid datetime
            df = df.dropna(subset=['DateTime'])
            
            # Extract features from each sensor
            for sensor in SENSOR_TARGETS.keys():
                col = f"PT_ECO.TR01_{sensor}_B15.AA.R2251_ActValue[°C]" if sensor != 'WU323' else f"PT_ECO.TR02_{sensor}_B15.AA.R2251_ActValue[°C]"
                
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
                        continue
                        
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
    
    return pd.concat(dfs) if dfs else pd.DataFrame()

# 3. Feature Engineering
def create_features(df):
    """Combine oven data with weather features"""
    if df.empty:
        return pd.DataFrame()
        
    features = []
    
    for _, row in df.iterrows():
        # For training, we'll use placeholder weather
        weather = {
            'temp': 25.0,  # Placeholder - replace with historical data if available
            'humidity': 60,
            'conditions': 'clear'
        }
        
        features.append({
            'sensor': row['sensor'],
            'start_temp': row['start_temp'],
            'ambient_temp': weather['temp'],
            'humidity': weather['humidity'],
            'target_temp': SENSOR_TARGETS[row['sensor']],
            'time_to_target': row['time_to_target']
        })
    
    return pd.DataFrame(features)

# 4. Model Training
def train_model(features):
    """Train and save the prediction model"""
    if features.empty:
        raise ValueError("No training data available")
    
    features = pd.get_dummies(features, columns=['sensor'])
    
    # Store feature names for prediction
    feature_names = features.drop('time_to_target', axis=1).columns.tolist()
    
    X = features.drop('time_to_target', axis=1)
    y = features['time_to_target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    st.success(f"Model trained successfully! MAE: {mae:.2f} minutes")
    
    joblib.dump((model, feature_names), MODEL_PATH)
    return model, feature_names

# 5. Prediction Function
def predict_heating_time(sensor, current_temp):
    """Predict time needed to reach target temp using current weather"""
    try:
        model, features = joblib.load(MODEL_PATH)
        weather = get_weather()
        
        # Create input with all possible features
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
        
        # Activate the requested sensor
        input_data[f'sensor_{sensor}'] = 1
        
        # Ensure we only use features the model was trained with
        final_input = input_data[features]
        
        return model.predict(final_input)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main Streamlit App
def main():
    st.title("Industrial Oven Heating Time Predictor")
    st.write("This app predicts the time needed for an industrial oven to reach target temperature")
    
    # File upload section
    st.header("1. Train Model")
    uploaded_files = st.file_uploader("Upload CSV training files", type="csv", accept_multiple_files=True)
    
    model_trained = False
    if uploaded_files:
        with st.spinner("Processing training data..."):
            oven_data = prepare_training_data(uploaded_files)
            
        if not oven_data.empty:
            st.success(f"Processed {len(oven_data)} training samples")
            features = create_features(oven_data)
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    try:
                        model, feature_names = train_model(features)
                        model_trained = True
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
        else:
            st.error("No valid training data could be processed")
    
    # Prediction section
    st.header("2. Make Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        current_oven_temp = st.number_input("Current oven temperature (°C)", min_value=0.0, max_value=300.0, value=25.0)
    with col2:
        sensor_type = st.selectbox("Select sensor", options=list(SENSOR_TARGETS.keys()))
    
    if st.button("Predict Heating Time"):
        if not os.path.exists(MODEL_PATH):
            st.error("Please train the model first by uploading CSV files")
        else:
            with st.spinner("Making prediction..."):
                prediction = predict_heating_time(
                    sensor=sensor_type,
                    current_temp=current_oven_temp
                )
                
                if prediction is not None:
                    weather = get_weather()
                    st.success(f"**Predicted time to target: {prediction + 10:.1f} minutes**")
                    st.info(f"Current weather in {LOCATION}: {weather['temp']}°C, {weather['humidity']}% humidity")
                    
                    # Display additional info
                    target_temp = SENSOR_TARGETS[sensor_type]
                    st.write(f"Target temperature for {sensor_type}: {target_temp}°C")
                    st.write(f"Current temperature: {current_oven_temp}°C")
                    st.write(f"Temperature difference: {target_temp - current_oven_temp:.1f}°C")

if __name__ == "__main__":
    main()