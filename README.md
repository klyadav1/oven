# Oven Heat-Up Time Predictor

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
