# Oven Heat-Up Time Predictor

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
