"""import socket
import pandas as pd
import joblib
import datetime

# Load trained model
model = joblib.load("K_fall_detection_model_fall_with_threshold_and_weights.pkl")

# UDP Socket Setup
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for real-time sensor data on port {UDP_PORT}...")

# Open CSV for logging
output_file = "predictions.csv"
with open(output_file, "a") as f:
    f.write("Timestamp,Sensor1,Sensor2,Sensor3,Sensor4,Sensor5,Sensor6,Sensor7,Sensor8,Sensor9,Predicted\n")

while True:
    data, addr = sock.recvfrom(1024)  # Receive data
    try:
        raw_data = data.decode().strip()
        print(f"Received Raw Data: {raw_data}")
        
        values = raw_data.split(',')
        
        if len(values) == 9:  # Expecting exactly 9 sensor values
            sensor_values = list(map(float, values))  # Convert all 9 values to float
            
            # Convert to DataFrame
            feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
            features = pd.DataFrame([sensor_values], columns=feature_names)

            # Predict
            predicted_activity = model.predict(features)[0]
            print(f"Predicted: {predicted_activity}")

            # Log to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, "a") as f:
                f.write(f"{timestamp},{','.join(map(str, sensor_values))},{predicted_activity}\n")

        else:
            print(f"Invalid data: Expected 9 values, got {len(values)}")

    except Exception as e:
        print(f"Error: {e}")
"""

import socket
import pandas as pd
import joblib
import datetime
import numpy as np

# Load trained model
model = joblib.load("K_fall_detection_model_fall_with_threshold_and_weights.pkl")

# Correct 13 features used during training
expected_features = [
    'acc_mag_mean', 'acc_mag_std', 'gyro_mag_mean', 'gyro_mag_std',
    'acc_x_peak', 'acc_y_peak', 'acc_z_peak',
    'gyro_x_peak', 'gyro_y_peak', 'gyro_z_peak',
    'resultant_acc_peak', 'resultant_gyro_peak',
    'jerk_mag_mean'
]

# UDP Socket Setup
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for real-time sensor data on port {UDP_PORT}...")

# Logging setup
output_file = "op_predictions.csv"
with open(output_file, "a") as f:
    header = "Timestamp," + ",".join(expected_features) + ",Prediction\n"
    f.write(header)

while True:
    data, addr = sock.recvfrom(1024)
    try:
        raw_data = data.decode().strip()
        print(f"Received Raw Data: {raw_data}")
        values = list(map(float, raw_data.split(",")))

        if len(values) == 13:
            # Prepare DataFrame
            df = pd.DataFrame([values], columns=expected_features)

            # Predict with probability and threshold
            proba = model.predict_proba(df)[0][1]
            prediction = 1 if proba >= 0.3 else 0

            print(f"Predicted: {prediction} (Probability: {proba:.3f})")

            # Log result
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, "a") as f:
                f.write(f"{timestamp}," + ",".join(map(str, values)) + f",{prediction}\n")
        else:
            print(f"Invalid data: Expected 13 values, got {len(values)}")

    except Exception as e:
        print(f"Error: {e}")
