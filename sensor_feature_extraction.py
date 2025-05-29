import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from scipy.stats import kurtosis, skew
import socket
import logging
import time
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fall_detection_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# UDP server settings
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005      # Match your HyperIMU setup
WINDOW_SIZE = 100    # 2 seconds at 50 Hz
SAMPLE_RATE = 50     # Hz
FEATURE_NAMES = [
    'acc_max', 'gyro_max', 'acc_kurtosis', 'gyro_kurtosis',
    'lin_max', 'acc_skewness', 'gyro_skewness', 'post_gyro_max', 'post_lin_max'
]
THRESHOLD = 0.9      # Increased to reduce false positives
CONFIRMATION_COUNT = 1  # Require 1 fall fall predictions
SMOOTHING_WINDOW = 5  # Moving average window for smoothing

# Load scaler and ONNX model
try:
    scaler = joblib.load('scaler.pkl')
    ort_session = ort.InferenceSession('K_fall_detection_model_optimized.onnx')
    logger.info("Loaded scaler and ONNX model successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(10.0)  # Increased timeout to reduce warnings
try:
    sock.bind((UDP_IP, UDP_PORT))
    logger.info(f"UDP server listening on {UDP_IP}:{UDP_PORT}")
except Exception as e:
    logger.error(f"Failed to bind UDP socket: {e}")
    raise

# Buffers for sensor data and smoothing
sensor_buffer = deque(maxlen=WINDOW_SIZE)
smoothing_buffer = deque(maxlen=SMOOTHING_WINDOW)

def smooth_data(data):
    """Apply moving average smoothing to sensor data."""
    smoothing_buffer.append(data)
    if len(smoothing_buffer) < SMOOTHING_WINDOW:
        return data  # Return unsmoothed until buffer is full
    smoothed = np.mean(list(smoothing_buffer), axis=0)
    return smoothed

def parse_hyperimu_data(data):
    """Parse HyperIMU UDP data based on observed format."""
    try:
        # Log raw data for debugging
        raw_data = data.decode('utf-8').strip()
        logger.debug(f"Raw data received: {raw_data}")
        
        # Split CSV and convert to floats
        values = [float(x) for x in raw_data.split(',') if x]
        
        # Expect 9 fields: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, lin_acc_x, lin_acc_y
        # Optionally handle 13 or 14 fields if orientation is included
        if len(values) not in [9, 13, 14]:
            logger.warning(f"Incomplete data: {len(values)} fields, expected 9, 13, or 14")
            return None
            
        # Extract fields
        timestamp = values[0]
        acc = np.array(values[1:4])  # acc_x, acc_y, acc_z
        gyro = np.array(values[4:7]) # gyro_x, gyro_y, gyro_z
        
        # Linear acceleration
        if len(values) >= 9:
            # 9 fields: lin_acc_x, lin_acc_y, set lin_acc_z to 0
            lin_acc = np.array(values[7:9] + [0.0])  # lin_acc_z approximated as 0
        else:
            # Fallback: use accelerometer if linear acceleration is missing
            logger.warning("Linear acceleration incomplete, using accelerometer data as fallback")
            lin_acc = acc
        
        # Apply smoothing
        acc = smooth_data(acc)
        gyro = smooth_data(gyro)
        lin_acc = smooth_data(lin_acc)
        
        return {
            'timestamp': timestamp,
            'acc': acc,
            'gyro': gyro,
            'lin_acc': lin_acc
        }
    except Exception as e:
        logger.warning(f"Error parsing data '{raw_data}': {e}")
        return None

def compute_features(window):
    """Compute features from sensor data window."""
    if len(window) != WINDOW_SIZE:
        logger.warning(f"Window size {len(window)}, expected {WINDOW_SIZE}")
        return None

    # Extract accelerometer, gyroscope, linear acceleration
    acc_data = np.array([d['acc'] for d in window])
    gyro_data = np.array([d['gyro'] for d in window])
    lin_data = np.array([d['lin_acc'] for d in window])

    # Compute magnitudes
    acc_mag = np.sqrt(np.sum(acc_data**2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
    lin_mag = np.sqrt(np.sum(lin_data**2, axis=1))

    # Compute features
    features = [
        np.max(acc_mag),                    # acc_max
        np.max(gyro_mag),                   # gyro_max
        kurtosis(acc_mag, fisher=False),    # acc_kurtosis
        kurtosis(gyro_mag, fisher=False),   # gyro_kurtosis
        np.max(lin_mag),                    # lin_max
        skew(acc_mag),                      # acc_skewness
        skew(gyro_mag),                     # gyro_skewness
        np.max(gyro_mag),                   # post_gyro_max (same as gyro_max)
        np.max(lin_mag)                     # post_lin_max (same as lin_max)
    ]
    logger.debug(f"Computed features: {dict(zip(FEATURE_NAMES, features))}")
    return np.array(features)

def predict_fall(features):
    """Run inference using ONNX model."""
    try:
        # Convert features to DataFrame to match scaler's expectations
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        # Scale features
        scaled_features = scaler.transform(features_df).astype(np.float32)
        logger.debug(f"Scaled features: {scaled_features}")
        
        # Prepare input for ONNX model
        input_name = ort_session.get_inputs()[0].name
        inputs = {input_name: scaled_features}
        
        # Run inference
        outputs = ort_session.run(None, inputs)
        logger.debug(f"ONNX outputs: {outputs}")
        
        # Adjust indexing based on ONNX output structure
        if len(outputs) > 1:  # Check if probabilities are in outputs[1]
            fall_prob = outputs[1][0][1]  # Probability of fall (class 1)
        else:
            fall_prob = outputs[0][0][1]  # Fallback to outputs[0]
        
        prediction = 1 if fall_prob >= THRESHOLD else 0
        return prediction, fall_prob
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None, None

def main():
    logger.info("Starting real-time fall detection...")
    logger.info("To see raw data packets and features, set logging level to DEBUG: logging.basicConfig(level=logging.DEBUG, ...)")
    last_prediction_time = time.time()
    fall_count = 0  # Counter for consecutive fall predictions

    while True:
        try:
            # Receive UDP data
            data, addr = sock.recvfrom(1024)
            parsed_data = parse_hyperimu_data(data)
            if parsed_data is None:
                continue

            # Add to buffer
            sensor_buffer.append(parsed_data)

            # Process window every 0.5 seconds
            if len(sensor_buffer) == WINDOW_SIZE and (time.time() - last_prediction_time) >= 0.5:
                features = compute_features(sensor_buffer)
                if features is not None:
                    prediction, fall_prob = predict_fall(features)
                    if prediction is not None:
                        # Confirmation mechanism
                        if prediction == 1:
                            fall_count += 1
                            status = f"Possible Fall (Count: {fall_count}/{CONFIRMATION_COUNT})"
                            if fall_count >= CONFIRMATION_COUNT:
                                status = "Fall Confirmed!"
                                fall_count = 0  # Reset counter after confirmation
                        else:
                            fall_count = 0  # Reset counter on non-fall
                            status = "No Fall"
                        logger.info(f"Prediction: {status}, Fall Probability: {fall_prob:.4f}")
                last_prediction_time = time.time()

        except socket.timeout:
            logger.warning("No data received within timeout period.")
            continue
        except KeyboardInterrupt:
            logger.info("Stopping real-time fall detection.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(0.1)  # Prevent CPU overload

    sock.close()

if __name__ == "__main__":
    main()