

### README.md

```markdown
# Multimodal Fall Detection System

This project implements a **multimodal fall detection system** that combines **sensor-based** and **video-based** approaches to detect falls. The system is designed to monitor individuals (e.g., elderly people) and detect falls in real-time, with potential applications in healthcare and safety monitoring.

- **Sensor-Based Fall Detection**: Utilizes accelerometer and gyroscope data (originally implemented in a React Native Android app) to detect falls using a pre-trained ONNX model.
- **Video-Based Fall Detection**: Processes video input from a webcam or pre-recorded video files using MediaPipe Pose for human pose estimation and a heuristic-based algorithm to detect falls.

The project can be extended to integrate both modalities into a single application, combining sensor data from a mobile device and video data from a webcam or CCTV camera.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Sensor-Based Fall Detection](#sensor-based-fall-detection)
  - [Video-Based Fall Detection](#video-based-fall-detection)
- [Fall Detection Logic](#fall-detection-logic)
  - [Sensor-Based Logic](#sensor-based-logic)
  - [Video-Based Logic](#video-based-logic)
- [File Structure](#file-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This multimodal fall detection system aims to provide a robust solution for detecting falls using two complementary approaches:
1. **Sensor-Based Detection**: Originally implemented in a React Native Android app, this component uses accelerometer and gyroscope data to detect falls. A pre-trained ONNX model (`k_fall_detection_model_optimized.onnx`) processes the sensor data to classify movements as falls or non-falls. The app also includes an OTP-based login system and SMS alert functionality (not covered in this README).
2. **Video-Based Detection**: Uses video input (webcam or pre-recorded videos) to detect falls by analyzing human poses with MediaPipe Pose. A heuristic-based algorithm identifies falls based on body landmark positions and movement velocity.

Both components can be used independently or integrated into a single system for enhanced reliability. For example, the video-based system can run on a laptop or Raspberry Pi with a camera, while the sensor-based system can run on a mobile device, with both communicating alerts to a central application.

## Features
- **Sensor-Based Fall Detection**:
  - Processes accelerometer and gyroscope data from an Android device.
  - Uses a pre-trained ONNX model for fall detection.
  - Originally integrated into a React Native app with OTP login and SMS alerts.
- **Video-Based Fall Detection**:
  - Supports both webcam (live) and pre-recorded video input.
  - Uses MediaPipe Pose for human pose estimation.
  - Implements a heuristic-based fall detection algorithm.
  - Displays pose landmarks on the video feed.
  - Prints "Fall Detected!" or "No Fall" in the terminal and overlays the status on the video feed.
- **Multimodal Potential**:
  - Combines sensor and video data for more accurate fall detection (future integration).

## Prerequisites
### General
- **Python 3.7+**: For video-based fall detection scripts.
- **Node.js and npm**: For the React Native app (sensor-based component).
- **Android Device/Emulator**: For running the React Native app.

### Sensor-Based Fall Detection
- **React Native Environment**: Set up for Android development (Expo recommended).
- **Android Device**: With accelerometer and gyroscope sensors.
- **Dependencies**:
  - React Native libraries: `@react-native-firebase/app`, `@react-native-firebase/auth`, `onnxruntime-react-native`, etc.
  - ONNX model: `k_fall_detection_model_optimized.onnx` (pre-trained model for fall detection).

### Video-Based Fall Detection
- **Webcam**: Required for live fall detection (`fall_detection_webcam.py`).
- **Video Files**: Pre-recorded videos for testing (`fall_detection_video.py`).
- **Dependencies**:
  - `opencv-python`: For video capture and frame processing.
  - `mediapipe`: For pose estimation.
  - `numpy`: For numerical computations.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multimodal-fall-detection.git
cd multimodal-fall-detection
```

### 2. Sensor-Based Fall Detection (React Native App)
#### Set Up the React Native Environment
1. **Install Node.js and npm**:
   - Download and install from [nodejs.org](https://nodejs.org/).
2. **Install Expo CLI**:
   ```bash
   npm install -g expo-cli
   ```
3. **Navigate to the App Directory**:
   ```bash
   cd FallDetectionApp
   ```
4. **Install Dependencies**:
   ```bash
   npm install
   ```
   - This installs required packages like `@react-native-firebase`, `onnxruntime-react-native`, and others listed in `package.json`.
5. **Add the ONNX Model**:
   - Place `k_fall_detection_model_optimized.onnx` in the `FallDetectionApp/models/` directory.
6. **Set Up Firebase** (for OTP login):
   - Follow the Firebase setup instructions in the [original project documentation](#future-improvements) to configure `google-services.json`.

#### Run the App
1. **Start the Expo Server**:
   ```bash
   npx expo start
   ```
2. **Run on Android**:
   - Open the Expo Go app on your Android device and scan the QR code, or use an Android emulator.
   - The app will collect sensor data and detect falls in the background.

### 3. Video-Based Fall Detection (Python Scripts)
#### Set Up the Python Environment
1. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv fall_detection_env
   source fall_detection_env/bin/activate  # On Windows: fall_detection_env\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. **Prepare Video Files** (for `fall_detection_video.py`):
   - Place your video files in a directory accessible by the script.
   - Update the `video_path` variable in `fall_detection_video.py` with the path to your video file.

## Usage
### Sensor-Based Fall Detection
1. **Launch the App**:
   - Run the React Native app as described above.
   - Log in using the OTP system (requires Firebase setup).
2. **Navigate to Fall Detection Screen**:
   - After logging in, the app starts collecting accelerometer and gyroscope data.
3. **Fall Detection**:
   - The app processes sensor data using the ONNX model.
   - If a fall is detected, it logs the event and can send an SMS alert (requires Twilio or Firebase setup).

### Video-Based Fall Detection
#### Using a Pre-Recorded Video (`fall_detection_video.py`)
1. **Update the Video Path**:
   - Open `fall_detection_video.py` and set the `video_path` variable:
     ```python
     video_path = "path/to/your/video.mp4"
     ```
2. **Run the Script**:
   ```bash
   python fall_detection_video.py
   ```
3. **Observe the Output**:
   - A window displays the video feed with pose landmarks.
   - The terminal prints "Fall Detected!" or "No Fall" for each frame.
   - The video feed shows "Fall Detected!" (red) or "No Fall" (green).
   - Press `q` to exit.

#### Using a Webcam (`fall_detection_webcam.py`)
1. **Ensure Webcam Access**:
   - Make sure your laptop’s webcam is working.
2. **Run the Script**:
   ```bash
   python fall_detection_webcam.py
   ```
3. **Observe the Output**:
   - A window displays the webcam feed with pose landmarks.
   - The terminal prints "Fall Detected!" or "No Fall".
   - The video feed shows "Fall Detected!" (red) or "No Fall" (green).
   - Press `q` to exit.

## Fall Detection Logic
### Sensor-Based Logic
- **Data Collection**:
  - Collects accelerometer and gyroscope data from an Android device at a frequency of 50 Hz.
- **Data Preprocessing**:
  - Computes features like acceleration magnitude and angular velocity.
  - Normalizes the data for input to the ONNX model.
- **ONNX Model Inference**:
  - Uses `k_fall_detection_model_optimized.onnx`, a pre-trained model that takes sensor data as input and outputs a binary classification (fall or no fall).
  - Input: 6 features (3 accelerometer axes, 3 gyroscope axes).
  - Output: 1 (fall) or 0 (no fall).
- **Alert**:
  - If a fall is detected, the app logs the event and can trigger an SMS alert (requires additional setup).

### Video-Based Logic
- **Pose Estimation**:
  - MediaPipe Pose detects human body landmarks (e.g., nose, hips) in each frame.
- **Landmark Analysis**:
  - **Head Position**: The y-coordinate of the nose (approximating the head). If `head.y > 0.7`, the head is near the bottom of the frame (close to the ground).
  - **Height Difference**: The vertical distance between the head (nose) and left hip. If `height_difference < 0.2`, the head is close to the body’s midsection (indicating a low posture).
  - **Velocity**: The velocity of the head’s vertical movement. If `velocity > 0.5`, the head is moving downward quickly (indicating a sudden fall).
- **Fall Detection**:
  - A fall is detected if all conditions are met: `height_difference < 0.2`, `head.y > 0.7`, and `velocity > 0.5`.
  - Outputs "Fall Detected!" (terminal and video in red) or "No Fall" (terminal and video in green).

## File Structure
```
multimodal-fall-detection/
│
├── FallDetectionApp/                   # React Native app for sensor-based fall detection
│   ├── App.js                          # Main app component (OTP login and fall detection)
│   ├── firebaseConfig.js               # Firebase configuration for OTP login
│   ├── models/                         # Directory for ONNX model
│   │   └── k_fall_detection_model_optimized.onnx  # Pre-trained ONNX model
│   └── package.json                    # React Native dependencies
│
├── fall_detection_video.py             # Python script for video-based fall detection (pre-recorded videos)
├── fall_detection_webcam.py            # Python script for video-based fall detection (webcam)
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies (optional, generate with `pip freeze > requirements.txt`)
```

## Future Improvements
- **Multimodal Integration**:
  - Combine sensor and video data in a single application for more accurate fall detection.
  - Example: Run the Python script as a backend server and communicate with the React Native app via WebSocket or API.
- **Custom Model Training for Video**:
  - Train a machine learning model using labeled video data to improve fall detection accuracy.
  - Export the model to ONNX for inference, similar to the sensor-based approach.
- **Enhanced Fall Detection Logic**:
  - Incorporate additional features like body orientation, fall duration, or velocity profiles for both sensor and video data.
- **Cross-Platform Deployment**:
  - Deploy the video-based system on a Raspberry Pi with a camera for standalone operation.
  - Enhance the React Native app to support iOS and integrate video processing.
- **Alert System**:
  - Add SMS or email alerts for the video-based system, similar to the sensor-based app.
  - Integrate with a cloud service for remote monitoring.

## Contributing
Contributions are welcome! If you’d like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows standard Python and JavaScript conventions and includes comments for clarity.

