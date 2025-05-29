import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables for fall detection
last_head_y = None
last_timestamp = None
fall_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (MediaPipe expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    # Convert back to BGR for OpenCV
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Extract landmarks for fall detection
        head = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # Head (nose)
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]  # Left hip
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Timestamp in seconds

        if head and hip:
            height_difference = abs(head.y - hip.y)

            # Calculate velocity of head movement
            velocity = 0
            if last_head_y is not None and last_timestamp is not None:
                time_diff = current_timestamp - last_timestamp
                if time_diff > 0:
                    velocity = abs(head.y - last_head_y) / time_diff

            last_head_y = head.y
            last_timestamp = current_timestamp

            # Fall detection logic: Head close to ground and moving fast
            if height_difference < 0.2 and head.y > 0.7 and velocity > 0.5:
                fall_detected = True
                print("Fall Detected!")
            else:
                fall_detected = False

    # Display fall detection status
    if fall_detected:
        cv2.putText(frame, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Fall Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()