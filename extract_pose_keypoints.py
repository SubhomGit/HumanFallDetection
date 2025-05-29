
import os
import cv2
import pandas as pd
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Helper function to extract pose keypoints from an image
def extract_pose_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return None

    keypoints = []
    for landmark in result.pose_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.visibility])
    return keypoints

# Set your extracted image dataset folder path here
base_dir = "images"  # Change this if needed

# Define labels
labels = {"fall_images": 1, "nonfall_images": 0, "non_fall_images": 0}

# Data collector
data = []

# Loop through train and val
for split in ["train", "val"]:
    for category in ["fall_images", "nonfall_images", "non_fall_images"]:
        category_path = os.path.join(base_dir, split, category)
        if not os.path.exists(category_path):
            continue
        label = labels[category]
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            keypoints = extract_pose_keypoints(file_path)
            if keypoints:
                data.append([split, file_path, label] + keypoints)

# Create DataFrame
columns = ["split", "file_path", "label"] + [f"{k}_{i}" for i in range(33) for k in ("x", "y", "v")]
pose_df = pd.DataFrame(data, columns=columns)

# Save to CSV
pose_df.to_csv("pose_keypoints.csv", index=False)
print("Pose keypoints saved to pose_keypoints.csv")
