import cv2
import numpy as np
import imutils
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
email_user = 'gokuben04@gmail.com'
email_password = 'vpou bsad rwqo ucbx'  # Update with your application-specific password
recipient_phone_number = '+919091086541'  # Replace with actual recipient phone number
carrier_email_domain = 'txt.att.net'  # Replace with carrier domain for email-to-SMS

def send_sms_alert():
    subject = ''
    body = 'A fall has been detected. Please check on the person immediately.'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = f'{recipient_phone_number}@{carrier_email_domain}'
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    text = msg.as_string()

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_user, email_password)
        server.sendmail(email_user, f'{recipient_phone_number}@{carrier_email_domain}', text)
        server.quit()
        print("Fall alert sent!")
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

# Load the pre-trained HOG + SVM model for person detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to detect falls
def detect_fall(box, previous_boxes):
    if len(previous_boxes) < 2:
        return False
    
    # Calculate the height change
    prev_box = previous_boxes[-1]
    prev_height = prev_box[3] - prev_box[1]
    current_height = box[3] - box[1]

    # Check if the height has reduced significantly
    if current_height < prev_height * 0.6:
        return True
    
    return False

# Initialize video capture (0 for webcam or 'path_to_video' for video file)
cap = cv2.VideoCapture(0)

previous_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect people in the frame
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))
    
    # Draw bounding boxes and detect falls
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if detect_fall((x, y, x + w, y + h), previous_boxes):
            cv2.putText(frame, "Fall Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            send_sms_alert()
        
        previous_boxes.append((x, y, x + w, y + h))
        
        # Keep the list of previous boxes short
        if len(previous_boxes) > 5:
            previous_boxes.pop(0)
    
    # Display the frame
    cv2.imshow('Elder Fall Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
