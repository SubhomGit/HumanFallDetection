Introduction:
Falls among elderly individuals can lead to severe injuries or fatalities if not promptly attended to. With an increasing number of elderly people living alone, there is a critical need for automated systems that can detect falls and alert caregivers immediately. This project leverages computer vision and modern communication technologies to enhance the safety and well-being of elderly individuals.
Technical Approach:
Video Capture and Preprocessing:
Use a webcam to capture live video streams.
Preprocess video frames by resizing and converting them to grayscale for faster processing.

Person Detection:
Implement the Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM) method to detect persons in video frames.

Fall Detection Algorithm:
Analyze bounding boxes of detected persons to determine if a fall has occurred based on a significant reduction in height.

SMS Alert System:
Configure an email-to-SMS gateway to send alert messages to caregivers when a fall is detected.
Methodology:
Setup and Configuration: Install necessary Python libraries (opencv-python, numpy, imutils, smtplib). Configure email account with an application-specific password for sending alerts.
Person Detection: Use pre-trained HOG + SVM model to detect persons.
Fall Detection: Track bounding boxes of detected persons across frames and identify falls based on height changes.
SMS Alerts: Use Python's smtplib to send emails via SMTP, configured to send as SMS using the recipient's phone number and carrier domain.
