import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

# Load face detector and predictor
shape_Predictor = r"C:\Users\PRIYANKA\Downloads\shape_predictor_68_face_landmarks.dat"  # Ensure this file is in your project folder
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_Predictor)

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Streamlit UI
st.title("🚘 Drowsiness Detection App 😴")
st.write("Detect drowsiness in real-time using OpenCV & Dlib!")

# OpenCV Video Capture in the main thread
cap = cv2.VideoCapture(0)

earThresh = 0.3  # Threshold for drowsiness detection
earFrames = 48    # Number of consecutive frames to trigger alert
count = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

frame_placeholder = st.empty()  # Placeholder to update video frames

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Failed to capture video")
        break

    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < earThresh:
            count += 1
            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            count = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Streamlit
    frame_placeholder.image(frame, channels="RGB")

cap.release()
st.write("**Press Stop to exit the app**")
