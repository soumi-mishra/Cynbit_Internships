import streamlit as st
import cv2
from fer import FER
import numpy as np
from realtime_emotion_detector import EmotionModel  # your custom model (if needed)

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("😊 Real-Time Emotion Detector")

# -----------------------------
# Session state for webcam
# -----------------------------
if 'run' not in st.session_state:
    st.session_state.run = False

# Start / Stop buttons
start = st.button("🎥 Start Webcam")
stop = st.button("❌ Stop Webcam")

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# -----------------------------
# Initialize webcam and detector
# -----------------------------
if st.session_state.run:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)

    # Optional: load your custom emotion model
    # model = EmotionModel()  

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        # Detect emotions using FER
        results = detector.detect_emotions(frame)
        for face in results:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]

            # Draw rectangle and put text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{top_emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert BGR to RGB for Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()