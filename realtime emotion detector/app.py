import streamlit as st
import cv2
from fer import FER
import numpy as np

st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("😊 Real-Time Emotion Detector")

# Initialize session state for webcam control
if 'run' not in st.session_state:
    st.session_state.run = False

# Start and Stop buttons
start = st.button("🎥 Start Webcam", key="start_button")
stop = st.button("❌ Stop Webcam", key="stop_button")

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# Display video feed if webcam is running
if st.session_state.run:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        result = detector.detect_emotions(frame)
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{top_emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Convert BGR to RGB and display
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Exit loop if user presses stop (next rerun)
        if not st.session_state.run:
            break

    cap.release()