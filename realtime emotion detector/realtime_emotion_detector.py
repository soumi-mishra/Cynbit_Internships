import cv2
from fer import FER

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load emotion detector
emotion_detector = FER(mtcnn=True)  # Better detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    result = emotion_detector.detect_emotions(frame)

    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Put text (emotion)
        confidence = emotions[top_emotion]
        cv2.putText(frame, f"{top_emotion} ({confidence:.2f})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()