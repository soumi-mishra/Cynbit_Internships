# app.py

import streamlit as st
import joblib
import emoji
import os
# Load model
# model = joblib.load("emotion_classifier.pkl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_classifier.pkl")
model = joblib.load(MODEL_PATH)

# Emoji dictionary
emojis = {
    "anger": "😠", "joy": "😊", "sadness": "😢",
    "fear": "😨", "love": "❤️", "surprise": "😮"
}

st.title("Emotion Detection from Text")
user_input = st.text_input("Enter a sentence:")

if st.button("Detect Emotion"):
    if user_input:
        prediction = model.predict([user_input])[0]
        probs = model.predict_proba([user_input])[0]
        st.write(f"**Predicted Emotion:** {prediction} {emojis.get(prediction, '')}")
        
        # Optional: Probability chart
        import pandas as pd
        prob_df = pd.DataFrame(probs, index=model.classes_, columns=["Probability"])
        st.bar_chart(prob_df)