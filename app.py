# app.py
import streamlit as st
from transformers import pipeline
from deepface import DeepFace
from PIL import Image
import numpy as np

st.set_page_config(page_title="Text + Image Sentiment", layout="centered")
st.title("🧠 Text + 🖼️ Image Sentiment Analysis")

# ---------------------------
# Text Sentiment (Hugging Face)
# ---------------------------
st.header("📝 Text Sentiment Analysis")
text_classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    device=-1
)

user_input = st.text_area("Enter text:")
if user_input:
    result = text_classifier(user_input)
    st.write(f"**Sentiment:** {result[0]['label']}")
    st.write(f"**Confidence:** {result[0]['score']:.2f}")

# ---------------------------
# Image Sentiment via FER2013 (DeepFace)
# ---------------------------
st.header("🖼️ Image Sentiment via Facial Expression")

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        try:
            result = DeepFace.analyze(img_path=np.array(image), actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            st.write(f"**Detected Emotion:** {emotion}")

            # Optional sentiment mapping
            positive_emotions = ['happy', 'surprise']
            negative_emotions = ['angry', 'disgust', 'fear', 'sad']
            sentiment = "Positive" if emotion.lower() in positive_emotions else (
                "Negative" if emotion.lower() in negative_emotions else "Neutral"
            )
            st.write(f"**Inferred Sentiment:** {sentiment}")
        except Exception as e:
            st.error(f"Error analyzing image: {e}")