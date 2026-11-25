import streamlit as st
import joblib
import re
from pathlib import Path

st.set_page_config(page_title="Customer Review Sentiment Analyzer", page_icon="💬")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("💬 Customer Review Sentiment Analyzer")
st.write("Enter a customer review to predict its sentiment.")

review = st.text_area(
    "Customer Review",
    placeholder="Example: The phone has excellent battery life and the camera quality is amazing."
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        try:
            model, vectorizer = load_artifacts()
            cleaned = clean_text(review)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]

            if prediction == "Positive":
                st.success("Predicted Sentiment: Positive")
            elif prediction == "Negative":
                st.error("Predicted Sentiment: Negative")
            else:
                st.info("Predicted Sentiment: Neutral")

        except FileNotFoundError:
            st.error("Model files not found. Run the notebook first.")
