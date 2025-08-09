import streamlit as st
import pickle
import time
import re
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup


# --- Model and Vectorizer Loading ---
@st.cache_resource
def load_artifacts():
    """Loads the pre-trained BoW model and vectorizer for sentiment analysis."""
    try:
        with open('BoW.pickle', 'rb') as f:
            vectorizer = pickle.load(f)
        # Corrected to load the model file from your notebook
        with open('bow_model.pickle', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model/vectorizer files not found. Ensure 'BoW.pickle' and 'model.pickle' are present.")
        return None, None

vectorizer, model = load_artifacts()

# --- Text Preprocessing ---
# This function should be identical to the one in your notebook
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses the input text for sentiment analysis."""
    # Convert to lowercase
    review = text.lower()
    # Remove HTML tags
    review = BeautifulSoup(review, "html.parser").get_text()
    # Remove emails and hyperlinks
    review = re.sub(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)", " ", review)
    # Keep only alphabets, numbers, hyphens, and spaces
    review = re.sub(r'[^a-zA-Z0-9\s-]', '', review)
    # Tokenize and remove stopwords
    tokens = [word for word in review.split() if word not in stop_words]
    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens]
    # Join tokens back into a string
    return " ".join(lemmatized_tokens)

# --- Streamlit App Interface ---
def main():
    st.set_page_config(page_title="Kindle Sentiment Analyzer", page_icon="📖", layout="centered")
    st.title("📖 Kindle Review Sentiment Analyzer")
    st.markdown("Enter a Kindle review to classify its sentiment as **Positive** or **Negative**.")
    
    message_input = st.text_area("Enter review text here:", "Great short read. I didn't want to put it down.", height=150)

    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if message_input and vectorizer and model:
            with st.spinner('Analyzing...'):
                processed_message = preprocess_text(message_input)
                message_vectorized = vectorizer.transform([processed_message])
                prediction = model.predict(message_vectorized)[0]
                prediction_proba = model.predict_proba(message_vectorized)

                st.subheader("Sentiment Analysis Result")
                if prediction == 1: # 1 for Positive sentiment
                    positive_probability = prediction_proba[0][1]
                    st.success("This review seems Positive.", icon="😊")
                    st.progress(positive_probability)
                    st.write(f"**Confidence:** {positive_probability*100:.2f}%")
                else: # 0 for Negative sentiment
                    negative_probability = prediction_proba[0][0]
                    st.error("This review seems Negative.", icon="😠")
                    st.progress(negative_probability)
                    st.write(f"**Confidence:** {negative_probability*100:.2f}%")
        elif not message_input:
            st.warning("Please enter a review to analyze.", icon="⚠️")

if __name__ == '__main__':
    if vectorizer and model:
        main()
