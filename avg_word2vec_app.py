import streamlit as st
import pickle
import time
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup


# --- Model Loading ---
@st.cache_resource
def load_w2v_artifacts():
    """Loads the pre-trained Word2Vec and classifier models."""
    try:
        w2v_model = Word2Vec.load("word2vec.model")
        with open('model_w2v.pickle', 'rb') as f:
            classifier = pickle.load(f)
        return w2v_model, classifier
    except FileNotFoundError:
        st.error("Model files not found. Ensure 'word2vec.model' and 'model_w2v.pickle' are present.")
        return None, None

w2v_model, classifier = load_w2v_artifacts()

# --- Text Preprocessing ---
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
def preprocess_text(text):
    """Cleans, tokenizes, and preprocesses the input text for sentiment analysis."""
    review = text.lower()
    review = BeautifulSoup(review, "html.parser").get_text()
    review = re.sub(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)", " ", review)
    review = re.sub(r'[^a-zA-Z0-9\s-]', '', review)
    tokens = [word for word in review.split() if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens]
    return lemmatized_tokens

# --- Vectorization ---
def get_average_vector(tokens, model):
    """Calculates the average vector for a list of tokens."""
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# --- Streamlit App Interface ---
def main():
    st.set_page_config(page_title="Kindle Sentiment Analyzer (Word2Vec)", page_icon="🧠", layout="centered")
    st.title("📖 Kindle Review Sentiment Analyzer (Word2Vec)")
    st.markdown("Enter a Kindle review to classify its sentiment using an Average Word2Vec model.")
    
    message_input = st.text_area("Enter review text here:", "The story was captivating and the characters were well-developed.", height=150)

    if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
        if message_input and w2v_model and classifier:
            with st.spinner('Analyzing...'):
                tokens = preprocess_text(message_input)
                avg_vector = get_average_vector(tokens, w2v_model).reshape(1, -1)
                
                prediction = classifier.predict(avg_vector)[0]
                prediction_proba = classifier.predict_proba(avg_vector)

                st.subheader("Sentiment Analysis Result")
                if prediction == 1: # 1 for Positive
                    positive_probability = prediction_proba[0][1]
                    st.success("This review seems Positive.", icon="😊")
                    st.progress(positive_probability)
                    st.write(f"**Confidence:** {positive_probability*100:.2f}%")
                else: # 0 for Negative
                    negative_probability = prediction_proba[0][0]
                    st.error("This review seems Negative.", icon="😠")
                    st.progress(negative_probability)
                    st.write(f"**Confidence:** {negative_probability*100:.2f}%")
        elif not message_input:
            st.warning("Please enter a review to analyze.", icon="⚠️")

if __name__ == '__main__':
    if w2v_model and classifier:
        main()
