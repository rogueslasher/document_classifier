import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return stopwords.words('english'), WordNetLemmatizer()


@st.cache_resource
def load_models():
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('deployment_info.pkl', 'rb') as f:
        info = pickle.load(f)
    with open('active_learning_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return model, vectorizer, info, results


def preprocess_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def main():
    st.set_page_config(
        page_title="Document Classifier with Active Learning",
        page_icon="📄",
        layout="wide"
    )

    stop_words, lemmatizer = load_nltk_data()
    model, vectorizer, info, results = load_models()
    categories = info['categories']
    improvement = results['improvement']

    st.title("📄 Document Classification with Active Learning")
    st.markdown("---")

    st.sidebar.header("📊 Project Stats")
    st.sidebar.metric("Categories", len(categories))
    st.sidebar.metric("Training Samples", f"{info['training_samples']:,}")
    st.sidebar.metric("Final Accuracy", f"{info['accuracy']*100:.2f}%")
    st.sidebar.metric("Vocabulary Size", f"{info['vocabulary_size']:,}")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Active Learning Advantage")
    st.sidebar.metric("Accuracy Gain", f"+{improvement:.2f}pp")

    tab1, tab2, tab3 = st.tabs(["🔮 Classify Text", "📈 Model Performance", "ℹ️ About"])

    with tab1:
        st.header("Classify a Document")

        user_input = st.text_area(
            "Enter text to classify:",
            height=200,
            placeholder="Paste your document here..."
        )

        if st.button("🚀 Classify", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                cleaned = preprocess_text(user_input, stop_words, lemmatizer)
                X = vectorizer.transform([cleaned])
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]

                st.success("Classification Complete")

                col1, col2 = st.columns([1, 2])

                with col1:
                    confidence = probabilities[prediction] * 100
                    st.markdown(f"### {categories[prediction]}")
                    st.metric("Confidence", f"{confidence:.1f}%")

                with col2:
                    prob_df = pd.DataFrame({
                        'Category': categories,
                        'Probability': probabilities * 100
                    }).sort_values('Probability', ascending=False)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(prob_df['Category'][:10], prob_df['Probability'][:10])
                    ax.set_xlabel("Probability (%)")
                    ax.set_title("Top 10 Categories")
                    st.pyplot(fig)

    with tab2:
        al_df = results['active_learning']
        random_df = results['random_sampling']

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(al_df['n_labeled'], al_df['accuracy']*100)
            ax.plot(random_df['n_labeled'], random_df['accuracy']*100)
            ax.set_xlabel("Labeled Examples")
            ax.set_ylabel("Accuracy (%)")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(al_df['n_labeled'], al_df['f1']*100)
            ax.plot(random_df['n_labeled'], random_df['f1']*100)
            ax.set_xlabel("Labeled Examples")
            ax.set_ylabel("F1-Score (%)")
            st.pyplot(fig)

    with tab3:
        st.header("About")
        st.write(
            "This project demonstrates Active Learning for text classification "
            "using Logistic Regression and TF-IDF features on the 20 Newsgroups dataset."
        )


if __name__ == "__main__":
    main()
