import streamlit as st
import pickle
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def load_nltk_data():
    download_nltk_resources()
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

    # Inject Custom CSS for modern design aesthetics
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
            
            /* Apply global Outfit font */
            html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', sans-serif !important;
            }
            
            /* Custom styled metric card wrappers */
            div[data-testid="stMetric"] {
                background-color: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 12px !important;
                padding: 15px 20px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
                transition: transform 0.2s, box-shadow 0.2s !important;
            }
            
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
                border-color: #3b82f6 !important;
            }

            /* Explicitly style metric labels and values to guarantee readability */
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                color: #0f172a !important;
            }
            div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
                color: #475569 !important;
            }
            div[data-testid="stMetric"] label {
                color: #475569 !important;
            }

            /* Custom styled main header */
            .main-title {
                font-size: 2.5rem !important;
                font-weight: 700 !important;
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1.5rem;
                display: inline-block;
            }
        </style>
    """, unsafe_allow_html=True)

    stop_words, lemmatizer = load_nltk_data()
    model, vectorizer, info, results = load_models()
    categories = info['categories']
    improvement = results['improvement']

    st.markdown('<h1 class="main-title">📄 Document Classification with Active Learning</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.header("📊 Project Stats")
    st.sidebar.metric("Categories", len(categories))
    st.sidebar.metric("Training Samples", f"{info['training_samples']:,}")
    st.sidebar.metric("Final Accuracy", f"{info['accuracy']*100:.2f}%")
    st.sidebar.metric("Vocabulary Size", f"{info['vocabulary_size']:,}")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Active Learning Advantage")
    st.sidebar.metric("Accuracy Gain", f"+{improvement:.2f}pp")

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Classify Text", "⚙️ Labeling Studio", "📈 Model Performance", "ℹ️ About"])

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
                    import altair as alt
                    prob_df = pd.DataFrame({
                        'Category': categories,
                        'Probability': probabilities * 100
                    }).sort_values('Probability', ascending=False)

                    top_prob_df = prob_df.head(10).copy()
                    chart = alt.Chart(top_prob_df).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('Probability:Q', title='Confidence (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Category:N', sort='-x', title='Category'),
                        color=alt.Color('Probability:Q', scale=alt.Scale(scheme='blues', reverse=True), legend=None),
                        tooltip=[
                            alt.Tooltip('Category:N', title='Category'),
                            alt.Tooltip('Probability:Q', format='.2f', title='Probability (%)')
                        ]
                    ).properties(
                        title="Top 10 Category Probabilities",
                        height=300
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Explain prediction - Word Highlights
                from src.utils import explain_prediction, get_highlighted_html
                with st.spinner("Analyzing feature contributions..."):
                    word_contributions = explain_prediction(
                        user_input, vectorizer, model, prediction, stop_words, lemmatizer
                    )
                    highlighted_html = get_highlighted_html(
                        user_input, word_contributions, lemmatizer
                    )

                st.markdown("---")
                st.markdown("### 🔍 Model Explanation (Word Contributions)")
                st.markdown(
                    "The highlighted words below drove the model's classification decision. "
                    "Hover over any highlighted word to inspect its numeric contribution score. "
                    "**Green** words increased confidence in this category; **red** words decreased it."
                )
                
                # HTML Container
                st.markdown(
                    f'<div style="background-color: #f8f9fa; color: #0f172a; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; font-family: monospace; line-height: 1.8; white-space: pre-wrap; font-size: 15px;">{highlighted_html}</div>',
                    unsafe_allow_html=True
                )

        # Batch classification section
        st.markdown("---")
        st.subheader("📂 Batch Classification (CSV Upload)")
        st.markdown(
            "Upload a CSV file containing multiple documents. Select the column containing "
            "the text, and download the prediction results as a new CSV file."
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            columns = df_upload.columns.tolist()
            text_col = st.selectbox("Select the column containing document text:", columns)

            if st.button("🚀 Classify Batch", type="primary"):
                if df_upload[text_col].isnull().any():
                    st.warning("Warning: Selected column contains empty/null rows. These will be classified as empty.")

                total_rows = len(df_upload)
                preds = []
                confidences = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in df_upload.iterrows():
                    text_val = str(row[text_col])
                    if not text_val.strip() or text_val == "nan":
                        preds.append("N/A")
                        confidences.append(0.0)
                    else:
                        cleaned_val = preprocess_text(text_val, stop_words, lemmatizer)
                        X_val = vectorizer.transform([cleaned_val])
                        pred_val = model.predict(X_val)[0]
                        prob_val = model.predict_proba(X_val)[0][pred_val]

                        preds.append(categories[pred_val])
                        confidences.append(prob_val * 100)

                    progress = (idx + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Processing row {idx + 1}/{total_rows}...")

                progress_bar.empty()
                status_text.success("Batch classification complete!")

                df_results = df_upload.copy()
                df_results['Predicted Category'] = preds
                df_results['Confidence (%)'] = confidences

                col_stats1, col_stats2 = st.columns([1, 1])
                with col_stats1:
                    st.markdown("#### Preview Predictions")
                    st.dataframe(df_results.head(10))

                with col_stats2:
                    st.markdown("#### Category Distribution")
                    dist_df = df_results['Predicted Category'].value_counts().reset_index()
                    dist_df.columns = ['Category', 'Count']

                    import altair as alt
                    dist_chart = alt.Chart(dist_df).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('Count:Q', title='Document Count'),
                        y=alt.Y('Category:N', sort='-x', title='Category'),
                        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues', reverse=True), legend=None),
                        tooltip=['Category', 'Count']
                    ).properties(height=250)
                    st.altair_chart(dist_chart, use_container_width=True)

                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="classified_documents.csv",
                    mime="text/csv"
                )

    with tab2:
        st.header("⚙️ Active Learning Labeling Studio")
        st.markdown(
            "Act as the **Human Oracle**! Label the documents that the model is **most uncertain** about "
            "based on your selected sampling strategy. The model will retrain instantly with your feedback."
        )

        # Initialize AL session state if not already done
        if 'al_initialized' not in st.session_state:
            with st.spinner("Initializing Active Learning Studio (loading datasets)..."):
                from src.utils import load_raw_train_test_data
                
                # Load full matrices
                with open('train_test_data.pkl', 'rb') as f:
                    full_data = pickle.load(f)
                
                st.session_state['X_train_full'] = full_data['X_train']
                st.session_state['y_train_full'] = full_data['y_train']
                st.session_state['X_test'] = full_data['X_test']
                st.session_state['y_test'] = full_data['y_test']
                
                # Load raw text data
                df_train, df_test = load_raw_train_test_data()
                st.session_state['df_train'] = df_train
                
                # Pick 500 random samples as the starting labeled set
                np.random.seed(42)
                initial_indices = np.random.choice(len(df_train), size=500, replace=False)
                pool_indices = [i for i in range(len(df_train)) if i not in initial_indices]
                
                # Slice matrices
                X_labeled = st.session_state['X_train_full'][initial_indices]
                y_labeled = st.session_state['y_train_full'][initial_indices]
                
                # Train baseline model for active learning
                model_al = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
                model_al.fit(X_labeled, y_labeled)
                
                # Evaluate
                y_pred = model_al.predict(st.session_state['X_test'])
                acc = accuracy_score(st.session_state['y_test'], y_pred)
                f1 = f1_score(st.session_state['y_test'], y_pred, average='weighted')
                
                st.session_state['al_X_train'] = X_labeled
                st.session_state['al_y_train'] = y_labeled
                st.session_state['al_pool_indices'] = pool_indices
                st.session_state['al_model'] = model_al
                st.session_state['al_history'] = [{'n_labeled': 500, 'accuracy': acc, 'f1': f1}]
                st.session_state['al_current_sample_idx'] = None
                st.session_state['al_labeled_count'] = 0
                st.session_state['al_initialized'] = True

        # Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
        with col_ctrl1:
            strategy = st.selectbox(
                "Sampling Strategy",
                ["Entropy", "Margin", "Random"],
                help="Entropy: Selects sample with highest prediction uncertainty across all classes.\n"
                     "Margin: Selects sample with smallest confidence margin between top 2 classes.\n"
                     "Random: Selects a random sample from the pool."
            )
        with col_ctrl2:
            st.metric("Samples Labeled in Session", st.session_state['al_labeled_count'])
        with col_ctrl3:
            current_acc = st.session_state['al_history'][-1]['accuracy'] * 100
            initial_acc = st.session_state['al_history'][0]['accuracy'] * 100
            gain = current_acc - initial_acc
            st.metric(
                "Live Model Accuracy", 
                f"{current_acc:.2f}%", 
                delta=f"+{gain:.2f}pp" if gain >= 0 else f"{gain:.2f}pp"
            )

        # Select sample
        if st.session_state['al_current_sample_idx'] is None:
            if len(st.session_state['al_pool_indices']) > 0:
                X_train_full = st.session_state['X_train_full']
                pool_indices = st.session_state['al_pool_indices']
                model_al = st.session_state['al_model']
                
                if strategy == "Random":
                    next_idx = int(np.random.choice(pool_indices))
                else:
                    X_pool = X_train_full[pool_indices]
                    probas = model_al.predict_proba(X_pool)
                    
                    if strategy == "Entropy":
                        entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
                        relative_idx = np.argmax(entropy)
                    elif strategy == "Margin":
                        sorted_probas = np.sort(probas, axis=1)
                        margin = sorted_probas[:, -1] - sorted_probas[:, -2]
                        relative_idx = np.argmin(margin)
                    
                    next_idx = pool_indices[relative_idx]
                
                st.session_state['al_current_sample_idx'] = next_idx
            else:
                st.info("Congratulations! All pool samples have been labeled.")
                st.session_state['al_current_sample_idx'] = None

        # Display active sample
        current_idx = st.session_state['al_current_sample_idx']
        if current_idx is not None:
            df_train = st.session_state['df_train']
            raw_text = df_train.iloc[current_idx]['text']
            true_cat = df_train.iloc[current_idx]['category']
            
            X_sample = st.session_state['X_train_full'][current_idx]
            model_al = st.session_state['al_model']
            pred_idx = model_al.predict(X_sample)[0]
            pred_prob = model_al.predict_proba(X_sample)[0][pred_idx] * 100
            pred_cat = categories[pred_idx]

            st.markdown("### 📄 Document to Label")
            
            st.text_area(
                "Document Text (Unlabeled)",
                value=raw_text,
                height=250,
                disabled=True
            )

            st.markdown(f"🤖 **Model's Current Prediction:** `{pred_cat}` (Confidence: {pred_prob:.1f}%)")

            try:
                default_sel_idx = categories.index(pred_cat)
            except ValueError:
                default_sel_idx = 0

            with st.form("labeling_form", clear_on_submit=True):
                selected_label_str = st.selectbox(
                    "Assign the correct category:",
                    categories,
                    index=default_sel_idx
                )
                
                submitted = st.form_submit_button("🚀 Submit Label & Retrain Model", type="primary")
                
                if submitted:
                    selected_label_idx = categories.index(selected_label_str)
                    
                    st.session_state['al_X_train'] = vstack([st.session_state['al_X_train'], X_sample])
                    st.session_state['al_y_train'] = np.append(st.session_state['al_y_train'], selected_label_idx)
                    st.session_state['al_pool_indices'].remove(current_idx)
                    
                    st.session_state['al_model'].fit(st.session_state['al_X_train'], st.session_state['al_y_train'])
                    
                    y_pred = st.session_state['al_model'].predict(st.session_state['X_test'])
                    new_acc = accuracy_score(st.session_state['y_test'], y_pred)
                    new_f1 = f1_score(st.session_state['y_test'], y_pred, average='weighted')
                    
                    st.session_state['al_history'].append({
                        'n_labeled': len(st.session_state['al_y_train']),
                        'accuracy': new_acc,
                        'f1': new_f1
                    })
                    
                    st.session_state['al_labeled_count'] += 1
                    st.session_state['al_current_sample_idx'] = None
                    
                    st.success(f"Feedback recorded! Retrained model. New Accuracy: {new_acc*100:.2f}%")
                    st.rerun()

    with tab3:
        st.header("📈 Model Performance Analysis")
        st.markdown(
            "Compare the learning efficiency of **Active Learning** vs. **Random Sampling**. "
            "If you have labeled samples in this session, your **live session** performance will also be shown!"
        )

        al_df = results['active_learning'].copy()
        random_df = results['random_sampling'].copy()

        al_df['Strategy'] = 'Active Learning (Pre-calculated)'
        al_df['Accuracy (%)'] = al_df['accuracy'] * 100
        al_df['F1-Score (%)'] = al_df['f1'] * 100

        random_df['Strategy'] = 'Random Sampling (Pre-calculated)'
        random_df['Accuracy (%)'] = random_df['accuracy'] * 100
        random_df['F1-Score (%)'] = random_df['f1'] * 100

        combined_df = pd.concat([al_df, random_df], ignore_index=True)

        if 'al_history' in st.session_state and len(st.session_state['al_history']) > 1:
            session_df = pd.DataFrame(st.session_state['al_history'])
            session_df['Strategy'] = 'Interactive Session (Live)'
            session_df['Accuracy (%)'] = session_df['accuracy'] * 100
            session_df['F1-Score (%)'] = session_df['f1'] * 100
            combined_df = pd.concat([combined_df, session_df], ignore_index=True)

        col1, col2 = st.columns(2)

        with col1:
            acc_chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X('n_labeled:Q', title='Number of Labeled Samples'),
                y=alt.Y('Accuracy (%):Q', title='Accuracy (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Strategy:N', scale=alt.Scale(
                    domain=['Active Learning (Pre-calculated)', 'Random Sampling (Pre-calculated)', 'Interactive Session (Live)'],
                    range=['#3b82f6', '#9ca3af', '#10b981']
                )),
                tooltip=['Strategy', 'n_labeled', alt.Tooltip('Accuracy (%):Q', format='.2f')]
            ).properties(
                title="Model Accuracy Comparison",
                height=350
            ).interactive()
            st.altair_chart(acc_chart, use_container_width=True)

        with col2:
            f1_chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X('n_labeled:Q', title='Number of Labeled Samples'),
                y=alt.Y('F1-Score (%):Q', title='F1-Score (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Strategy:N', scale=alt.Scale(
                    domain=['Active Learning (Pre-calculated)', 'Random Sampling (Pre-calculated)', 'Interactive Session (Live)'],
                    range=['#3b82f6', '#9ca3af', '#10b981']
                )),
                tooltip=['Strategy', 'n_labeled', alt.Tooltip('F1-Score (%):Q', format='.2f')]
            ).properties(
                title="Model F1-Score Comparison",
                height=350
            ).interactive()
            st.altair_chart(f1_chart, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Confusion Matrix Heatmap")
        st.markdown("Hover over the squares to inspect how categories are confused by the baseline model on the test set.")

        with st.spinner("Computing Confusion Matrix..."):
            from sklearn.metrics import confusion_matrix
            if 'X_test' not in st.session_state:
                with open('train_test_data.pkl', 'rb') as f:
                    full_data = pickle.load(f)
                st.session_state['X_test'] = full_data['X_test']
                st.session_state['y_test'] = full_data['y_test']

            X_test_mat = st.session_state['X_test']
            y_test_lbl = st.session_state['y_test']

            y_pred_base = model.predict(X_test_mat)
            cm = confusion_matrix(y_test_lbl, y_pred_base)

            cm_data = []
            for i in range(len(categories)):
                for j in range(len(categories)):
                    cm_data.append({
                        'Actual': categories[i],
                        'Predicted': categories[j],
                        'Count': int(cm[i, j])
                    })
            cm_df = pd.DataFrame(cm_data)

            cm_chart = alt.Chart(cm_df).mark_rect().encode(
                x=alt.X('Predicted:N', title='Predicted Category', sort=categories, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Actual:N', title='Actual Category', sort=categories),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Number of Docs'),
                tooltip=['Actual', 'Predicted', 'Count']
            ).properties(
                title="Confusion Matrix Heatmap (Baseline Model)",
                height=500
            )
            st.altair_chart(cm_chart, use_container_width=True)

    with tab4:
        st.header("About")
        st.write(
            "This project demonstrates Active Learning for text classification "
            "using Logistic Regression and TF-IDF features on the 20 Newsgroups dataset."
        )



if __name__ == "__main__":
    main()
