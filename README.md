# 📄 Document Classifier with Active Learning & Explainable AI (LIME & LOO)

A premium, interactive Machine Learning web application that demonstrates **Active Learning** (Human-in-the-Loop) and **Explainable AI (XAI)** on the 20 Newsgroups text classification dataset. Built with Python, Streamlit, Scikit-Learn, Sentence-Transformers, NLTK, and Altair.

---

## 🌟 Key Features

### 1. 🔮 Explainable Document Classification
*   **Feature Representation Toggle:** Toggle dynamically between classic sparse **TF-IDF Baseline** features and dense **Sentence Embeddings (MiniLM)** semantic features.
*   **Dual Explainability Engines:**
    *   *TF-IDF Baseline:* Calculates word contributions based on token TF-IDF weights and model parameters (coefficients/log-probabilities).
    *   *Sentence Embeddings:* Utilizes a custom **Leave-One-Out (LOO)** perturbation-based approach to dynamically highlight word contributions by measuring prediction probability drops when individual tokens are removed.
*   **Intuitive Visual Coding:**
    *   **Vibrant Green:** Words that drove the classification *towards* the predicted class (positive influence).
    *   **Vibrant Red:** Words that drove the classification *away* from the predicted class (negative influence).
*   **Interactive Tooltips:** Hover over any highlighted word to inspect its numeric contribution score.

### 2. 🛠️ Classifier Selection & Hyperparameter Tuning (Sidebar)
*   **Algorithm Selector:** Toggle between three classic classifiers: **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM (SGD)**.
*   **Embedding Compatibility Filtering:** Since dense sentence embeddings contain negative values, **Multinomial Naive Bayes is automatically disabled** and a helpful notice is shown when "Sentence Embeddings" is active.
*   **Hyperparameter Sliders:** Tune regularization strength ($C$ for Logistic Regression, $\alpha$ for SVM) and Laplace smoothing ($\alpha$ for Naive Bayes) in real-time.
*   **Live Metrics Evaluation:** Adjusting configurations retrains the model dynamically on the active features, instantly updating the test accuracy card.

### 3. ⚙️ Interactive Active Learning Studio (Human-in-the-Loop)
*   **Feature Set Synchronized:** Automatically matches the selected feature set (TF-IDF sparse matrices or MiniLM dense embeddings arrays).
*   **Auto-Reset:** Automatically resets the labeling history and pool state when toggling between TF-IDF and Sentence Embeddings to prevent shape mismatch errors.
*   **Uncertainty Sampling Strategies:** Select how to sample documents using **Entropy** or **Margin** uncertainty, or **Random** selection.
*   **Live Retraining:** Submit the correct category label. The selected model instantly retrains on the updated training pool and plots its live accuracy gains on the chart.

### 4. 🔍 Category Feature Explorer
*   **Keyword Association Charts:** Select any of the 20 categories to inspect the **top 15 most indicative words** learned by the model (active for TF-IDF representations).
*   **Latent Space Explanations:** Displays an educational explanation when Sentence Embeddings are active, describing why latent semantic dimensions cannot be directly mapped back to raw keywords.

### 5. 📈 Advanced Interactive Analytics (Altair)
*   **Interactive Learning Curves:** Compare Active Learning vs. Random Sampling, overlaid with your interactive live labeling session history.
*   **Confusion Matrix Heatmap:** Hover over the interactive 20x20 grid to see actual vs. predicted counts and identify common model confusions.

### 6. 📂 Bulk Document Classification
*   **CSV File Upload:** Process batches of documents by uploading a CSV and selecting the text column.
*   **Dual Batch Inference:**
    *   *TF-IDF:* Preprocesses and classifies documents row-by-row.
    *   *Sentence Embeddings:* Performs fast batch embedding and classification using Sentence Transformers to maximize throughput and performance.
*   **Distribution & Export:** View live progress bars, prediction previews, predicted class distribution charts, and export results as a new CSV.

### 7. 🎨 Modern Design Aesthetics
*   **Custom Typography & CSS Cards:** Integrates Google's *Outfit* font and applies modern shadow cards that float upward on hover with active border-color highlights.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rogueslasher/document_classifier.git
    cd document_classifier
    ```

2.  **Activate the Virtual Environment:**
    *   On Windows:
        ```powershell
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirement.txt
    ```

4.  **Run the Web App:**
    ```bash
    streamlit run app.py
    ```

---

## 🔬 Model Details
*   **Feature Representations:**
    *   *TF-IDF Baseline:* Sparse representation (Max features: 5,000, unigrams, minimum document frequency = 5).
    *   *Sentence Embeddings:* Dense representation (384-dimensional vectors from `all-MiniLM-L6-v2` Sentence Transformer, cached locally to `X_train_emb.npy` and `X_test_emb.npy` for millisecond-level load times).
*   **Classifiers:** Logistic Regression, Linear SVM (SGD), and Multinomial Naive Bayes (TF-IDF only).
*   **Dataset:** 20 Newsgroups (10,994 total documents split 80/20 into train/test sets).
