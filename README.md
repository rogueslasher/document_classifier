# Document Classifier with Active Learning & Local Explanations

This is a Streamlit web application that demonstrates **Active Learning** (Human-in-the-Loop) and **Explainable AI (XAI)** on the 20 Newsgroups text classification dataset. The project supports both classic sparse TF-IDF features and dense semantic sentence embeddings.

---

## 🌟 Key Features

### 1. Local Word Explanations
*   **TF-IDF Explanations:** Highlights word contributions using model coefficients (or log-odds ratios for Naive Bayes).
*   **Embedding Explanations:** Uses a custom **Leave-One-Out (LOO)** perturbation method. It sequentially removes each token from the text and measures the drop in prediction probability to calculate word importance.
*   **Visual Highlights:** Green words increase confidence in the predicted class, while red words decrease it. You can hover over any highlighted word to inspect its numeric contribution score.

### 2. Live Model Configuration
*   **Multiple Classifiers:** Swap between **Logistic Regression**, **Multinomial Naive Bayes**, and a **Linear SVM** (trained via SGD with modified Huber loss).
*   **Hyperparameter Tuning:** Adjust parameters (like regularization strength $C$ or Laplace smoothing $\alpha$) using sidebar sliders and watch the model retrain in milliseconds.
*   **Embedding Constraints:** The app automatically disables Multinomial Naive Bayes when using dense sentence embeddings because embeddings contain negative values.

### 3. Active Learning Labeling Studio
*   **Human-in-the-Loop:** Act as the oracle to label documents that the model is most uncertain about.
*   **Sampling Strategies:** Choose between **Entropy**, **Margin** uncertainty, or **Random** sampling.
*   **Live Retraining:** Once you submit a label, the model retrains instantly, updates the metrics, and plots your progress against pre-calculated benchmarks.
*   **Auto-Reset:** Switching feature sets automatically resets the labeling studio to avoid shape mismatch issues.

### 4. Category Feature Explorer
*   **Word Associations:** Select a class to see the top 15 words most strongly associated with it based on model coefficients (available when TF-IDF is active).
*   **Latent Space Explanation:** Shows an educational notice when embeddings are active, explaining why 384-dimensional dense semantic vectors cannot be mapped directly back to single words.

### 5. Batch Classification
*   **CSV Upload:** Process batches of documents by uploading a CSV.
*   **Optimized Inference:** Uses fast batch encoding for Sentence Transformers to maximize processing speed.
*   **Export:** Preview predictions, inspect the predicted category distribution, and download the results as a CSV.

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

## 🔬 Technical Details

*   **Dataset:** 20 Newsgroups corpus (10,994 documents split 80/20 into train/test sets).
*   **Feature Representations:**
    *   *TF-IDF Baseline:* Sparse vectors (5,000 max features, unigrams, min document frequency = 5).
    *   *Sentence Embeddings:* Dense vectors from `all-MiniLM-L6-v2` (cached locally to `X_train_emb.npy` and `X_test_emb.npy` after the first run to allow instant subsequent loading).
*   **Models:** Logistic Regression (L-BFGS), SGDClassifier (linear SVM), and Multinomial Naive Bayes.
