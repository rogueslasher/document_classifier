# 📄 Document Classifier with Active Learning & Explainable AI (LIME)

A premium, interactive Machine Learning web application that demonstrates **Active Learning** (Human-in-the-Loop) and **Explainable AI (XAI)** on the 20 Newsgroups text classification dataset. Built with Python, Streamlit, Scikit-Learn, NLTK, and Altair.

---

## 🌟 Key Features

### 1. 🔮 Explainable Document Classification
*   **Real-time Inference:** Paste any text and classify it into one of 20 categories using the dynamically configured model.
*   **Generalized Explainability Highlights:** Uses TF-IDF token weights and model parameters to calculate word contributions. Supports standard coefficients for *Logistic Regression* & *SVM*, and log-odds ratios vs. other classes for *Multinomial Naive Bayes*:
    *   **Vibrant Green:** Words that drove the classification *towards* the predicted class (positive influence).
    *   **Vibrant Red:** Words that drove the classification *away* from the predicted class (negative influence).
*   **Interactive Tooltips:** Hover over any highlighted word to inspect its numeric contribution score.

### 2. 🛠️ Classifier Selection & Hyperparameter Tuning (Sidebar)
*   **Algorithm Selector:** Toggle between three classic classifiers: **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM (SGD)**.
*   **Hyperparameter Sliders:** Tune regularization strength ($C$ for Logistic Regression, $\alpha$ for SVM) and Laplace smoothing ($\alpha$ for Naive Bayes) in real-time.
*   **Live Metrics Evaluation:** Adjusting sliders retrains the model dynamically on 8,700+ samples in milliseconds, instantly updating the test accuracy card.

### 3. ⚙️ Interactive Active Learning Studio (Human-in-the-Loop)
*   **Unlabeled Pool:** Queries the raw unlabeled training corpus in the background.
*   **Uncertainty Sampling Strategies:** Select how to sample documents using **Entropy** or **Margin** uncertainty, or **Random** selection.
*   **Live Retraining:** Submit the correct category label. The selected model instantly retrains on the updated training pool and plots its live accuracy gains on the chart.

### 4. 🔍 Category Feature Explorer
*   **Keyword Association Charts:** Select any of the 20 categories to inspect the **top 15 most indicative words** learned by the model.
*   **Altair Visualizations:** Displays an interactive horizontal bar chart of the word coefficients or log probabilities.

### 5. 📈 Advanced Interactive Analytics (Altair)
*   **Interactive Learning Curves:** Compare Active Learning vs. Random Sampling, overlaid with your interactive live labeling session history.
*   **Confusion Matrix Heatmap:** Hover over the interactive 20x20 grid to see actual vs. predicted counts and identify common model confusions.

### 6. 📂 Bulk Document Classification
*   **CSV File Upload:** Process batches of documents by uploading a CSV and selecting the text column.
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
*   **Features:** TF-IDF Vectorizer (Max features: 5,000, unigrams, minimum document frequency = 5).
*   **Classifier:** Logistic Regression (L-BFGS solver, multi-class, L2 regularization).
*   **Dataset:** 20 Newsgroups (10,994 total documents split 80/20 into train/test sets).
